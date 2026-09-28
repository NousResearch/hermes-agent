import { atom, computed } from 'nanostores'

import { readJson, writeKey } from '@/lib/storage'
import { normalize } from '@/lib/text'

import { $rightRailActiveTabId, type RightRailTabId, selectRightRailTab } from './layout'
import { clearExplicitPreviewOpen, noteExplicitPreviewOpen } from './preview-explicit'
import { normalizeProfileKey } from './profile'
import { canOpenBrowserWindow, openBrowserInNewWindow } from './windows'

/**
 * PREVIEW RAIL — one list of tabs, one way in.
 *
 * Everything the rail can show is a `PreviewTarget` in `$previewTabs`: a file
 * on disk, a live URL, or a generated artifact. There is no privileged "live
 * preview" slot alongside the tabs; `openPreview` is the only entry point, so
 * a tool result, a file-browser click, and an artifact card all travel the
 * same road and behave identically once open.
 *
 * Tabs are global and outlive the session that created them, like tabs
 * anywhere else — they close when you close them.
 */

/** How an HTML file target shows: the live page, or its source. */
export type PreviewRenderMode = 'preview' | 'source'

export interface PreviewTarget {
  binary?: boolean
  byteSize?: number
  /** Inline image bytes (a `data:` URL) when the renderer already holds them —
   * e.g. a pasted/dropped screenshot whose only on-disk copy is a transient
   * path the preview can't reliably re-read. Rendered directly and NOT
   * persisted (it would bloat localStorage). */
  dataUrl?: string
  /** `artifact` targets have nothing behind them on disk or on the network —
   * `url` is an id into the artifact registry, which owns the content. They
   * are what lets the rail preview generated HTML the workspace never saw. */
  kind: 'artifact' | 'file' | 'url'
  label: string
  large?: boolean
  language?: string
  mimeType?: string
  path?: string
  previewKind?: 'binary' | 'html' | 'image' | 'pdf' | 'text'
  renderMode?: PreviewRenderMode
  source: string
  /** Runtime-only target that cannot be restored from persisted state. */
  transient?: boolean
  url: string
}

export interface PreviewServerRestart {
  message?: string
  status: 'complete' | 'error' | 'running'
  taskId: string
  url: string
}

export interface PreviewTab {
  id: RightRailTabId
  target: PreviewTarget
}

const TABS_STORAGE_KEY = 'hermes.desktop.previewTabs.v2'
/** Superseded by the tab list above; cleared so it can't leak forever. */
const LEGACY_SESSION_REGISTRY_KEY = 'hermes.desktop.sessionPreviews.v1'

function isPreviewTarget(value: unknown): value is PreviewTarget {
  if (!value || typeof value !== 'object') {
    return false
  }

  const r = value as Record<string, unknown>

  return (
    (r.kind === 'artifact' || r.kind === 'file' || r.kind === 'url') &&
    typeof r.label === 'string' &&
    typeof r.source === 'string' &&
    typeof r.url === 'string'
  )
}

// Artifact tabs are never written (their registry is memory-only), so a
// restored artifact row is stale storage — drop it rather than reviving a tab
// with nothing behind it.
function isPreviewTab(value: unknown): value is PreviewTab {
  if (!value || typeof value !== 'object') {
    return false
  }

  const r = value as Record<string, unknown>

  return typeof r.id === 'string' && (r.id.startsWith('file:') || r.id.startsWith('url:')) && isPreviewTarget(r.target)
}

function isPdfFileTarget(target: PreviewTarget): boolean {
  if (target.kind !== 'file') {
    return false
  }

  if (target.mimeType?.toLowerCase() === 'application/pdf') {
    return true
  }

  if ([target.path, target.source].some(value => (value ? /\.pdf$/i.test(value) : false))) {
    return true
  }

  try {
    return /\.pdf$/i.test(new URL(target.url).pathname)
  } catch {
    return false
  }
}

/** Upgrade tabs persisted by builds that classified PDFs as generic binary.
 * Without this restore-time migration, an already-open PDF keeps taking the
 * obsolete raw-binary path after Desktop itself has been upgraded. */
export function decodePreviewTabs(raw: string): PreviewTab[] {
  return parseTabList(JSON.parse(raw) as unknown)
}

function parseTabList(parsed: unknown): PreviewTab[] {
  return (Array.isArray(parsed) ? parsed.filter(isPreviewTab) : []).map(tab =>
    isPdfFileTarget(tab.target) && tab.target.previewKind === 'binary'
      ? { ...tab, target: { ...tab.target, previewKind: 'pdf' as const } }
      : tab
  )
}

/** The tabs a profile's rail is showing, keyed by profile. */
type TabsByProfile = Record<string, PreviewTab[]>

/** Read every profile's bucket. A value written by a build that stored ONE
 *  global array is held back and adopted by the first scope to arrive rather
 *  than dropped — tabs the user can see are the tabs that must survive. */
let pendingLegacyTabs: PreviewTab[] | null = null

function loadTabsByProfile(): TabsByProfile {
  const stored = readJson<unknown>(TABS_STORAGE_KEY)

  if (Array.isArray(stored)) {
    pendingLegacyTabs = parseTabList(stored)

    return {}
  }

  if (!stored || typeof stored !== 'object') {
    return {}
  }

  const byProfile: TabsByProfile = {}

  for (const [key, value] of Object.entries(stored as Record<string, unknown>)) {
    byProfile[normalizeProfileKey(key)] = parseTabList(value)
  }

  return byProfile
}

const tabsByProfile = loadTabsByProfile()

/** Inline bytes are not restorable. Strip them from images, and skip remote
 *  HTML and artifact tabs that cannot render without their in-memory payload. */
function persistableTabs(tabs: PreviewTab[]): PreviewTab[] {
  return tabs.filter(
    tab =>
      tab.target.kind !== 'artifact' &&
      !tab.target.transient &&
      !(tab.target.previewKind === 'html' && tab.target.dataUrl)
  )
}

function persistTabs() {
  const buckets: TabsByProfile = {}

  for (const [key, tabs] of Object.entries(tabsByProfile)) {
    const persistable = persistableTabs(tabs)

    if (persistable.length > 0) {
      buckets[key] = persistable
    }
  }

  // `dataUrl` holds inline bytes that cannot be restored; drop the key wherever
  // it survives the filter above (an image tab). An empty map removes the key
  // rather than storing `{}`, matching the tiles store.
  writeKey(
    TABS_STORAGE_KEY,
    Object.keys(buckets).length === 0
      ? null
      : JSON.stringify(buckets, (key, value) => (key === 'dataUrl' ? undefined : value))
  )
}

// Tabs are scoped to THE CHAT ON SCREEN, not to the window's gateway socket.
// `session-states.ts` resolves the focused session's owner and pushes it here
// via `setPreviewScope`; the two must not be conflated, because a focused tab
// does not swap the socket — every bot chat is served by one pooled backend, so
// a socket-keyed rail showed one agent's preview in every agent's chat. That is
// the same trap `bot-row.tsx` documents for the roster highlight. (It also owns
// the resolver, so it pushes rather than having this module reach for it — this
// file is already imported by session-states.ts.)
//
// Which bucket the atom mirrors. A RENAME moves the view without the scope
// changing, so this has to follow the rename or the persist subscriber below
// would resurrect the bucket the rename just deleted.
let viewKey = 'default'

export const $previewTabs = atom<PreviewTab[]>([])

// Adoption phase: emissions that carry storage THIS MODULE JUST READ, not a
// change. nanostores' subscribe fires immediately, and writing what was just
// read back is a data-loss clobber: every renderer boots against storage it
// has not adopted yet, and echoing the empty view back overwrites the real
// record before adoption can read it. A legacy single-array store is wiped
// this way before `pendingLegacyTabs` is ever adopted; a bucket store loses
// its `default` bucket the same way.
let adoptingStoredTabs = true

$previewTabs.subscribe(tabs => {
  if (adoptingStoredTabs) {
    return
  }

  // `subscribe` hands a readonly view; the bucket is a mutable store of its own.
  tabsByProfile[viewKey] = [...tabs]
  persistTabs()
})

// Seed the view with this renderer's own bucket. Without it the primary
// profile's rail never restores: `viewKey` already IS 'default', so
// `setPreviewScope` early-returns and nothing else moves the bucket into the
// atom. Suppressed like the creation emission above — a persist here would
// echo the just-read record back out (wiping a legacy store before adoption).
$previewTabs.set(tabsByProfile[viewKey] ?? [])
adoptingStoredTabs = false

/** Re-home the rail onto the profile that owns the chat on screen. Called by
 *  `session-states.ts` whenever the focused session (or its resolved owner)
 *  changes; the previous agent's tabs must not leak into the next one. */
export function setPreviewScope(scope: string) {
  const next = normalizeProfileKey(scope) || 'default'

  if (next === viewKey) {
    return
  }

  applyPreviewScope(next)
}

/** Swap the view onto `next`'s bucket (legacy tabs ride along into it). Split
 *  from `setPreviewScope` so `adoptPersistedBrowserTab` can force a re-home
 *  onto the bucket a persisted tab lives in — the same-key early return above
 *  would skip exactly that case (a fresh pop-out renderer starts on 'default'
 *  while the popped tab belongs to another profile). */
function applyPreviewScope(next: string) {
  if (pendingLegacyTabs) {
    tabsByProfile[next] = [...(tabsByProfile[next] ?? []), ...pendingLegacyTabs]
    pendingLegacyTabs = null
    persistTabs()
  }

  viewKey = next
  $previewTabs.set(tabsByProfile[next] ?? [])
}

/** Drop one profile's rail. Delete counterpart of the tiles store's
 *  `dropTilesForProfile`, which profile deletion calls. */
export function dropPreviewTabsForProfile(profile: string) {
  const key = normalizeProfileKey(profile)

  delete tabsByProfile[key]
  persistTabs()

  if (key === viewKey) {
    $previewTabs.set([])
  }
}

/** Move one profile's rail to another. Rename counterpart of the tiles store's
 *  `migrateTilesForProfile`: without it a rename strands the tabs under a
 *  profile that no longer exists. */
export function migratePreviewTabsForProfile(oldProfile: string, newProfile: string) {
  const from = normalizeProfileKey(oldProfile)
  const to = normalizeProfileKey(newProfile)

  if (from === to) {
    return
  }

  const moved = tabsByProfile[from]

  if (moved) {
    delete tabsByProfile[from]
    tabsByProfile[to] = [...(tabsByProfile[to] ?? []), ...moved]
  }

  // The view belongs to the renamed profile; only its NAME changed. Re-point it
  // BEFORE the atom is set, so the persist subscriber writes the new bucket
  // rather than resurrecting the one just deleted.
  const wasInView = from === viewKey

  if (wasInView) {
    viewKey = to
  }

  persistTabs()

  if (wasInView) {
    $previewTabs.set(tabsByProfile[to] ?? [])
  }
}

if (typeof window !== 'undefined') {
  try {
    window.localStorage.removeItem(LEGACY_SESSION_REGISTRY_KEY)
  } catch {
    // Storage access can throw in locked-down contexts; nothing depends on it.
  }
}

/** The tab the rail actually shows. A stale or missing selection falls back to
 *  the first tab, so the strip, `⌘W`, and the pane never disagree about which
 *  tab is on screen. */
function resolveActiveTab(tabs: PreviewTab[], activeTabId: RightRailTabId | null): PreviewTab | null {
  return tabs.find(tab => tab.id === activeTabId) ?? tabs[0] ?? null
}

function activePreviewTab(): PreviewTab | null {
  return resolveActiveTab($previewTabs.get(), $rightRailActiveTabId.get())
}

// A restored active id whose tab didn't survive validation would leave the rail
// pointing at nothing.
selectRightRailTab(activePreviewTab()?.id ?? null)

/** The target the rail is currently showing, or null when it has no tabs. */
export const $previewTarget = computed(
  [$previewTabs, $rightRailActiveTabId],
  (tabs, activeTabId) => resolveActiveTab(tabs, activeTabId)?.target ?? null
)

/** Raw `source` strings of every open tab, for the composer rows that toggle a
 *  preview open and closed by the target they were handed. */
export const $previewTabSources = computed($previewTabs, tabs => tabs.map(tab => tab.target.source))

export interface BrowserPage {
  title: string
  url: string
}

/**
 * What each Browser tab is SHOWING right now, as opposed to the target it was
 * opened with. Kept out of the target on purpose: the pane builds its guest
 * from `target.url`, so folding navigation back in would tear the webview down
 * and lose the history behind it. Memory-only — a restored tab reports again
 * on its first load.
 */
export const $browserPages = atom<Record<string, BrowserPage>>({})

export function noteBrowserPage(tabId: string, page: BrowserPage) {
  const current = $browserPages.get()[tabId]

  if (current?.title === page.title && current.url === page.url) {
    return
  }

  $browserPages.set({ ...$browserPages.get(), [tabId]: page })
}

export function forgetBrowserPage(tabId: string) {
  const { [tabId]: gone, ...rest } = $browserPages.get()

  if (gone) {
    $browserPages.set(rest)
  }
}

/** Write the page a Browser is showing back onto its persisted tab. The
 *  webview is built from `target.url`, so this is for hand-off (pop-out /
 *  dock-back), not for every in-page hop — that would tear the guest down. */
export function commitBrowserTabLocation(tabId: string, url: string, title?: string) {
  const nextUrl = url.trim()

  if (!tabId || !nextUrl) {
    return
  }

  const tabs = $previewTabs.get()
  const index = tabs.findIndex(tab => tab.id === tabId)

  if (index === -1) {
    return
  }

  const tab = tabs[index]
  const nextTitle = title?.trim()

  if (tab.target.kind !== 'url' || (tab.target.url === nextUrl && (!nextTitle || tab.target.label === nextTitle))) {
    return
  }

  $previewTabs.set(
    tabs.map((item, i) =>
      i === index
        ? {
            ...item,
            target: {
              ...item.target,
              ...(nextTitle ? { label: nextTitle } : {}),
              url: nextUrl
            }
          }
        : item
    )
  )
}

/** Pull one tab out of shared storage into this renderer's view. Two callers,
 *  two shapes (#119850):
 *
 *  - The docked mirror when a pop-out closes (`onBrowserPopoutClosed`): the
 *    tab is already in this view, so adopt the newer URL/label the sibling
 *    window committed — every bucket is fair game, because the sibling writes
 *    through its own scoped view, which is not necessarily this one.
 *  - A fresh pop-out renderer (`PreviewTilePane` in `?win=browser`): no
 *    session ever pushes a scope there, so the scoped view starts empty. Find
 *    the bucket that owns the tab and re-home the view onto it. Re-homing
 *    rather than splicing the tab into the current bucket keeps this window's
 *    later writes (address-bar navigation) in the OWNER's bucket — a splice
 *    would duplicate the tab into the primary profile's rail.
 *
 *  Reads every profile bucket plus the pre-scoping single-array shape. */
export function adoptPersistedBrowserTab(tabId: string) {
  if (!tabId) {
    return
  }

  try {
    const stored = readJson<unknown>(TABS_STORAGE_KEY)

    if (!stored) {
      return
    }

    const buckets: Array<[string, PreviewTab[]]> = Array.isArray(stored)
      ? [['default', parseTabList(stored)]]
      : Object.entries(stored as Record<string, unknown>).map(
          ([key, value]) => [normalizeProfileKey(key), parseTabList(value)] as [string, PreviewTab[]]
        )

    if ($previewTabs.get().some(tab => tab.id === tabId)) {
      const persisted = buckets.flatMap(([, tabs]) => tabs).find(tab => tab.id === tabId)

      if (persisted?.target.kind === 'url') {
        commitBrowserTabLocation(tabId, persisted.target.url, persisted.target.label)
      }

      return
    }

    for (const [key, tabs] of buckets) {
      if (tabs.some(tab => tab.id === tabId)) {
        applyPreviewScope(key || 'default')

        return
      }
    }
  } catch {
    // Storage can throw; the in-memory tab stays as it was.
  }
}

/** Pop the in-app Browser into its own OS window. Shared by the address-bar
 *  glyph and the tab context menu so they cannot drift. */
export function popOutBrowserTab(tabId: string) {
  if (!tabId || !canOpenBrowserWindow()) {
    return
  }

  const tab = $previewTabs.get().find(item => item.id === tabId)

  if (!tab || tab.target.kind !== 'url') {
    return
  }

  const page = $browserPages.get()[tabId]

  markBrowserTabPopped(tabId, true)
  commitBrowserTabLocation(tabId, page?.url || tab.target.url, page?.title)
  void openBrowserInNewWindow(tabId).then(ok => {
    if (!ok) {
      markBrowserTabPopped(tabId, false)
    }
  })
}

/** Tabs currently shown in a popped-out Browser window. The docked tree
 *  hides them so the page isn't in two places; closing the window docks
 *  them again. Memory-only — a relaunch with no pop-out window restores. */
export const $poppedBrowserTabIds = atom<ReadonlySet<string>>(new Set())

export function markBrowserTabPopped(tabId: string, popped: boolean) {
  const current = $poppedBrowserTabIds.get()

  if (current.has(tabId) === popped) {
    return
  }

  const next = new Set(current)

  if (popped) {
    next.add(tabId)
  } else {
    next.delete(tabId)
  }

  $poppedBrowserTabIds.set(next)
}

/** Preview tabs that still belong in the layout tree (not popped out). */
export const $dockedPreviewTabs = computed([$previewTabs, $poppedBrowserTabIds], (tabs, popped) =>
  popped.size === 0 ? tabs : tabs.filter(tab => !popped.has(tab.id))
)

export const $previewReloadRequest = atom(0)
export const $previewServerRestart = atom<PreviewServerRestart | null>(null)
export const $previewServerRestartStatus = computed($previewServerRestart, restart => restart?.status ?? 'idle')

/** The tab that owns `target`. Files and artifacts are keyed by IDENTITY —
 *  the same file is always the same tab, reopening it re-fronts the one it
 *  already has. A URL has no identity here: a Browser tab is a vessel you
 *  navigate, so it is picked (`browserTabId`) rather than derived. */
export function previewTabId(target: PreviewTarget): RightRailTabId {
  return `${target.kind}:${target.url}`
}

const isBrowserTab = (tab: PreviewTab): boolean => tab.target.kind === 'url'

/** A Browser tab's id, minted the way a terminal's is — there is no identity to
 *  derive one from. Random rather than the lowest free slot: an id is never
 *  handed out twice, so per-tab state keyed by it (`$browserPages`, the console
 *  buffer) cannot resurface under a later tab if a close ever fails to wipe it. */
function mintBrowserTabId(): RightRailTabId {
  const unique =
    globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`

  return `url:browser-${unique}`
}

/** The Browser a URL should open in: the one you're looking at, else the one
 *  you used last. A link from chat navigates the browser you already have
 *  rather than stacking another identical tab — new tabs are something you
 *  ask for (the strip's "+"), the way they are in a real browser. */
function browserTabId(tabs: PreviewTab[]): RightRailTabId {
  const active = tabs.find(tab => tab.id === $rightRailActiveTabId.get())

  if (active && isBrowserTab(active)) {
    return active.id
  }

  return tabs.findLast(isBrowserTab)?.id ?? mintBrowserTabId()
}

/** HTML files open rendered unless the caller asks for a mode. A re-open keeps
 *  the mode the tab is already in, so refreshing the target never undoes a
 *  user's Source pick. */
function withRenderMode(target: PreviewTarget, existing?: PreviewTarget): PreviewTarget {
  if (target.kind !== 'file' || target.previewKind !== 'html' || target.renderMode) {
    return target
  }

  return { ...target, renderMode: existing?.renderMode ?? 'preview' }
}

/** An agent hand-over means "show the page": an HTML file opens rendered even
 *  when its tab is sitting in Source, unlike a re-open from the Files pane. */
export function renderedHtmlTarget(target: PreviewTarget): PreviewTarget {
  return target.kind === 'file' && target.previewKind === 'html' && !target.renderMode
    ? { ...target, renderMode: 'preview' }
    : target
}

/** Flip a tab between live Render and Source in place. Same tab id. */
export function setPreviewRenderMode(tabId: string, renderMode: PreviewRenderMode) {
  const current = $previewTabs.get()
  const index = current.findIndex(tab => tab.id === tabId)

  if (index === -1 || current[index]?.target.renderMode === renderMode) {
    return
  }

  $previewTabs.set(current.map((item, i) => (i === index ? { ...item, target: { ...item.target, renderMode } } : item)))
}

/** Open (or re-front) the tab for `target`. Re-opening an existing tab refreshes
 *  its target so a stale label/path can't outlive the thing it points at. The
 *  only way anything reaches a preview. */
export function openPreview(target: PreviewTarget) {
  const current = $previewTabs.get()
  const id = target.kind === 'url' ? browserTabId(current) : previewTabId(target)
  const index = current.findIndex(tab => tab.id === id)
  const tab: PreviewTab = { id, target: withRenderMode(target, current[index]?.target) }

  $previewTabs.set(index === -1 ? [...current, tab] : current.map((item, i) => (i === index ? tab : item)))
  noteExplicitPreviewOpen(id)
  selectRightRailTab(id)
}

const blankPage = (): PreviewTarget => ({ kind: 'url', label: 'Browser', source: 'about:blank', url: 'about:blank' })

/** Show the Browser — the surface, not a page. Keeps whatever it was last
 *  showing so the hotkey re-fronts your page instead of wiping it; with no
 *  browser open it lands on `about:blank`, where the pane's empty state
 *  invites an address. */
export function openBrowserTab() {
  const tabs = $previewTabs.get()
  const current = tabs.find(tab => tab.id === browserTabId(tabs))

  openPreview(current?.target ?? blankPage())
}

/** Another Browser, always — the strip's "+". */
export function newBrowserTab() {
  const id = mintBrowserTabId()

  $previewTabs.set([...$previewTabs.get(), { id, target: blankPage() }])
  noteExplicitPreviewOpen(id)
  selectRightRailTab(id)
}

export function closeRightRailTab(tabId: string) {
  const current = $previewTabs.get()
  const index = current.findIndex(tab => tab.id === tabId)

  if (index === -1) {
    return
  }

  const next = current.filter(tab => tab.id !== tabId)

  $previewTabs.set(next)

  if ($rightRailActiveTabId.get() === tabId) {
    const nextId = next[Math.min(index, next.length - 1)]?.id ?? null

    if (nextId) {
      noteExplicitPreviewOpen(nextId)
    } else {
      clearExplicitPreviewOpen()
    }

    selectRightRailTab(nextId)
  }

  if (next.length === 0) {
    selectRightRailTab(null)
  }
}

/** Close the tab showing `source`, if one is open. Returns whether it closed. */
export function closePreviewForSource(source: string): boolean {
  return closePreviewMatching(source)
}

/** Close the first tab whose source, url, or label matches any candidate.
 *  Empty candidates are a no-op so a missed match cannot wipe the rail —
 *  closing the whole pane is `closeRightRail`. */
export function closePreviewMatching(...candidates: string[]): boolean {
  const queries = [...new Set(candidates.map(value => value.trim()).filter(Boolean))]

  if (queries.length === 0) {
    return false
  }

  const tab = $previewTabs.get().find(item => {
    const fields = [item.target.source, item.target.url, item.target.label]

    return queries.some(query => fields.includes(query))
  })

  if (!tab) {
    return false
  }

  closeRightRailTab(tab.id)

  return true
}

/** Artifact tabs can't outlive the registry they read from, so clearing it
 *  closes them. File and URL tabs re-read from their source and are left alone. */
export function closeArtifactPreviewTabs() {
  for (const tab of $previewTabs.get()) {
    if (tab.target.kind === 'artifact') {
      closeRightRailTab(tab.id)
    }
  }
}

/** Close every tab so the rail's panes leave the tree. */
export function closeRightRail() {
  clearExplicitPreviewOpen()
  $previewTabs.set([])
  selectRightRailTab(null)
}

export function requestPreviewReload() {
  $previewReloadRequest.set($previewReloadRequest.get() + 1)
}

export function beginPreviewServerRestart(taskId: string, url: string) {
  $previewServerRestart.set({ status: 'running', taskId, url })
}

export function completePreviewServerRestart(taskId: string, text: string) {
  const current = $previewServerRestart.get()

  if (current?.taskId !== taskId) {
    return
  }

  $previewServerRestart.set({
    ...current,
    message: text,
    status: normalize(text).startsWith('error:') ? 'error' : 'complete'
  })
}

export function progressPreviewServerRestart(taskId: string, text: string) {
  const current = $previewServerRestart.get()

  if (current?.taskId !== taskId || current.status !== 'running') {
    return
  }

  $previewServerRestart.set({
    ...current,
    message: text
  })
}

export function failPreviewServerRestart(taskId: string, message: string) {
  const current = $previewServerRestart.get()

  if (current?.taskId !== taskId || current.status !== 'running') {
    return
  }

  $previewServerRestart.set({
    ...current,
    message,
    status: 'error'
  })
}
