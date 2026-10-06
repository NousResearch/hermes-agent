import { atom, computed } from 'nanostores'

import { dismissTreePane, isPaneVisible } from '@/components/pane-shell/tree/store'
import { translateNow } from '@/i18n'
import {
  capturePreviewAnnotateDestination,
  clearPreviewAnnotateDestination,
  rememberPreviewAnnotateDestination
} from '@/lib/preview-annotate/handoff'
import { readJson, writeKey } from '@/lib/storage'
import { normalize } from '@/lib/text'

import type { BrowserWorkspace } from '../../electron/browser-workspace-types'

import { isBrowserSessionRetired } from './browser-conversation'
import { recordFeatureUse } from './desktop-metrics'
import { $rightRailActiveTabId, type RightRailTabId, selectRightRailTab } from './layout'
import { notifyError } from './notifications'
import { clearExplicitPreviewOpen, noteExplicitPreviewOpen, PREVIEW_TILE_PREFIX } from './preview-explicit'
import {
  $pendingRuntimeByTab,
  $rotatedSessionIds,
  $selectionIsListed,
  bucketTabsFor,
  currentSessionId,
  drawerShowsTab,
  ownerIdentity,
  type PreviewOwner,
  setPendingRuntime
} from './preview-ownership'
import { normalizeProfileKey } from './profile'
import { $activeSessionId } from './session'
import { $focusedSessionIsTile, $focusedStoredSessionId } from './session-focus'
import { canOpenBrowserWindow, isBrowserWindow, openBrowserInNewWindow, windowBrowserTabId, windowBrowserWorkspaceId } from './windows'

/**
 * PREVIEW RAIL — one list of tabs, one way in.
 *
 * Everything the rail can show is a `PreviewTarget` in `$previewTabs`: a file
 * on disk, a live URL, or a generated artifact. There is no privileged "live
 * preview" slot alongside the tabs; `openPreview` is the only entry point, so
 * a tool result, a file-browser click, and an artifact card all travel the
 * same road and behave identically once open.
 *
 * Every tab belongs to the session that opened it (#73890): the drawer shows
 * the focused session's tabs plus the pinned ones, and a hidden session's
 * tabs stay alive until it comes back. Tabs close when you close them.
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
  /** `directory`/`missing` are typed non-previewable results from main-process
   * normalization (#101683): they never reach `openPreview` — callers branch on
   * them for the native folder action / not-found reporting instead. */
  previewKind?: 'binary' | 'directory' | 'html' | 'image' | 'missing' | 'pdf' | 'text'
  renderMode?: PreviewRenderMode
  /** Tombstone set when a read/watch confirmed the file is gone. The tab stays
   *  open for the session showing an explicit "file no longer exists" state,
   *  but is dropped at the next restore so day-2 boots stop re-probing it. */
  missing?: boolean
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
  /** Stored id of the session that owns the tab. Absent on a tab opened in a
   *  fresh draft (adopted when the draft becomes a session) and on legacy rows
   *  written before session scoping (decodePreviewTabs migrates those to
   *  pinned). */
  sessionId?: string
  /** Pinned tabs render in EVERY session — the explicit cross-session
   *  workspace. Everything else is visible only in the session that owns it.
   *  Always written by this build, so a missing flag marks a legacy row. */
  pinned?: boolean
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
  const pdfUpgraded = (Array.isArray(parsed) ? parsed.filter(isPreviewTab) : [])
    .map(tab =>
      isPdfFileTarget(tab.target) && tab.target.previewKind === 'binary'
        ? { ...tab, target: { ...tab.target, previewKind: 'pdf' as const } }
        : tab
    )
    // Drop tombstoned file tabs (a previous session confirmed the file is
    // gone). Keeping them would re-probe a known-dead path on every boot.
    .filter(tab => !tab.target.missing)

  // Legacy rows (written before session scoping) have no owner and no way to
  // recover one — keep them as workspace-pinned rather than dropping them or
  // dumping every stale tab into one chat. Ids are kept verbatim: a Browser's
  // minted id is how the pop-out window finds its tab (#119850).
  return pdfUpgraded.map(tab =>
    tab.sessionId !== undefined || tab.pinned !== undefined ? tab : { ...tab, pinned: true }
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
      : JSON.stringify(buckets, (key, value) => (key === 'dataUrl' || key === 'pendingRuntimeId' ? undefined : value))
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
  if (adoptingStoredTabs || windowBrowserWorkspaceId()) {
    return
  }

  // `subscribe` hands a readonly view; the bucket is a mutable store of its own.
  tabsByProfile[viewKey] = [...tabs]
  persistTabs()
  forgetGonePendingTabs()
  publishBrowserOwnership()
})

function publishBrowserOwnership(rotation?: { previousId: string; nextId: string }) {
  if (windowBrowserWorkspaceId()) {return}
  window.hermesDesktop?.browserWorkspace?.updateOwnership?.({
    tabs: Object.values(tabsByProfile).flat().map(tab => ({
      id: tab.id,
      sessionId: tab.sessionId,
      pinned: Boolean(tab.pinned),
      pendingRuntimeId: $pendingRuntimeByTab.get().get(tab.id)
    })),
    rotation
  })
}

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
  if (windowBrowserWorkspaceId()) {
    return
  }

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
  } else {
    forgetGonePendingTabs()
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

/** The tabs an agent tool acting for `owner` (default: the focused session)
 *  may read or drive: never another session's hidden tab, and never another
 *  profile's pin. A popped-out Browser renderer answers only for the one tab
 *  it shows (the chat window decided the requester may see it —
 *  `previewTabIdsVisibleTo`). */
export function previewTabsFor(owner?: PreviewOwner): PreviewTab[] {
  // The standalone shell has no focused chat. Unscoped *UI* navigation uses
  // only its main-admitted membership; agent callers always pass an owner.
  if (windowBrowserWorkspaceId() && owner === undefined) {return $previewTabs.get()}

  if (isBrowserWindow() && !windowBrowserWorkspaceId()) {
    const own = windowBrowserTabId()

    return $previewTabs.get().filter(tab => tab.id === own)
  }

  return bucketTabsFor($previewTabs.get(), ownerIdentity(owner === undefined ? $focusedStoredSessionId.get() : owner), viewKey, viewKey)
}

/** Ids of the tabs `owner` may see in ANY profile's rail, for scoping a
 *  request to a popped-out Browser window: a background session's Browser
 *  can be popped out while another profile is in view. Its own tabs count in
 *  every bucket; pins only in its own profile's. */
export function previewTabIdsVisibleTo(owner: PreviewOwner): string[] {
  const who = ownerIdentity(owner)

  const buckets: [string, readonly PreviewTab[]][] = [
    [viewKey, $previewTabs.get()],
    ...Object.entries(tabsByProfile).filter(([key]) => key !== viewKey)
  ]

  return buckets.flatMap(([key, tabs]) => bucketTabsFor(tabs, who, key, viewKey)).map(tab => tab.id)
}

/** Tabs the FOCUSED session sees. The layout-tree mirror renders only these,
 *  so a session switch swaps the drawer; hidden tabs stay in `$previewTabs`.
 *  The primary also shows the tabs its runtime opened before its stored id
 *  arrived: the selection can name that id a beat before the runtime binds it,
 *  and a Browser hidden in that gap would lose its page. Never under a listed
 *  session: a resume selects it a beat before it unbinds the runtime. */
export const $visiblePreviewTabs = computed(
  [
    $previewTabs,
    $focusedStoredSessionId,
    $rotatedSessionIds,
    $pendingRuntimeByTab,
    $activeSessionId,
    $focusedSessionIsTile,
    $selectionIsListed
  ],
  (tabs, sessionId, rotated, pending, activeRuntime, focusedIsTile, selectionIsListed) =>
    tabs.filter(tab =>
      drawerShowsTab(tab, { activeRuntime, focusedIsTile, pending, rotated, selectionIsListed, sessionId })
    )
)

// The tab each session last had in front, so switching back to a session
// fronts what it was showing instead of its first tab. Memory-only.
const activeTabBySession = new Map<string, RightRailTabId>()

function rememberActiveTab(sessionId: null | string, tabId: RightRailTabId | null): void {
  const key = currentSessionId(sessionId)

  if (key && tabId) {
    activeTabBySession.set(key, tabId)
  }
}

/** Rekey a rotated conversation's tabs onto its new stored id. Every profile
 *  bucket, not just the view: a background tile's conversation rotates too. */
export function rekeyPreviewTabsSession(previousId: string, nextId: string): void {
  // Both ends resolve to their current tips first: a late or replayed event
  // can name an id that already rotated, or point back at an older tip, and
  // linking anything but two distinct tips would close an alias cycle.
  const from = currentSessionId(previousId)
  const to = currentSessionId(nextId)

  if (!from || !to || from === to) {
    return
  }

  const owned = (tab: PreviewTab) => currentSessionId(tab.sessionId) === from

  const rekey = (tabs: PreviewTab[]) =>
    tabs.some(owned) ? tabs.map(tab => (owned(tab) ? { ...tab, sessionId: to } : tab)) : null

  let backgroundChanged = false

  for (const [key, tabs] of Object.entries(tabsByProfile)) {
    const next = key === viewKey ? null : rekey(tabs)

    if (next) {
      tabsByProfile[key] = next
      backgroundChanged = true
    }
  }

  if (backgroundChanged) {
    persistTabs()
  }

  const view = rekey($previewTabs.get())

  // Only after the rekey: `owned` resolves through the map as it was.
  $rotatedSessionIds.set(new Map([...$rotatedSessionIds.get(), [from, to]]))

  if (view) {
    $previewTabs.set(view)
  }

  publishBrowserOwnership({ previousId: from, nextId: to })

  const remembered = activeTabBySession.get(from)

  if (remembered) {
    activeTabBySession.set(to, remembered)
  }
}

$rightRailActiveTabId.listen(tabId => {
  const sessionId = $focusedStoredSessionId.get()

  if (tabId && $visiblePreviewTabs.get().some(tab => tab.id === tabId)) {
    rememberActiveTab(sessionId, tabId)
  }
})

/** The tab the focused session's drawer should front: the current selection
 *  when it is visible, else the one this session last had in front, else its
 *  first tab. */
export function preferredVisibleTabId(): RightRailTabId | null {
  const visible = $visiblePreviewTabs.get()
  const isVisible = (id: null | RightRailTabId | undefined) => Boolean(id && visible.some(tab => tab.id === id))
  const active = $rightRailActiveTabId.get()

  if (isVisible(active)) {
    return active
  }

  const key = currentSessionId($focusedStoredSessionId.get())
  const remembered = key ? activeTabBySession.get(key) : undefined

  return isVisible(remembered) ? remembered! : (visible[0]?.id ?? null)
}

/** A fresh draft has no session yet, so tabs opened there are ownerless (the
 *  drawer of every draft shows them). Called where the draft's first send
 *  assigns its stored id — beside the composer draft's own hand-over — so the
 *  tabs follow it into the conversation. Never on a focus change: clicking an
 *  existing session or a side tile must leave the draft's tabs in the draft. */
export function adoptDraftPreviewTabs(storedSessionId: string): void {
  const pending = $pendingRuntimeByTab.get()
  // A tab a live runtime opened before its stored id is that runtime's, not
  // the draft's.
  const draftOwned = (tab: PreviewTab) => tab.sessionId == null && !tab.pinned && !pending.has(tab.id)
  const tabs = $previewTabs.get()

  if (tabs.some(draftOwned)) {
    $previewTabs.set(tabs.map(tab => (draftOwned(tab) ? { ...tab, sessionId: storedSessionId } : tab)))
    const runtimeId = $activeSessionId.get()

    if (runtimeId) {publishBrowserOwnership({ previousId: runtimeId, nextId: storedSessionId })}
  }
}

/** Drop the runtime notes of tabs no profile bucket holds any more (closed,
 *  pruned, or their profile dropped). */
function forgetGonePendingTabs(): void {
  const pending = $pendingRuntimeByTab.get()

  if (pending.size === 0) {
    return
  }

  const alive = new Set<string>(Object.values(tabsByProfile).flatMap(tabs => tabs.map(tab => tab.id)))

  if ([...pending.keys()].some(id => !alive.has(id))) {
    $pendingRuntimeByTab.set(new Map([...pending].filter(([id]) => alive.has(id))))
  }
}

/** `runtimeId`'s stored id was just bound (its session state went from no
 *  stored id to `storedSessionId`): the tabs it opened before then are that
 *  session's. Called by the session-state layer at that transition — never on
 *  a selection change, which is also what resuming another session looks
 *  like. Every profile bucket: the runtime may not be the one in view. */
export function adoptPendingRuntimeTabs(runtimeId: string, storedSessionId: string): void {
  const ids = new Set([...$pendingRuntimeByTab.get()].filter(([, runtime]) => runtime === runtimeId).map(([id]) => id))

  if (ids.size === 0) {
    return
  }

  const adopt = (tabs: PreviewTab[]) =>
    tabs.some(tab => ids.has(tab.id) && tab.sessionId == null)
      ? tabs.map(tab => (ids.has(tab.id) && tab.sessionId == null ? { ...tab, sessionId: storedSessionId } : tab))
      : null

  let backgroundChanged = false

  for (const [key, tabs] of Object.entries(tabsByProfile)) {
    const next = key === viewKey ? null : adopt(tabs)

    if (next) {
      tabsByProfile[key] = next
      backgroundChanged = true
    }
  }

  if (backgroundChanged) {
    persistTabs()
  }

  const view = adopt($previewTabs.get())

  // Owner first, then the note: the other order leaves a beat where the tab
  // is neither owned nor pending and drops out of the drawer.
  if (view) {
    $previewTabs.set(view)
  }

  $pendingRuntimeByTab.set(new Map([...$pendingRuntimeByTab.get()].filter(([id]) => !ids.has(id))))
  publishBrowserOwnership({ previousId: runtimeId, nextId: storedSessionId })
}

/** The tab the rail actually shows. A stale or missing selection falls back to
 *  the first tab, so the strip, `⌘W`, and the pane never disagree about which
 *  tab is on screen. */
function resolveActiveTab(tabs: PreviewTab[], activeTabId: RightRailTabId | null): PreviewTab | null {
  return tabs.find(tab => tab.id === activeTabId) ?? tabs[0] ?? null
}

// A restored active id whose tab didn't survive validation would leave the rail
// pointing at nothing. Checked against every tab, not the visible ones: at
// boot no session is focused yet, and re-homing onto the focused session's
// tabs is the preview tiles' job once one is.
selectRightRailTab(resolveActiveTab($previewTabs.get(), $rightRailActiveTabId.get())?.id ?? null)

/** The target the rail is currently showing, or null when it has no tabs. */
export const $previewTarget = computed(
  [$visiblePreviewTabs, $rightRailActiveTabId],
  (tabs, activeTabId) => resolveActiveTab(tabs, activeTabId)?.target ?? null
)

/** Raw `source` strings of every tab the active session sees, for the composer
 *  rows that toggle a preview open and closed by the target they were handed. */
export const $previewTabSources = computed($visiblePreviewTabs, tabs => tabs.map(tab => tab.target.source))

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
  const windowId = windowBrowserWorkspaceId()

  if (windowId) {
    void window.hermesDesktop?.browserWorkspace?.command(windowId, {
      kind: 'page',
      tabId,
      url: page.url,
      title: page.title
    })
  }
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

  // Pin the exact chat/group surface that owns this Browser before the new
  // renderer opens. Comment Mode in the pop-out uses this route to hand its
  // saved batch back without guessing from whichever composer is active later.
  const anchor =
    typeof document !== 'undefined' && document.activeElement instanceof Element ? document.activeElement : null

  const destination = capturePreviewAnnotateDestination(anchor)

  if (window.hermesDesktop?.browserWorkspace && !destination?.conversation) {
    notifyError(new Error(translateNow('preview.popOutOwnerUnavailable')), translateNow('preview.popOutFailed'))

    return
  }

  rememberPreviewAnnotateDestination(tabId, destination)
  noteExplicitPreviewOpen(tab.id)
  markBrowserTabPopped(tabId, true)
  selectRightRailTab(tab.id)
  commitBrowserTabLocation(tabId, page?.url || tab.target.url, page?.title)

  const workspace = window.hermesDesktop?.browserWorkspace
    ? {
        tab: {
          id: tab.id as `url:${string}`,
          sessionId: tab.sessionId,
          pinned: Boolean(tab.pinned),
          pendingRuntimeId: $pendingRuntimeByTab.get().get(tab.id),
          target: {
            kind: 'url' as const,
            label: page?.title || tab.target.label,
            source: tab.target.source,
            url: page?.url || tab.target.url
          }
        },
        scope: viewKey,
        destination
      }
    : undefined

  void openBrowserInNewWindow(tabId, workspace).then(ok => {
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
    clearPreviewAnnotateDestination(tabId)
  }

  $poppedBrowserTabIds.set(next)
}

/** The FOCUSED session's tabs that still belong in the docked layout tree —
 *  the layout-tree mirror renders only these (#73890): switching sessions
 *  swaps the drawer, and a popped-out Browser pane stays out. */
export const $dockedVisiblePreviewTabs = computed([$visiblePreviewTabs, $poppedBrowserTabIds], (tabs, popped) =>
  popped.size === 0 ? tabs : tabs.filter(tab => !popped.has(tab.id))
)

export const $previewReloadRequest = atom(0)
export const $previewServerRestart = atom<PreviewServerRestart | null>(null)
export const $previewServerRestartStatus = computed($previewServerRestart, restart => restart?.status ?? 'idle')

/** The tab that owns `target`. Files and artifacts are keyed by IDENTITY —
 *  the same file is always the same tab, reopening it re-fronts the one it
 *  already has. A URL has no identity here: a Browser tab is a vessel you
 *  navigate, so it is picked (`browserTabId`) rather than derived.
 *
 *  A FILE tab is additionally owned by a session — the same file opened in
 *  two conversations is two tabs (#73890) — so a session-owned file tab's id
 *  carries the owner it was opened by. The id is identity only: lookups match
 *  a file by its canonical path and owner (`fileTabFor`), so an id never needs
 *  rekeying when the owner changes (draft adoption, compression rotation). */
export function previewTabId(target: PreviewTarget, sessionId?: null | string): RightRailTabId {
  if (target.kind === 'file' && sessionId) {
    return `file:${sessionId}:${target.url}`
  }

  return `${target.kind}:${target.url}`
}

/** A file's identity independent of entry point: a `file://` URL and a plain
 *  path name the same file. */
function canonicalFileKey(target: Pick<PreviewTarget, 'path' | 'url'>): string {
  return (target.path || target.url).replace(/^file:\/\//, '')
}

/** The tab already showing `target`'s file in `tabs` (a session's visible set). */
function fileTabFor(tabs: readonly PreviewTab[], target: PreviewTarget): PreviewTab | undefined {
  const key = canonicalFileKey(target)

  return tabs.find(tab => tab.target.kind === 'file' && canonicalFileKey(tab.target) === key)
}

/** `base`, suffixed until no tab holds it: a file's owner-derived id can be
 *  taken by a same-path tab whose owner later changed. */
function unusedTabId(base: RightRailTabId, tabs: readonly PreviewTab[]): RightRailTabId {
  let id = base

  for (let n = 2; tabs.some(tab => tab.id === id); n++) {
    id = `${base}#${n}` as RightRailTabId
  }

  return id
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
 *  ask for (the strip's "+"), the way they are in a real browser. `tabs` is
 *  the opening session's visible set: another session's Browser is never
 *  taken over. */
function browserTabFor(tabs: readonly PreviewTab[]): PreviewTab | undefined {
  const active = tabs.find(tab => tab.id === $rightRailActiveTabId.get())

  return active && isBrowserTab(active) ? active : tabs.findLast(isBrowserTab)
}

function browserTabId(tabs: readonly PreviewTab[]): RightRailTabId {
  return browserTabFor(tabs)?.id ?? mintBrowserTabId()
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
 *  only way anything reaches a preview.
 *
 *  The tab belongs to `owner` — the stored id of the session that asked for
 *  it, the focused one unless the caller knows better (an agent turn in a
 *  side tile). Reuse is decided among the tabs `owner` can see (its own plus
 *  the pinned ones), so one session's open never takes over another
 *  session's tab; a reused tab keeps its owner and pin. Opening for a session
 *  that is not focused does not touch the focused drawer's selection — the
 *  tab is fronted when that session's drawer shows. */
export function openPreview(
  target: PreviewTarget,
  requestedOwner: null | string = $focusedStoredSessionId.get(),
  /** The runtime asking: an ownerless tab is handed to it once its stored id
   *  binds. An agent event passes its own session id; anything else is the
   *  primary's runtime. */
  runtimeId: null | string = $activeSessionId.get(),
  /** The asking agent's profile, when it is not the chat on screen: only that
   *  profile's pins may be reused. Omitted = the viewed profile. */
  profile?: null | string
) {
  // Stamp the tip, not an alias: the alias map is memory-only, so a tab
  // stamped with a rotated-away id would be orphaned by the next relaunch.
  const owner = currentSessionId(requestedOwner)
  const current = $previewTabs.get()

  // Reuse never crosses runtimes: with no stored id yet, the runtime decides
  // which ownerless tabs are this opener's (its own pending ones, or the
  // draft's when it is the draft on screen).
  const visible = bucketTabsFor(
    current,
    ownerIdentity({ profile, runtimeId: owner === null ? runtimeId : null, sessionId: owner }),
    viewKey,
    viewKey
  )

  const existing =
    target.kind === 'url'
      ? browserTabFor(visible)
      : target.kind === 'file'
        ? fileTabFor(visible, target)
        : current.find(tab => tab.id === previewTabId(target))

  const id =
    existing?.id ?? (target.kind === 'url' ? mintBrowserTabId() : unusedTabId(previewTabId(target, owner), current))

  const tab: PreviewTab = {
    id,
    pinned: Boolean(existing?.pinned),
    // An artifact is one tab per artifact; opening it from another session
    // re-owns it, or the session that just asked for it would not see it.
    sessionId: (existing?.pinned ? existing.sessionId : owner) ?? undefined,
    target: withRenderMode(target, existing?.target)
  }

  $previewTabs.set(existing ? current.map(item => (item === existing ? tab : item)) : [...current, tab])

  setPendingRuntime(id, tab.sessionId == null && !tab.pinned ? runtimeId : null)
  publishBrowserOwnership()

  if (!$visiblePreviewTabs.get().some(item => item.id === id)) {
    rememberActiveTab(owner, id)

    return
  }

  noteExplicitPreviewOpen(id)
  selectRightRailTab(id)
}

const blankPage = (): PreviewTarget => ({ kind: 'url', label: 'Browser', source: 'about:blank', url: 'about:blank' })

/** Tombstone the tabs for a confirmed-missing file: keep them open this
 *  session (the pane shows "file no longer exists"), but flag the target so
 *  the next restore drops them instead of re-probing the dead path on every
 *  boot. Takes a tab id or the file's url/path; a file that is gone is gone
 *  for every session showing it. */
export function markPreviewTabMissing(tabIdOrUrl: string) {
  const current = $previewTabs.get()
  const key = canonicalFileKey({ url: tabIdOrUrl.replace(/^file:(?!\/\/)/, '') })

  const hit = (tab: PreviewTab) =>
    tab.target.kind === 'file' &&
    !tab.target.missing &&
    (tab.id === tabIdOrUrl || tab.target.url === tabIdOrUrl || canonicalFileKey(tab.target) === key)

  if (!current.some(hit)) {
    return
  }

  $previewTabs.set(current.map(tab => (hit(tab) ? { ...tab, target: { ...tab.target, missing: true } } : tab)))
}

/** Show the Browser — the surface, not a page. Keeps whatever it was last
 *  showing so the hotkey re-fronts your page instead of wiping it; with no
 *  browser open it lands on `about:blank`, where the pane's empty state
 *  invites an address. */
export function openBrowserTab() {
  const tabs = $visiblePreviewTabs.get()
  const current = tabs.find(tab => tab.id === browserTabId(tabs))

  recordFeatureUse('browser_pane')
  openPreview(current?.target ?? blankPage())
}

/** ⌘⇧L is a TOGGLE: show the Browser when it's away, fold it away when it's
 *  the thing on screen. "Away" includes dismissed (Close/⌘W), hidden, or
 *  parked behind a sibling tab — each re-opens through openBrowserTab's reveal
 *  path with the page it was last showing. "On screen" means the mirrored
 *  preview-tile pane the layout tree keeps is actually visible, i.e. not
 *  dismissed/hidden/minimized AND holding its zone's active slot. */
export function toggleBrowserTab() {
  const id = browserTabId($visiblePreviewTabs.get())

  if (isPaneVisible(`${PREVIEW_TILE_PREFIX}:${id}`)) {
    dismissTreePane(`${PREVIEW_TILE_PREFIX}:${id}`)

    return
  }

  openBrowserTab()
}

/** Another Browser, always — the strip's "+". */
/** Apply only this workspace's membership, never another renderer's whole
 * profile bucket. Detached guests keep their initial target URL: page reports
 * update persistence in the owner, not the guest's mount identity. */
export function applyBrowserWorkspace(state: BrowserWorkspace, previous?: BrowserWorkspace) {
  if (windowBrowserWorkspaceId()) {
    if (state.id !== windowBrowserWorkspaceId()) {
      return
    }

    const existing = $previewTabs.get()
    viewKey = state.owner.scope
    $previewTabs.set(state.tabs.map(tab => {
      const mounted = previous && existing.find(item => item.id === tab.id)
      setPendingRuntime(tab.id, tab.pendingRuntimeId ?? null)

      return mounted && mounted.sessionId === tab.sessionId && Boolean(mounted.pinned) === Boolean(tab.pinned)
        ? mounted
        : { ...tab, pinned: Boolean(tab.pinned), target: mounted ? mounted.target : tab.target }
    }))
    selectRightRailTab(state.activeTabId as RightRailTabId | null)

    for (const tab of state.tabs) {
      rememberPreviewAnnotateDestination(tab.id, state.owner.conversation?.kind === 'session' && isBrowserSessionRetired(state.owner.conversation.id) ? null : state.owner.destination)
    }

    return
  }

  if (state.profileRetirement) {
    // Reconcile against disk, not a stale opener cache. This terminal snapshot
    // is also safe on cold reload before the opener acknowledged the transfer.
    const stored = loadTabsByProfile()

    for (const bucket of Object.keys(tabsByProfile)) {
      const transient = tabsByProfile[bucket]!.filter(tab => !persistableTabs([tab]).length)
      tabsByProfile[bucket] = [...(stored[bucket] ?? []), ...transient]

      if (!tabsByProfile[bucket]!.length) {delete tabsByProfile[bucket]}
    }

    for (const bucket of Object.keys(stored)) {
      if (!(bucket in tabsByProfile)) {tabsByProfile[bucket] = stored[bucket]!}
    }

    const from = state.owner.scope
    const to = state.profileRetirement.replacementScope
    const moved = (tabsByProfile[from] ?? []).filter(tab => !state.removed.includes(tab.id))
    delete tabsByProfile[from]

    if (to) {
      const kept = (tabsByProfile[to] ?? []).filter(tab => !state.removed.includes(tab.id))
      const merged = new Map([...kept, ...moved, ...state.docked].map(tab => [tab.id, tab]))
      tabsByProfile[to] = [...merged.values()]

      if (viewKey === from) {viewKey = to}
    }

    for (const id of state.removed) {
      markBrowserTabPopped(id, false)
      forgetBrowserPage(id)
      clearPreviewAnnotateDestination(id)
    }

    $previewTabs.set(tabsByProfile[viewKey] ?? [])

    return
  }

  const retired = new Set([...state.removed, ...state.docked.map(tab => tab.id)])
  const priorDocked = new Set(previous?.docked.map(tab => tab.id) ?? [])

  const incoming = [...state.tabs, ...state.docked.filter(tab => !priorDocked.has(tab.id))]
    .filter(tab => tab.pinned || !tab.sessionId || !isBrowserSessionRetired(tab.sessionId))
    .map(tab => ({ ...tab, pinned: Boolean(tab.pinned), sessionId: tab.sessionId && !isBrowserSessionRetired(tab.sessionId) ? currentSessionId(tab.sessionId) ?? undefined : undefined }))

  const key = state.owner.scope
  // The detached renderer never writes storage. Its opener applies only this
  // workspace's IDs over the freshest persisted buckets, so another opener's
  // page updates cannot be overwritten by this renderer's startup snapshot.
  const stored = loadTabsByProfile()
  const local = key === viewKey ? $previewTabs.get() : (tabsByProfile[key] ?? [])

  // Absence is a deletion, not permission to revive a cached persistent tab.
  // Keep runtime-only tabs; replace every persistent bucket from storage.
  for (const bucket of Object.keys(tabsByProfile)) {
    const transient = tabsByProfile[bucket]!.filter(tab => !persistableTabs([tab]).length)
    tabsByProfile[bucket] = [...(stored[bucket] ?? []), ...transient]

    if (!tabsByProfile[bucket]!.length) {delete tabsByProfile[bucket]}
  }

  for (const bucket of Object.keys(stored)) {
    if (!(bucket in tabsByProfile)) {tabsByProfile[bucket] = stored[bucket]!}
  }

  const current = [...(stored[key] ?? [])]

  for (const tab of local) {
    if (!persistableTabs([tab]).length && !current.some(item => item.id === tab.id)) {
      current.push(tab)
    }
  }

  // Page snapshots cannot roll back an opener's newer pin or session binding.
  const replacements = new Map(incoming.map(tab => {
    const owned = current.find(item => item.id === tab.id)
    const next = owned ? { ...tab, pinned: Boolean(owned.pinned), sessionId: owned.sessionId } : tab

    return [tab.id, next] as const
  }))

  const next = current
    .filter(tab => !state.removed.includes(tab.id) && (tab.pinned || !tab.sessionId || !isBrowserSessionRetired(tab.sessionId)))
    .map(tab => replacements.get(tab.id as `url:${string}`) ?? tab)

  for (const tab of incoming) {
    if (!next.some(item => item.id === tab.id)) {
      // New detached tabs inherit pending ownership from main. Record it before
      // publishing the bucket; existing tabs keep the opener's newer metadata.
      setPendingRuntime(tab.id, tab.pendingRuntimeId ?? null)
      next.push(tab)
    }
  }

  for (const id of retired) {
    markBrowserTabPopped(id, false)
  }

  for (const tab of state.tabs) {
    markBrowserTabPopped(tab.id, incoming.some(item => item.id === tab.id))
  }

  for (const tab of incoming) {
    rememberPreviewAnnotateDestination(tab.id, state.owner.conversation?.kind === 'session' && isBrowserSessionRetired(state.owner.conversation.id) ? null : state.owner.destination)
  }

  for (const id of state.removed) {
    forgetBrowserPage(id)
    clearPreviewAnnotateDestination(id)
  }

  tabsByProfile[key] = next

  if (key === viewKey) {
    $previewTabs.set(next)
  } else {
    // Reconcile the visible cache too: a subsequent local edit must not write
    // a bucket another window just deleted back out of this stale atom.
    $previewTabs.set(tabsByProfile[viewKey] ?? [])
  }
}

export function previewScope(): string {
  return viewKey
}

export function newBrowserTab() {
  const id = mintBrowserTabId()

  recordFeatureUse('browser_pane')
  $previewTabs.set([
    ...$previewTabs.get(),
    { id, pinned: false, sessionId: currentSessionId($focusedStoredSessionId.get()) ?? undefined, target: blankPage() }
  ])
  noteExplicitPreviewOpen(id)
  selectRightRailTab(id)
}

/** Pin or unpin a preview tab. Pinned tabs render in EVERY session — the
 *  explicit cross-session workspace; unpinning returns it to its session
 *  (adopting the current one when the tab never had an owner). */
export function setPreviewTabPinned(tabId: string, pinned: boolean): void {
  const currentSession = currentSessionId($focusedStoredSessionId.get()) ?? undefined

  $previewTabs.set(
    $previewTabs
      .get()
      .map(tab =>
        tab.id === tabId ? { ...tab, pinned, sessionId: tab.sessionId ?? (pinned ? undefined : currentSession) } : tab
      )
  )
}

/** Drop the tabs a deleted session opened. Pinned tabs survive — they belong
 *  to the workspace, not the session that opened them. Every profile bucket,
 *  not just the view: a session can be deleted while another profile's chat
 *  is on screen, and its tabs live in its own profile's bucket. */
export function prunePreviewTabsForSession(sessionId: string): void {
  const doomed = currentSessionId(sessionId)
  const keep = (tab: PreviewTab) => tab.pinned || currentSessionId(tab.sessionId) !== doomed

  const detachOwner = (tab: PreviewTab): PreviewTab => {
    if (currentSessionId(tab.sessionId) !== doomed) {return tab}
    clearPreviewAnnotateDestination(tab.id)

    return { ...tab, sessionId: undefined }
  }

  let backgroundChanged = false

  for (const [key, tabs] of Object.entries(tabsByProfile)) {
    if (key !== viewKey && tabs.some(tab => currentSessionId(tab.sessionId) === doomed)) {
      tabsByProfile[key] = tabs.filter(keep).map(detachOwner)
      backgroundChanged = true
    }
  }

  if (backgroundChanged) {
    persistTabs()
    forgetGonePendingTabs()
  }

  $previewTabs.set($previewTabs.get().filter(keep).map(detachOwner))
}

export function closeRightRailTab(tabId: string) {
  closeRightRailTabs(new Set([tabId]))
}

/** Close `tabIds` in one write, then re-home the selection once. */
function closeRightRailTabs(tabIds: ReadonlySet<string>) {
  const current = $previewTabs.get()

  if (!current.some(tab => tabIds.has(tab.id))) {
    return
  }

  const next = current.filter(tab => !tabIds.has(tab.id))
  // The neighbour comes from the focused drawer: a hidden session's tab
  // must not become the selection.
  const activeId = $rightRailActiveTabId.get()
  const visible = $visiblePreviewTabs.get()
  const visibleIndex = visible.findIndex(tab => tab.id === activeId)
  const remaining = visible.filter(tab => !tabIds.has(tab.id))

  for (const tabId of tabIds) {
    forgetBrowserPage(tabId)
  }

  $previewTabs.set(next)

  if (activeId && tabIds.has(activeId)) {
    const nextId = remaining[Math.min(Math.max(visibleIndex, 0), remaining.length - 1)]?.id ?? null

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

/** Close the tab showing `source` in the CURRENT session, if one is open.
 *  Returns whether it closed. */
export function closePreviewForSource(source: string): boolean {
  return closePreviewMatching(source)
}

/** Close the first docked Browser tab whose current page URL matches.
 *  Browsers keep navigation state outside their persisted target so matching
 *  only target.url misses redirects and in-page navigation. */
export function closeBrowserPreviewMatchingLiveUrl(...candidates: string[]): boolean {
  return closeBrowserMatchingLiveUrlIn($visiblePreviewTabs.get(), candidates)
}

function closeBrowserMatchingLiveUrlIn(tabs: readonly PreviewTab[], candidates: string[]): boolean {
  const queries = new Set(
    candidates
      .map(value => {
        try {
          const url = new URL(value.trim())

          return url.protocol === 'http:' || url.protocol === 'https:' ? url.href : ''
        } catch {
          return ''
        }
      })
      .filter(Boolean)
  )

  if (queries.size === 0) {
    return false
  }

  const pages = $browserPages.get()
  const popped = $poppedBrowserTabIds.get()
  const activeId = $rightRailActiveTabId.get()
  const ordered = [...tabs.filter(tab => tab.id === activeId), ...tabs.filter(tab => tab.id !== activeId)]

  const tab = ordered.find(item => {
    if (item.target.kind !== 'url' || popped.has(item.id)) {
      return false
    }

    const liveUrl = pages[item.id]?.url

    if (!liveUrl) {
      return false
    }

    try {
      return queries.has(new URL(liveUrl).href)
    } catch {
      return false
    }
  })

  if (!tab) {
    return false
  }

  closeRightRailTab(tab.id)

  return true
}

function closePreviewMatchingTabs(tabs: readonly PreviewTab[], candidates: string[]): boolean {
  const queries = [...new Set(candidates.map(value => value.trim()).filter(Boolean))]

  if (queries.length === 0) {
    return false
  }

  const tab = tabs.find(item => {
    const fields = [item.target.source, item.target.url, item.target.label]

    return queries.some(query => fields.includes(query))
  })

  if (!tab) {
    return false
  }

  closeRightRailTab(tab.id)

  return true
}

/** Close the first tab whose source, url, or label matches any candidate.
 *  Empty candidates are a no-op so a missed match cannot wipe the rail —
 *  closing the whole pane is `closeRightRail`. */
export function closePreviewMatching(...candidates: string[]): boolean {
  return closePreviewMatchingTabs($visiblePreviewTabs.get(), candidates)
}

/** Agent-driven close is scoped to the docked rail; an independent Browser
 *  window owns popped tabs and must not lose its backing state here. */
export function closeDockedPreviewMatching(...candidates: string[]): boolean {
  return closePreviewMatchingTabs(dockedTabs($visiblePreviewTabs.get()), candidates)
}

function dockedTabs(tabs: readonly PreviewTab[]): PreviewTab[] {
  const popped = $poppedBrowserTabIds.get()

  return tabs.filter(tab => !popped.has(tab.id))
}

/** An agent's `close_preview` for the session that ran it (`owner`). With
 *  candidates it closes the first matching tab that session can see (live
 *  Browser page first, then source/url/label); without, it closes every tab
 *  that session owns. Never another session's tabs (another runtime's pending
 *  ones included), never another profile's pin, never a pin the agent did not
 *  name. */
export function closeAgentPreview(owner: PreviewOwner, candidates: string[]): void {
  const visible = bucketTabsFor($previewTabs.get(), ownerIdentity(owner), viewKey, viewKey)

  if (candidates.length > 0) {
    if (!closeBrowserMatchingLiveUrlIn(visible, candidates)) {
      closePreviewMatchingTabs(dockedTabs(visible), candidates)
    }

    return
  }

  closeRightRailTabs(new Set(visible.filter(tab => !tab.pinned).map(tab => tab.id)))
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
