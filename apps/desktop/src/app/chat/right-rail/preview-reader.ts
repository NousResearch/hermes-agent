/**
 * PREVIEW READER — the read_preview tool's window into the preview pane, the
 * preview analog of the terminal's buffer registry (see right-sidebar/
 * terminal/buffer.ts).
 *
 * A URL/HTML preview renders in a sandboxed <webview> owned by PreviewPane;
 * that pane registers a PAGE READER here (url + title + rendered text), keyed
 * by tab id. `readActivePreview` resolves the preview the user is looking at
 * (hovered zone, else focused zone, else the store) and owns the windowing:
 * a registered reader answers with the live page's text;
 * a tab with no reader (a file peek, an artifact) still answers with its
 * identity and a note pointing the agent at the tool that reads that content
 * directly (read_file / the conversation's artifact).
 */

import { findGroup } from '@/components/pane-shell/tree/model'
import { $activeTreeGroup, $hoveredTreeGroup, $layoutTree } from '@/components/pane-shell/tree/store'
import { $rightRailActiveTabId } from '@/store/layout'
import { $previewTabs, type PreviewTab } from '@/store/preview'
import { explicitOpenBlocksZone, PREVIEW_TILE_PREFIX } from '@/store/preview-explicit'
// Runtime→stored identity translation for the durable-ownership leg below. No
// cycle: the session-store tree never imports this right-rail leaf.
import { storedSessionIdForRuntimeId } from '@/store/session-states'

import { nudgeOverlay } from './preview-nudge'

export interface PreviewReadOptions {
  /** Characters to return from `start` (capped at PREVIEW_READ_MAX_CHARS). */
  count?: number
  /** 0-indexed character offset into the page text. */
  start?: number
}

export interface PreviewReadTabSummary {
  id: string
  kind: string
  label: string
  url: string
}

export interface PreviewReadResult {
  /** Set when more than one preview is mounted — the tab this read used. */
  active_tab_id?: string
  end: number
  kind: string
  note?: string
  path?: string
  start: number
  /** Open preview tabs, set when more than one is mounted. */
  tabs?: PreviewReadTabSummary[]
  text: string
  title: string
  total_chars: number
  url: string
}

/** What a pane's page reader extracts — the reader module owns the windowing. */
interface PreviewPage {
  text: string
  title: string
  url: string
}

type PageReader = () => Promise<PreviewPage>

/** Default + hard cap on one read — a page's innerText can be megabytes, and
 *  this crosses the gateway into model context. Page with start/count. */
export const PREVIEW_READ_MAX_CHARS = 24_000

const readers = new Map<string, PageReader>()

/** Owning session (runtime id) for each registered reader — the session whose
 *  agent opened this preview, bound at registration time from the tab's own
 *  owner stamps, never from whichever chat happens to be selected. */
const readerSessions = new Map<string, string>()

/** Durable (stored-id) owner for each registered reader. Survives the runtime-
 *  id rotation a Desktop restart performs; see isLivePreviewTabOwnedBySession. */
const readerStoredSessions = new Map<string, string>()

/** True when THIS preview tab is a live reader owned by `sessionId` and still
 *  open in `$previewTabs`. Asks about ONE specific tab — the one a mutation
 *  targets — so authorization depends on the identity of the preview being
 *  acted on, never on registration order (#95459).
 *
 *  Ownership matches EITHER identity kind. A live runtime id matches the
 *  runtime stamp exactly; across a restart (runtime ids rotate, stored ids
 *  don't) the tab's DURABLE owner still admits the same conversation's new
 *  runtime id, translated here through the session store's runtime→stored map
 *  (#95459's restart sequence: restart → interact with the ALREADY-open
 *  preview, no fresh openPreview re-stamp). Fail-closed when either side is
 *  unknown: proving something is on screen proves nothing about who owns it. */
export function isLivePreviewTabOwnedBySession(tabId: string, sessionId: string): boolean {
  if (!sessionId || !tabId) {
    return false
  }

  const tab = $previewTabs.get().find(candidate => candidate.id === tabId)

  if (!tab || !readers.has(tabId)) {
    return false
  }

  if (readerSessions.get(tabId) === sessionId) {
    return true
  }

  // Durable leg: the tab's stored owner names a CONVERSATION. A runtime id of
  // that same conversation owns the tab even after restart rotated the runtime
  // id. The persisted tab stamp and the live reader binding agree by
  // construction; check both so a tab re-registered before its persistence
  // write lands still admits.
  const durableOwner = readerStoredSessions.get(tabId) ?? tab.ownerStoredSessionId

  if (durableOwner && storedSessionIdForRuntimeId(sessionId) === durableOwner) {
    return true
  }

  return false
}

/** Register a live preview's page reader; returns an idempotent unregister.
 *  The session that owns this preview is bound at registration time, in both
 *  identity kinds when a durable (stored) owner is known — never from ambient
 *  UI state, which is whichever chat the user is LOOKING at, not the session
 *  that created the preview (#95475 review). */
export function registerPreviewPageReader(tabId: string, reader: PageReader, sessionId?: string, storedSessionId?: string): () => void {
  readers.set(tabId, reader)

  if (sessionId) {
    readerSessions.set(tabId, sessionId)
  } else {
    readerSessions.delete(tabId)
  }

  if (storedSessionId) {
    readerStoredSessions.set(tabId, storedSessionId)
  } else {
    readerStoredSessions.delete(tabId)
  }

  return () => {
    if (readers.get(tabId) === reader) {
      readers.delete(tabId)
      readerSessions.delete(tabId)
      readerStoredSessions.delete(tabId)
    }
  }
}

function windowText(
  base: Omit<PreviewReadResult, 'end' | 'start' | 'text' | 'total_chars'>,
  text: string,
  opts: PreviewReadOptions
): PreviewReadResult {
  const total = text.length
  const from = Math.max(0, Math.min(opts.start ?? 0, total))
  const want = Math.min(Math.max(1, opts.count ?? PREVIEW_READ_MAX_CHARS), PREVIEW_READ_MAX_CHARS)
  const to = Math.max(from, Math.min(from + want, total))

  return { ...base, end: to, start: from, text: text.slice(from, to), total_chars: total }
}

function tabIdFromPreviewPane(paneId: string | undefined): null | string {
  if (!paneId?.startsWith(`${PREVIEW_TILE_PREFIX}:`)) {
    return null
  }

  return paneId.slice(PREVIEW_TILE_PREFIX.length + 1)
}

/** Active preview tab in a layout zone, if that tab is still open. */
function openTabInGroup(groupId: null | string, tabs: PreviewTab[]): null | PreviewTab {
  const tree = $layoutTree.get()

  if (!tree || !groupId) {
    return null
  }

  const tabId = tabIdFromPreviewPane(findGroup(tree, groupId)?.active)

  if (!tabId) {
    return null
  }

  return tabs.find(tab => tab.id === tabId) ?? null
}

/**
 * The preview the user is looking at: hovered zone, else focused zone, else
 * the store. A focused zone that is still the pre-open zone does not override
 * an explicit open living in a different group — that is follow()'s clobber,
 * not a look.
 */
export function resolveActivePreviewTab(tabs: PreviewTab[] = $previewTabs.get()): null | PreviewTab {
  if (tabs.length === 0) {
    return null
  }

  const hovered = openTabInGroup($hoveredTreeGroup.get(), tabs)

  if (hovered) {
    return hovered
  }

  const focusedId = $activeTreeGroup.get()
  const focused = openTabInGroup(focusedId, tabs)
  const openIds = tabs.map(tab => tab.id)

  if (focused && !explicitOpenBlocksZone(focusedId, openIds)) {
    return focused
  }

  return tabs.find(tab => tab.id === $rightRailActiveTabId.get()) ?? tabs[0] ?? null
}

function tabSummary(tab: PreviewTab): PreviewReadTabSummary {
  return { id: tab.id, kind: tab.target.kind, label: tab.target.label, url: tab.target.url }
}

function zoneMeta(tab: PreviewTab, tabs: PreviewTab[]): { active_tab_id?: string; tabs?: PreviewReadTabSummary[] } {
  if (tabs.length < 2) {
    return {}
  }

  return { active_tab_id: tab.id, tabs: tabs.map(tabSummary) }
}

function withMultiNote(note: string | undefined, multi: boolean): string | undefined {
  if (!multi) {
    return note
  }

  const extra = 'Multiple preview tabs are open; this read used the hovered or focused preview (see tabs).'

  return note ? `${note} ${extra}` : extra
}

/** Read the preview the user is looking at. Null only when no tab is open at all. */
export async function readActivePreview(opts: PreviewReadOptions = {}): Promise<null | PreviewReadResult> {
  const tabs = $previewTabs.get()
  const tab = resolveActivePreviewTab(tabs)

  if (!tab) {
    return null
  }

  const { target } = tab
  const reader = readers.get(tab.id)
  const multi = tabs.length > 1
  const meta = zoneMeta(tab, tabs)

  if (reader) {
    try {
      const page = await reader()

      // Say it on the page. Reading is by far the cheapest thing the agent
      // does — a few hundredths of a second against a model round trip either
      // side of it — so a run of reads used to leave the pane dark for the
      // twenty seconds it took to page through a document, immediately after
      // the one moment that showed anything.
      nudgeOverlay('read')

      return windowText(
        {
          ...meta,
          kind: target.kind,
          note: withMultiNote(undefined, multi),
          path: target.path,
          title: page.title || target.label,
          url: page.url || target.url
        },
        page.text,
        opts
      )
    } catch {
      // Webview not ready (still booting / just navigated) — fall through to
      // the identity answer, whose note says to retry.
    }
  }

  // No live webview behind the tab (a file peek, an artifact, or a page still
  // booting): answer with the tab's identity so the agent knows what's on
  // screen and which of its own tools reads the content directly.
  const identity =
    target.kind === 'file'
      ? 'File preview — read the file itself with read_file.'
      : target.kind === 'artifact'
        ? 'Generated artifact — its content is in the conversation that produced it.'
        : 'The page has not finished loading — retry in a moment.'

  return windowText(
    {
      ...meta,
      kind: target.kind,
      note: withMultiNote(identity, multi),
      path: target.path,
      title: target.label,
      url: target.url
    },
    '',
    opts
  )
}
