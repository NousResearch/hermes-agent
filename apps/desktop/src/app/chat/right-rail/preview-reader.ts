/**
 * PREVIEW READER — the read_preview tool's window into the preview pane, the
 * preview analog of the terminal's buffer registry (see right-sidebar/
 * terminal/buffer.ts).
 *
 * A URL/HTML preview renders in a sandboxed <webview> owned by PreviewPane;
 * that pane registers a PAGE READER here (url + title + rendered text), keyed
 * by tab id. `readActivePreview` resolves the tab the user is looking at (or
 * the one tools just opened) and owns the windowing: a registered reader
 * answers with the live page's text; a tab with no reader (a file peek, an
 * artifact) still answers with its identity and a note pointing the agent at
 * the tool that reads that content directly (read_file / the conversation's
 * artifact).
 */

import { findGroup } from '@/components/pane-shell/tree/model'
import { $activeTreeGroup, $hoveredTreeGroup, $layoutTree } from '@/components/pane-shell/tree/store'
import { $rightRailActiveTabId, type RightRailTabId } from '@/store/layout'
import { $previewTabs, type PreviewTab } from '@/store/preview'

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
  /** Tab id that was read (`file:…` / `url:browser-…` / `artifact:…`). */
  active_tab_id: string
  end: number
  kind: string
  note?: string
  path?: string
  start: number
  /** Open preview tabs at read time — multi-tab layouts used to desync eyes vs tool. */
  tabs: PreviewReadTabSummary[]
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

/** Must match `PREVIEW_TILE_PREFIX` in preview-tile.tsx (pane id = prefix:tabId). */
const PREVIEW_TILE_PREFIX = 'preview-tile'

const readers = new Map<string, PageReader>()

/** Register a live preview's page reader; returns an idempotent unregister. */
export function registerPreviewPageReader(tabId: string, reader: PageReader): () => void {
  readers.set(tabId, reader)

  return () => {
    if (readers.get(tabId) === reader) {
      readers.delete(tabId)
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

function tabSummary(tab: PreviewTab): PreviewReadTabSummary {
  return {
    id: tab.id,
    kind: tab.target.kind,
    label: tab.target.label,
    url: tab.target.url
  }
}

function tabIdFromPreviewPane(paneId: string | undefined): null | RightRailTabId {
  if (!paneId?.startsWith(`${PREVIEW_TILE_PREFIX}:`)) {
    return null
  }

  return paneId.slice(PREVIEW_TILE_PREFIX.length + 1) as RightRailTabId
}

/** Active preview-tile tab id in a layout group, if that tab is still open. */
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
 * Resolve which preview tab `read_preview` should serialize.
 *
 * Ladder (mirrors keyboard tab-verb eligibility: hover → focus → store):
 * 1. Hovered zone's active preview tile, if still open
 * 2. Focused (`$activeTreeGroup`) zone's active preview tile, if still open
 * 3. `$rightRailActiveTabId` when it still names an open tab
 * 4. Most recent open Browser tab (`url:browser-*`) — agent-opened pages
 * 5. First open tab
 *
 * Split layouts used to leave global `$rightRailActiveTabId` on a file tab while
 * Browser stayed visible in another zone; steps 1–2 and 4 close that gap.
 *
 * Browser tab ids are minted (`url:browser-<uuid>`), not a fixed singleton, so
 * step 4 picks the last open URL vessel rather than a hard-coded id.
 */
export function resolveActivePreviewTab(tabs: PreviewTab[] = $previewTabs.get()): null | PreviewTab {
  if (tabs.length === 0) {
    return null
  }

  const byId = (id: null | string | undefined) => (id ? (tabs.find(tab => tab.id === id) ?? null) : null)
  const lastBrowser = () => {
    for (let i = tabs.length - 1; i >= 0; i--) {
      if (tabs[i]?.target.kind === 'url') {
        return tabs[i] ?? null
      }
    }

    return null
  }

  return (
    openTabInGroup($hoveredTreeGroup.get(), tabs) ??
    openTabInGroup($activeTreeGroup.get(), tabs) ??
    byId($rightRailActiveTabId.get()) ??
    lastBrowser() ??
    tabs[0] ??
    null
  )
}

function identityNote(kind: PreviewTab['target']['kind'], multiTab: boolean): string {
  const base =
    kind === 'file'
      ? 'File preview — read the file itself with read_file.'
      : kind === 'artifact'
        ? 'Generated artifact — its content is in the conversation that produced it.'
        : 'The page has not finished loading — retry in a moment.'

  if (!multiTab) {
    return base
  }

  return `${base} Multiple preview tabs are open; this read used the focused/hovered preview (see tabs).`
}

/** Read the ACTIVE preview tab. Null only when no tab is open at all. */
export async function readActivePreview(opts: PreviewReadOptions = {}): Promise<null | PreviewReadResult> {
  const tabs = $previewTabs.get()
  const tab = resolveActivePreviewTab(tabs)

  if (!tab) {
    return null
  }

  const summaries = tabs.map(tabSummary)
  const multiTab = tabs.length > 1
  const { target } = tab
  const reader = readers.get(tab.id)
  const meta = {
    active_tab_id: tab.id,
    kind: target.kind,
    path: target.path,
    tabs: summaries
  }

  if (reader) {
    try {
      const page = await reader()
      const multiNote = multiTab
        ? 'Multiple preview tabs are open; this read used the focused/hovered preview (see tabs).'
        : undefined

      // Say it on the page. Reading is by far the cheapest thing the agent
      // does — a few hundredths of a second against a model round trip either
      // side of it — so a run of reads used to leave the pane dark for the
      // twenty seconds it took to page through a document, immediately after
      // the one moment that showed anything.
      nudgeOverlay('read')

      return windowText(
        {
          ...meta,
          note: multiNote,
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
  return windowText(
    {
      ...meta,
      note: identityNote(target.kind, multiTab),
      title: target.label,
      url: target.url
    },
    '',
    opts
  )
}
