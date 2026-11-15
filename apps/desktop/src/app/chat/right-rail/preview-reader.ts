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
 * directly (read_file / the conversation's artifact). A browser-hosted
 * Desktop renders URLs in a sandboxed iframe it can never read, so its URL
 * tabs answer `surface: 'iframe'` with a note naming web_extract instead.
 */

import { hostHas } from '@/lib/host-capabilities'
import { type PreviewTab, previewTabsFor } from '@/store/preview'
import type { PreviewOwner } from '@/store/preview-ownership'

import { resolveActivePreviewTab } from './preview-active-tab'
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
  /** Set to 'iframe' when the page renders in the browser-hosted Desktop's
   *  sandboxed iframe, whose text this renderer cannot read. */
  surface?: 'iframe'
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

/** A browser host mounts URL previews in a sandboxed (opaque-origin) iframe:
 *  no page reader can ever register, so waiting would never help. */
const IFRAME_UNREADABLE_NOTE =
  "Page text isn't readable in browser-hosted Desktop: the page renders in a sandboxed cross-origin iframe. Read this URL with web_extract or browser_exec instead."

/** Default + hard cap on one read — a page's innerText can be megabytes, and
 *  this crosses the gateway into model context. Page with start/count. */
export const PREVIEW_READ_MAX_CHARS = 24_000

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

/** Read the preview the user is looking at, among the tabs `sessionId` (the
 *  requesting session; default the focused one) can see. Null only when that
 *  session has no tab open at all. */
export async function readActivePreview(
  opts: PreviewReadOptions = {},
  sessionId?: PreviewOwner
): Promise<null | PreviewReadResult> {
  const tabs = previewTabsFor(sessionId)
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
      nudgeOverlay('read', sessionId)

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

  // No live webview behind the tab (a file peek, an artifact, a page still
  // booting, or a browser host's iframe): answer with the tab's identity so the
  // agent knows what's on screen and which of its own tools reads the content.
  const iframe = target.kind === 'url' && !hostHas('webviewPreview')

  const identity =
    target.kind === 'file'
      ? 'File preview — read the file itself with read_file.'
      : target.kind === 'artifact'
        ? 'Generated artifact — its content is in the conversation that produced it.'
        : iframe
          ? IFRAME_UNREADABLE_NOTE
          : 'The page has not finished loading — retry in a moment.'

  return windowText(
    {
      ...meta,
      kind: target.kind,
      note: withMultiNote(identity, multi),
      path: target.path,
      ...(iframe ? { surface: 'iframe' as const } : {}),
      title: target.label,
      url: target.url
    },
    '',
    opts
  )
}
