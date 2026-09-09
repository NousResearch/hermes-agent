/**
 * PREVIEW READER — the read_preview tool's window into the preview pane, the
 * preview analog of the terminal's buffer registry (see right-sidebar/
 * terminal/buffer.ts).
 *
 * A URL/HTML preview renders in a sandboxed <webview> owned by PreviewPane;
 * that pane registers a PAGE READER here (url + title + rendered text), keyed
 * by tab id. `readActivePreview` resolves the ACTIVE tab from the store and
 * owns the windowing: a registered reader answers with the live page's text;
 * a tab with no reader (a file peek, an artifact) still answers with its
 * identity and a note pointing the agent at the tool that reads that content
 * directly (read_file / the conversation's artifact).
 */

import { $rightRailActiveTabId } from '@/store/layout'
import { $previewTabs } from '@/store/preview'

import { nudgeOverlay } from './preview-nudge'

export interface PreviewReadOptions {
  /** Characters to return from `start` (capped at PREVIEW_READ_MAX_CHARS). */
  count?: number
  /** 0-indexed character offset into the page text. */
  start?: number
}

export interface PreviewReadResult {
  end: number
  kind: string
  note?: string
  path?: string
  start: number
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

/** Read the ACTIVE preview tab. Null only when no tab is open at all. */
export async function readActivePreview(opts: PreviewReadOptions = {}): Promise<PreviewReadResult | null> {
  const tabs = $previewTabs.get()
  const tab = tabs.find(t => t.id === $rightRailActiveTabId.get()) ?? tabs[0]

  if (!tab) {
    return null
  }

  const { target } = tab
  const reader = readers.get(tab.id)

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
        { kind: target.kind, path: target.path, title: page.title || target.label, url: page.url || target.url },
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
      kind: target.kind,
      note:
        target.kind === 'file'
          ? 'File preview — read the file itself with read_file.'
          : target.kind === 'artifact'
            ? 'Generated artifact — its content is in the conversation that produced it.'
            : 'The page has not finished loading — retry in a moment.',
      path: target.path,
      title: target.label,
      url: target.url
    },
    '',
    opts
  )
}
