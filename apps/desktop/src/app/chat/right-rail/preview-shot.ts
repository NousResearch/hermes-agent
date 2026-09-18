/**
 * PREVIEW CAMERA REGISTRY — the desktop_preview screenshot action's view of the
 * preview pane, the picture analog of preview-reader's page reader.
 *
 * A live browser pane registers its webview's CAMERA here, keyed by tab id;
 * `screenshotActivePreview` resolves the ACTIVE tab from the store. The PNG is
 * written by the main process (electron/preview-capture.ts), so nothing but a
 * path crosses back — a data URL would land in model context.
 *
 * The answer carries the page's host, never its URL: a preview is often opened
 * on a one-time `access_url`, and the query string is a live credential.
 */

import { isRemoteGateway } from '@/lib/media'
import { $rightRailActiveTabId } from '@/store/layout'
import { $previewTabs } from '@/store/preview'

/** Photographs the pane's guest page; resolves what the main process wrote. */
export type PreviewCamera = () => Promise<{
  height: number
  path: string
  title: string
  url: string
  width: number
}>

export interface PreviewShotResult {
  height: number
  host?: string
  kind: string
  path: string
  success: true
  title: string
  width: number
}

export interface PreviewShotFailure {
  error: string
  success: false
}

const cameras = new Map<string, PreviewCamera>()

/** Register a live preview's camera; returns an idempotent unregister. */
export function registerPreviewCamera(tabId: string, camera: PreviewCamera): () => void {
  cameras.set(tabId, camera)

  return () => {
    if (cameras.get(tabId) === camera) {
      cameras.delete(tabId)
    }
  }
}

/** The page's host alone — the rest of the URL may carry a one-time token. */
function hostOf(url: string): string {
  try {
    return new URL(url).host
  } catch {
    return ''
  }
}

const NOT_PHOTOGRAPHABLE: Record<string, string> = {
  artifact: 'That preview tab is a generated artifact, not a page — its content is in the conversation that produced it.',
  file: 'That preview tab is a file peek, not a rendered page — read the file itself with read_file.'
}

/** Photograph the ACTIVE preview tab. Fails closed: no path without a PNG. */
export async function screenshotActivePreview(): Promise<PreviewShotFailure | PreviewShotResult> {
  const tabs = $previewTabs.get()
  const tab = tabs.find(t => t.id === $rightRailActiveTabId.get()) ?? tabs[0]

  if (!tab) {
    return { error: 'No preview tab is open — open one with desktop_preview first.', success: false }
  }

  // The PNG lands on THIS disk. A remote gateway's tools cannot open that path,
  // so answering it would be a success the agent cannot use — refuse up front,
  // before a file is written.
  if (isRemoteGateway()) {
    return {
      error:
        'Preview screenshots are saved on the desktop machine, which this remote gateway cannot read — use desktop_preview read for the page text instead.',
      success: false
    }
  }

  const camera = cameras.get(tab.id)

  if (!camera) {
    return {
      error:
        NOT_PHOTOGRAPHABLE[tab.target.kind] ??
        'The preview page has not finished loading — retry in a moment.',
      success: false
    }
  }

  try {
    const shot = await camera()

    if (!shot?.path) {
      return { error: 'The preview capture produced no image.', success: false }
    }

    const host = hostOf(shot.url)

    return {
      height: shot.height,
      ...(host ? { host } : {}),
      kind: tab.target.kind,
      path: shot.path,
      success: true,
      title: shot.title || tab.target.label,
      width: shot.width
    }
  } catch (error) {
    return { error: error instanceof Error ? error.message : String(error), success: false }
  }
}
