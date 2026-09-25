/**
 * Import from the web onto the pen canvas.
 *
 * The capture itself runs in main (`electron/pen/web-import.ts`) against the
 * preview pane's <webview> guest; this store is the renderer's view of it —
 * which page is being picked from, what is picked, how far an import is — and
 * the two doors that start one: the strip's control and the agent's
 * `pen_canvas(action='import')`. Both end at `runPenImport`, which makes sure
 * the chat has a canvas before the payload lands.
 */

import { atom } from 'nanostores'

import { penCanvasTileOpen, penCanvasTileVisible, revealPenCanvasTile } from '@/app/chat/pen-tile'
import { activePreviewImport, previewImportHandle } from '@/app/chat/right-rail/preview-import'
import { openAgentPreview } from '@/app/session/hooks/open-agent-preview'
import type { PenImportOptions, PenImportPick, PenImportResult } from '@/global'
import { translateNow } from '@/i18n'
import { hostOf } from '@/lib/pen-web-import-intent'
import { notify, notifyError } from '@/store/notifications'
import { openPenCanvas, restorePenCanvas } from '@/store/pen'
import { $selectedStoredSessionId } from '@/store/session'

export interface PenImportState {
  /** The preview tab whose guest the picker is on. */
  tabId: string
  guestId: number
  /** Crosshair up or selection live (pen.dev keeps the pick live until Esc / import). */
  picking: boolean
  pick: PenImportPick | null
  /** Capture progress in [0, 1] while an import runs; null when idle. */
  progress: null | number
}

export const $penImport = atom<null | PenImportState>(null)

function update(patch: Partial<PenImportState>): void {
  const current = $penImport.get()

  if (current) {
    $penImport.set({ ...current, ...patch })
  }
}

function guestOf(tabId: string): number | undefined {
  return previewImportHandle(tabId)?.guestId()
}

/** Toggle pen.dev's crosshair on the tab's page. */
export async function togglePenImportPick(tabId: string): Promise<void> {
  const pen = window.hermesDesktop?.pen
  const guestId = guestOf(tabId)

  if (!pen || guestId === undefined) {
    return
  }

  const current = $penImport.get()
  const active = current?.tabId === tabId && current.picking

  if (active) {
    $penImport.set(null)
    await pen.import.pick(guestId, false)

    return
  }

  $penImport.set({ guestId, pick: null, picking: true, progress: null, tabId })
  await pen.import.pick(guestId, true)
}

export function hoverPenImportPath(tabId: string, index: null | number): void {
  const pen = window.hermesDesktop?.pen
  const guestId = guestOf(tabId)

  if (pen && guestId !== undefined) {
    void pen.import.hoverPathEntry(guestId, index)
  }
}

export function selectPenImportPath(tabId: string, index: number): void {
  const pen = window.hermesDesktop?.pen
  const guestId = guestOf(tabId)

  if (pen && guestId !== undefined) {
    void pen.import.selectPathEntry(guestId, index)
  }
}

/** A short canvas name from the page: its title, else its host. */
export function canvasNameForPage(page: { title: string; url: string }): string | undefined {
  const title = page.title.trim()

  return title ? title.slice(0, 60) : hostOf(page.url) || undefined
}

/**
 * Capture the tab's page (or its pick / `selector`) and drop it on the chat's
 * canvas, opening one named after the page when the chat has none.
 */
export async function runPenImport(
  tabId: string,
  options: PenImportOptions = {},
  sessionId?: null | string
): Promise<PenImportResult> {
  const pen = window.hermesDesktop?.pen
  const handle = previewImportHandle(tabId)
  const guestId = handle?.guestId()

  if (!pen || !handle || guestId === undefined) {
    return { error: 'no browser page is open in the preview pane', success: false }
  }

  // A canvas the chat can see. Main may still hold a document whose pane was
  // put away, so the pane — not the document list — decides; the chat's own
  // canvas comes back first, a new one named after the page otherwise.
  let fresh = false

  if (!penCanvasTileOpen()) {
    const chat = sessionId ?? $selectedStoredSessionId.get()
    const restored = chat ? await restorePenCanvas(chat) : false

    if (!restored) {
      const doc = await openPenCanvas({ name: canvasNameForPage(handle.page()) }, chat)

      if (!doc) {
        return { error: 'could not open a canvas for this chat', success: false }
      }

      fresh = true
    }
  }

  const prior = $penImport.get()
  const live = prior?.tabId === tabId ? prior : null

  $penImport.set({ guestId, pick: live?.pick ?? null, picking: live?.picking ?? false, progress: 0, tabId })

  try {
    return await pen.import.run(guestId, { ...options, fresh })
  } catch (error) {
    return { error: error instanceof Error ? error.message : String(error), success: false }
  } finally {
    $penImport.set(null)
  }
}

/** The strip's Import / Whole page buttons: run, then tell the user how it went. */
export async function importFromPreviewStrip(tabId: string, mode: 'page' | 'selection'): Promise<void> {
  const result = await runPenImport(tabId, { mode })

  if (result.success) {
    // The picker keeps the browser in front on purpose (imports come in runs);
    // a canvas that is behind another tab or minimized gets a way over.
    notify({
      action: penCanvasTileVisible() ? undefined : { label: translateNow('pen.showCanvas'), onClick: () => void revealPenCanvasTile() },
      kind: 'success',
      message:
        result.imported === 'selection' && result.element
          ? translateNow('pen.importedElement', result.element)
          : translateNow('pen.imported')
    })

    return
  }

  const failed = translateNow('pen.importFailed')

  notifyError(new Error(result.error ?? failed), failed)
}

/**
 * Agent door: `pen_canvas(action='import', args={url?, selector?})`. Works on
 * the ACTIVE preview tab, landing `url` there first when given.
 */
export async function importActivePreviewToCanvas(
  options: PenImportOptions & { url?: string },
  sessionId: null | string
): Promise<PenImportResult & { url?: string }> {
  if (options.url && !(await openAgentPreview(options.url))) {
    return { error: `the preview pane cannot open ${options.url}`, success: false }
  }

  const active = await settledActivePreview()

  if (!active) {
    return { error: 'no browser page is open in the preview pane — pass a url, or open one with desktop_preview', success: false }
  }

  const result = await runPenImport(active.tabId, { mode: options.mode, selector: options.selector }, sessionId)

  return { ...result, url: active.handle.page().url }
}

const SETTLE_POLL_MS = 100
const SETTLE_MAX_MS = 20_000
const PICKER_END_GRACE_MS = 250

/** The active preview once its guest exists and has finished loading. */
async function settledActivePreview() {
  const deadline = Date.now() + SETTLE_MAX_MS

  while (Date.now() < deadline) {
    const active = activePreviewImport()

    if (active && active.handle.guestId() !== undefined && !active.handle.loading()) {
      return active
    }

    await new Promise(resolve => setTimeout(resolve, SETTLE_POLL_MS))
  }

  return activePreviewImport()
}

/** Mirror main's picker / progress / Enter into the store. Called once. */
export function watchPenImport(): () => void {
  const pen = window.hermesDesktop?.pen

  if (!pen) {
    return () => {}
  }

  // The SDK reports `undefined` between picker sessions too (startPicking and
  // select() restart the crosshair), so a null is only the end when nothing
  // follows it — give the next state a beat before treating it as Esc.
  let endTimer: null | ReturnType<typeof setTimeout> = null

  const offPicker = pen.import.onPicker(({ guestId, state }) => {
    const current = $penImport.get()

    if (!current || current.guestId !== guestId) {
      return
    }

    if (endTimer) {
      clearTimeout(endTimer)
      endTimer = null
    }

    if (!state) {
      endTimer = setTimeout(() => {
        endTimer = null

        const latest = $penImport.get()

        if (latest?.guestId === guestId && latest.progress === null) {
          $penImport.set(null)
        }
      }, PICKER_END_GRACE_MS)

      return
    }

    update({ pick: state.pick ?? null, picking: true })
  })

  const offAction = pen.import.onAction(({ action, guestId }) => {
    const current = $penImport.get()

    if (action === 'import' && current?.guestId === guestId && current.progress === null) {
      void importFromPreviewStrip(current.tabId, 'selection')
    }
  })

  const offProgress = pen.import.onProgress(({ fraction, guestId }) => {
    if ($penImport.get()?.guestId === guestId) {
      update({ progress: fraction })
    }
  })

  return () => {
    offPicker()
    offAction()
    offProgress()
  }
}
