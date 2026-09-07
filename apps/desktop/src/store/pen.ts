/** Renderer doors for the pen.dev canvas pane. Main owns documents and ties. */

import { atom } from 'nanostores'

import { closePenCanvasTile, hidePenCanvasTile, openPenCanvasTile, penCanvasTileOpen } from '@/app/chat/pen-tile'
import { ensurePenWebview } from '@/app/chat/pen-webview'
import type { PenStatus, PenToolResult } from '@/global'
import { translateNow } from '@/i18n'
import { notifyError } from '@/store/notifications'
import { $selectedStoredSessionId, $sessions } from '@/store/session'

export async function refreshPenStatus(): Promise<PenStatus | null> {
  const pen = window.hermesDesktop?.pen

  if (!pen) {
    return null
  }

  try {
    return await pen.status()
  } catch {
    return null
  }
}

export async function openPenCanvas(
  options: { name?: string; path?: string } = {},
  sessionId?: null | string
) {
  const pen = window.hermesDesktop?.pen

  if (!pen) {
    return null
  }

  try {
    // Agent route (always real) beats the selected atom, which is null in a draft.
    const tieTo = sessionId ?? $selectedStoredSessionId.get() ?? undefined
    let name = options.name

    if (!name && !options.path && tieTo) {
      const title = $sessions.get().find(s => s.id === tieTo)?.title?.trim()

      if (title) {
        name = title.slice(0, 60)
      }
    }

    const { doc, url } = await pen.open({
      ...options,
      name,
      sessionId: tieTo
    })

    if (doc && url) {
      ensurePenWebview(url)

      const focused = $selectedStoredSessionId.get()

      if (!focused || focused === tieTo || penCanvasTileOpen()) {
        openPenCanvasTile({ docId: doc.docId, title: doc.displayName || 'Canvas', url })
      }
    }

    return doc
  } catch (error) {
    notifyError(error, translateNow('pen.openFailed'))

    return null
  }
}

export async function runPenTool(name: string, payload?: Record<string, unknown>): Promise<PenToolResult> {
  const pen = window.hermesDesktop?.pen

  if (!pen) {
    return { success: false, error: 'pen canvas is only available in the Hermes desktop app' }
  }

  try {
    return await pen.tool(name, payload)
  } catch (error) {
    return { success: false, error: error instanceof Error ? error.message : String(error) }
  }
}

export const $penLibraryOpen = atom(false)

export function openPenLibrary(): void {
  $penLibraryOpen.set(true)
}

async function refreshPenSessionSuggestion(sessionId: null | string): Promise<void> {
  const { refreshPenSessionSuggestion: refresh } = await import('@/store/suggestion-providers/pen')

  await refresh(sessionId)
}

export async function restorePenCanvas(sessionId: string): Promise<boolean> {
  const pen = window.hermesDesktop?.pen

  if (!pen?.restore) {
    return false
  }

  const restored = await pen.restore(sessionId).catch(() => null)

  if (!restored) {
    return false
  }

  const docId = restored.doc?.docId ?? restored.docId
  const url = restored.url

  if (docId && url) {
    ensurePenWebview(url)
    openPenCanvasTile({ docId, title: restored.doc?.displayName || 'Canvas', url })

    return true
  }

  return false
}

/** Follow the focused chat's .pen. Same file keeps the guest; no file hides the pane. */
export function watchPenSession(): () => void {
  const pen = window.hermesDesktop?.pen

  if (!pen?.session) {
    return () => {}
  }

  let applied: null | string = null

  const sync = async (sessionId: null | string) => {
    if (sessionId === applied) {
      return
    }

    const wasDraft = applied === null

    applied = sessionId

    if (!sessionId) {
      return
    }

    if (wasDraft && penCanvasTileOpen()) {
      await pen.adopt?.(sessionId).catch(() => {})
      void refreshPenSessionSuggestion(sessionId)

      return
    }

    const entry = await pen.session(sessionId).catch(() => null)

    if (applied !== sessionId) {
      return
    }

    if (entry && !entry.closed) {
      await restorePenCanvas(sessionId)
    } else if (penCanvasTileOpen()) {
      hidePenCanvasTile()
    }

    void refreshPenSessionSuggestion(sessionId)
  }

  void sync($selectedStoredSessionId.get())

  const offEvents =
    pen.onEvent?.(({ event }) => {
      if (event === 'close-document') {
        void pen.status?.().then(status => {
          if (!status?.openDocuments.length) {
            closePenCanvasTile()
          }
        })
      }

      if (event === 'open-document' || event === 'close-document') {
        void refreshPenSessionSuggestion($selectedStoredSessionId.get())
      }
    }) ?? (() => {})

  const offSession = $selectedStoredSessionId.subscribe(sessionId => void sync(sessionId ?? null))

  return () => {
    offEvents()
    offSession()
  }
}
