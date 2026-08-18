/**
 * PEN CANVAS — pen.dev design documents hosted by hermes.
 *
 * The canvas is a LAYOUT-TREE PANE (src/app/chat/pen-tile.tsx) hosting the
 * user's installed pen.dev editor in a <webview> on hermes-pen://. Main owns
 * the documents (create/serve/save/session ties); this store is the renderer's
 * doors: open/close/status, the agent bridge's tool runner, the session
 * follower, and the library dialog's state.
 */

import { atom } from 'nanostores'

import type { PenStatus, PenToolResult } from '@/global'
import { translateNow } from '@/i18n'
import { notifyError } from '@/store/notifications'
import { projectIdForCwd } from '@/store/projects'
import { $selectedStoredSessionId, $sessions } from '@/store/session'

/** The project owning a session's cwd — the scope a canvas belongs to when
 *  the chat lives in a project. Null outside projects; everything then keys
 *  per-session exactly as before. */
function projectIdForSession(sessionId: null | string | undefined): null | string {
  if (!sessionId) {
    return null
  }

  const cwd = $sessions.get().find(s => s.id === sessionId)?.cwd

  return cwd ? projectIdForCwd(cwd) : null
}

import { closePenCanvasTile, openPenCanvasTile, penCanvasTileOpen } from '@/app/chat/pen-tile'

/** pen.dev host availability — drives the ⌘K rows' enabled state. Refreshed on
 *  demand, not polled. */
export const $penStatus = atom<PenStatus | null>(null)

export async function refreshPenStatus(): Promise<PenStatus | null> {
  const pen = window.hermesDesktop?.pen

  if (!pen) {
    return null
  }

  try {
    const status = await pen.status()

    $penStatus.set(status)

    return status
  } catch {
    return null
  }
}

/** Open a canvas pane: a .pen file when `path` is given, else a fresh library
 *  document (blank canvas by default; pass a template like `shadcn` for a
 *  design-kit start). Re-opening an open document re-fronts its pane.
 *
 *  The canvas is tied to the session that opened it, so it comes back with
 *  that chat — on a later switch, or a later launch. */
export async function openPenCanvas(
  options: { name?: string; path?: string; template?: string } = {},
  sessionId?: null | string
) {
  const pen = window.hermesDesktop?.pen

  if (!pen) {
    return null
  }

  try {
    // The tie target: an explicit session (the agent's route — always real)
    // beats the selected atom, which is NULL in a draft chat — the silent
    // hole that produced untied canvases and a reopen pill that never fired.
    const tieTo = sessionId ?? $selectedStoredSessionId.get() ?? undefined

    // Friendly names, the way sessions get auto titles: a NEW canvas (no
    // path, no explicit name) borrows the chat's title — the same "derive
    // from intent, instantly, free" stage session titling starts with. The
    // agent's opens pass an explicit name; a title-less draft stays
    // Untitled N and the library rename remains the user's override.
    let name = options.name

    if (!name && !options.path && tieTo) {
      const title = $sessions.get().find(s => s.id === tieTo)?.title?.trim()

      if (title) {
        name = title.slice(0, 60)
      }
    }

    const { doc, url } = await pen.open({ ...options, name, projectId: projectIdForSession(tieTo) ?? undefined, sessionId: tieTo })

    if (doc && url) {
      openPenCanvasTile({ docId: doc.docId, title: doc.displayName || 'Canvas', url })
    }

    // Refresh status after an open: primes the pane tab's pen.dev icon (main
    // caches it lazily) and flips openDocuments for the pills.
    void refreshPenStatus()

    return doc
  } catch (error) {
    notifyError(error, translateNow('pen.openFailed'))

    return null
  }
}

/** Run a pen design tool (execute / get_app_state / get_guidelines / …)
 *  against the live canvas. The agent bridge routes through here so the tool
 *  works wherever the CLIENT is, remote backends included. */
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

/** The canvas library (~/.hermes/pens) — browse, reopen, delete. Opened from
 *  ⌘K; the dialog itself is mounted once by the contrib root. */
export const $penLibraryOpen = atom(false)

export function openPenLibrary(): void {
  $penLibraryOpen.set(true)
}

/** The reopen pill lives in the suggestion provider, which already imports
 *  this module — so it's pulled in lazily to keep the cycle from becoming a
 *  load-order problem. */
async function refreshPenSessionSuggestion(sessionId: null | string): Promise<void> {
  const { refreshPenSessionSuggestion: refresh } = await import('@/store/suggestion-providers/pen')

  await refresh(sessionId)
}

/** Restore a session's canvas into a pane (used by the session watcher and
 *  the reopen pill). */
export async function restorePenCanvas(sessionId: string): Promise<boolean> {
  const pen = window.hermesDesktop?.pen

  if (!pen?.restore) {
    return false
  }

  const restored = await pen.restore(sessionId, projectIdForSession(sessionId) ?? undefined).catch(() => null)

  if (!restored) {
    return false
  }

  const docId = restored.doc?.docId ?? restored.docId
  const url = restored.url

  if (docId && url) {
    openPenCanvasTile({ docId, title: restored.doc?.displayName || 'Canvas', url })
    void refreshPenStatus()

    return true
  }

  return false
}

/**
 * Keep the canvas tied to the chat you're looking at.
 *
 * A canvas belongs to a session, so switching sessions swaps it: the new
 * session's canvas is restored, and a session with no canvas closes whatever
 * was on screen rather than inheriting someone else's document. That's also
 * what makes it survive a restart — on mount the active session is asked what
 * it had, so a canvas comes back with its chat instead of being re-requested.
 *
 * ONE canvas pane, by design. This watcher is what makes that correct rather
 * than confusing: the pane always shows the active session's canvas, never a
 * stale one from another chat.
 */
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

    // No session yet (fresh draft): leave whatever is open alone rather than
    // yanking the canvas out from under a draft that's about to get an id.
    if (!sessionId) {
      return
    }

    // The draft just became a real session with a canvas already on screen —
    // the canvas was opened BEFORE the id existed, so no tie was recorded.
    // Adopt it now: this is the same chat, promoted, and losing the tie here
    // is how canvases silently detached from their conversations.
    if (wasDraft && penCanvasTileOpen()) {
      await pen.adopt?.(sessionId, projectIdForSession(sessionId) ?? undefined).catch(() => {})
      void refreshPenSessionSuggestion(sessionId)

      return
    }

    const entry = await pen.session(sessionId, projectIdForSession(sessionId) ?? undefined).catch(() => null)

    // Guard against an out-of-order answer: the user may have switched again
    // while this was in flight.
    if (applied !== sessionId) {
      return
    }

    if (entry && !entry.closed) {
      // Swap, not accumulate: fold the previous session's canvas first (its
      // tie survives — `keep`), then bring in this session's. Without this,
      // A→B with canvases on both sides leaves two panes on screen.
      if (penCanvasTileOpen()) {
        await pen.close?.({ keep: true }).catch(() => {})
      }

      await restorePenCanvas(sessionId)
    } else if (penCanvasTileOpen()) {
      // `keep` — this is a swap between sessions, not the user closing the
      // canvas, so the previous session keeps its tie. A closed tie lands
      // here too: the canvas stays attached but stays PUT AWAY until the
      // reopen pill (or the library) brings it back.
      await pen.close?.({ keep: true }).catch(() => {})
    }

    // The reopen pill mirrors this state: offered when the session has a
    // canvas that isn't on screen, withdrawn once it is.
    void refreshPenSessionSuggestion(sessionId)
  }

  void sync($selectedStoredSessionId.get())

  // Host-side document lifecycle → tab list. close-document prunes the pane
  // (the ✕, the agent's close, a delete — all converge here), and both edges
  // re-evaluate the reopen pill.
  const offEvents =
    pen.onEvent?.(({ event, payload }) => {
      const docId = (payload as { docId?: string } | null)?.docId

      if (event === 'close-document' && docId) {
        closePenCanvasTile(docId)
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
