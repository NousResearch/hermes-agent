import { atom, type WritableAtom } from 'nanostores'

import type { SessionRenderedCompletion } from './session'

// "Is the thread parked at the bottom" is owned by use-stick-to-bottom inside
// ThreadMessageList (the scroll container). That state lives only in that
// subtree, so ThreadMessageList mirrors it into these atoms for the composer,
// status stack, and floating jump button — all of which render OUTSIDE the thread.
//
// `$threadScrolledUp` dims the composer / status stack; `$threadJumpButtonVisible`
// shows the floating jump control. Both track `!isAtBottom` today, but stay
// separate so their thresholds can diverge again without touching consumers.
export const $threadScrolledUp = atom(false)
export const $threadJumpButtonVisible = atom(false)

// Every mounted transcript surface (primary or tile) registers its durable
// session key and own bottom state. Pane tabs unmount inactive content, so this
// is also the source of truth for whether a transcript is genuinely painted.
// Keep tokens per mount: split layouts may show the same session more than once.
interface TranscriptSurfaceState {
  atBottom: boolean
  rendered: null | SessionRenderedCompletion
}

const transcriptSurfaces = new Map<string, Map<symbol, TranscriptSurfaceState>>()

export const $visibleTranscriptSessionIds = atom<string[]>([])

function publishVisibleTranscriptSessions(force = false): void {
  const visible = [...transcriptSurfaces]
    .filter(([, surfaces]) => [...surfaces.values()].some(surface => surface.atBottom))
    .map(([sessionId]) => sessionId)

  const current = $visibleTranscriptSessionIds.get()

  if (force || current.length !== visible.length || current.some((sessionId, index) => sessionId !== visible[index])) {
    $visibleTranscriptSessionIds.set(visible)
  }
}

export interface TranscriptSurfaceRegistration {
  dispose: () => void
  setAtBottom: (atBottom: boolean) => void
  setRenderedCompletion: (rendered: null | SessionRenderedCompletion) => void
}

export function registerTranscriptSurface(sessionId: string, atBottom: boolean): TranscriptSurfaceRegistration {
  const token = Symbol(sessionId)
  const surfaces = transcriptSurfaces.get(sessionId) ?? new Map<symbol, TranscriptSurfaceState>()

  surfaces.set(token, { atBottom, rendered: null })
  transcriptSurfaces.set(sessionId, surfaces)
  publishVisibleTranscriptSessions()

  return {
    dispose: () => {
      const current = transcriptSurfaces.get(sessionId)

      current?.delete(token)

      if (current?.size === 0) {
        transcriptSurfaces.delete(sessionId)
      }

      publishVisibleTranscriptSessions()
    },
    setAtBottom: next => {
      const current = transcriptSurfaces.get(sessionId)

      const surface = current?.get(token)

      if (!current || !surface || surface.atBottom === next) {
        return
      }

      current.set(token, { ...surface, atBottom: next })
      publishVisibleTranscriptSessions()
    },
    setRenderedCompletion: rendered => {
      const current = transcriptSurfaces.get(sessionId)
      const surface = current?.get(token)

      if (!current || !surface || surface.rendered === rendered) {
        return
      }

      current.set(token, { ...surface, rendered })
      publishVisibleTranscriptSessions(true)
    }
  }
}

export function visibleRenderedTranscriptCompletions(): Array<SessionRenderedCompletion & { sessionId: string }> {
  const rendered: Array<SessionRenderedCompletion & { sessionId: string }> = []

  for (const [sessionId, surfaces] of transcriptSurfaces) {
    for (const surface of surfaces.values()) {
      if (surface.atBottom && surface.rendered) {
        rendered.push({ ...surface.rendered, sessionId })
      }
    }
  }

  return rendered
}

export function transcriptIsVisibleAtBottom(sessionId: string | null | undefined): boolean {
  return Boolean(sessionId && $visibleTranscriptSessionIds.get().includes(sessionId))
}

// Skip no-op writes so subscribers don't churn on every scroll tick.
const setter = (target: WritableAtom<boolean>) => (value: boolean) => {
  if (target.get() !== value) {
    target.set(value)
  }
}

const setScrolledUp = setter($threadScrolledUp)
const setJumpButtonVisible = setter($threadJumpButtonVisible)

export const setThreadAtBottom = (isAtBottom: boolean) => {
  setScrolledUp(!isAtBottom)
  setJumpButtonVisible(!isAtBottom)
}

export const resetThreadScroll = () => setThreadAtBottom(true)

// Cross-component bridge: the jump button lives by the composer, the viewport's
// `scrollToBottom` lives inside the thread. The bridge registers a handler; the
// button fires it. Mirrors the composer focus/insert emitter pattern.
const handlers = new Set<() => void>()

export const onScrollToBottomRequest = (handler: () => void) => {
  handlers.add(handler)

  return () => void handlers.delete(handler)
}

export const requestScrollToBottom = () => handlers.forEach(handler => handler())

// Inline edit grows a sticky human bubble. Fire on pointerdown so the viewport
// escapes stick-to-bottom before focus/layout; close clears the edit flag when
// the inline composer unmounts.
const editOpenHandlers = new Set<() => void>()
const editCloseHandlers = new Set<() => void>()

export const onThreadEditOpen = (handler: () => void) => {
  editOpenHandlers.add(handler)

  return () => void editOpenHandlers.delete(handler)
}

export const notifyThreadEditOpen = () => editOpenHandlers.forEach(handler => handler())

export const onThreadEditClose = (handler: () => void) => {
  editCloseHandlers.add(handler)

  return () => void editCloseHandlers.delete(handler)
}

export const notifyThreadEditClose = () => editCloseHandlers.forEach(handler => handler())
