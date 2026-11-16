import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'

/**
 * Chats whose newest reply is paused for an automatic continuation. Something
 * stopped the turn to apply a change and will go on with it by itself: a host
 * plugin interrupts a sandboxed turn to apply a grant (`session.interrupt` with
 * `resume`), restarts or reloads the sandbox, then continues the reply with a
 * hidden prompt. Core reports that turn's end as `message.complete`
 * `{status: 'interrupted', paused: true}`.
 *
 * A pause has two phases. `awaiting`: until the continuation's turn starts, the
 * reply reads "Continuing…" rather than "Response stopped". `continuing`: the
 * continuation runs as a turn of its own and the reply goes on. Through both,
 * the composer holds a send in the chat's queue and sends it once the
 * continuation has completed: never into it (a steer, or the gateway's
 * interrupt for a plain send) and not ahead of it. Keyed by stored session id:
 * a sandbox restart gives the chat a new runtime session before the
 * continuation arrives. The continuation's completion ends the pause; so does
 * PAUSE_CAP_MS without a start or a completion, and a never-continued reply then
 * reads as stopped again, which it is.
 */
export const PAUSE_CAP_MS = 120_000

type PausePhase = 'awaiting' | 'continuing'

const $pausedReplies = atom<Readonly<Record<string, PausePhase>>>({})
const caps = new Map<string, ReturnType<typeof setTimeout>>()

function setPhase(chatId: string, phase: PausePhase, capMs: number): void {
  clearTimeout(caps.get(chatId))
  caps.set(
    chatId,
    setTimeout(() => endReplyPause(chatId), capMs)
  )

  if ($pausedReplies.get()[chatId] !== phase) {
    $pausedReplies.set({ ...$pausedReplies.get(), [chatId]: phase })
  }
}

export function markReplyPaused(chatId: null | string | undefined, capMs = PAUSE_CAP_MS): void {
  if (chatId) {
    setPhase(chatId, 'awaiting', capMs)
  }
}

/** The paused chat's next turn started: the reply goes on, and sends still wait for that turn to end. */
export function continueReplyPause(chatId: null | string | undefined, capMs = PAUSE_CAP_MS): void {
  if (chatId && $pausedReplies.get()[chatId] === 'awaiting') {
    setPhase(chatId, 'continuing', capMs)
  }
}

/** A turn of the chat ended: a pause whose continuation was running is over. */
export function endContinuedReplyPause(chatId: null | string | undefined): void {
  if (chatId && $pausedReplies.get()[chatId] === 'continuing') {
    endReplyPause(chatId)
  }
}

export function endReplyPause(chatId: null | string | undefined): void {
  if (!chatId) {
    return
  }

  clearTimeout(caps.get(chatId))
  caps.delete(chatId)

  if ($pausedReplies.get()[chatId]) {
    const { [chatId]: _ended, ...rest } = $pausedReplies.get()
    $pausedReplies.set(rest)
  }
}

type ChatIds = (null | string | undefined)[]

const inPhase = (paused: Readonly<Record<string, PausePhase>>, chatIds: ChatIds, phase?: PausePhase) =>
  chatIds.some(id => Boolean(id && paused[id] && (!phase || paused[id] === phase)))

/** Whether any of these ids (a chat's stored id, its runtime id) names a paused chat (either phase): sends wait. */
export const isReplyPaused = (...chatIds: ChatIds): boolean => inPhase($pausedReplies.get(), chatIds)

/** Whether a paused chat still waits for its continuation to start: its reply reads "Continuing…". */
export const isReplyAwaitingContinuation = (...chatIds: ChatIds): boolean =>
  inPhase($pausedReplies.get(), chatIds, 'awaiting')

export function useReplyPaused(...chatIds: ChatIds): boolean {
  return inPhase(useStore($pausedReplies), chatIds)
}

export function useReplyAwaitingContinuation(...chatIds: ChatIds): boolean {
  return inPhase(useStore($pausedReplies), chatIds, 'awaiting')
}

/** Every paused chat, for a drainer that walks all queues (use-background-queue-drain.ts). */
export function usePausedChats(): Readonly<Record<string, PausePhase>> {
  return useStore($pausedReplies)
}
