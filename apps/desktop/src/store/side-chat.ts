/**
 * Side chat (renderer side) — the state of the floating `/btw` window, and the
 * primary window's bridge back into the real `prompt.btw` path.
 *
 * The side window carries NO gateway connection. It hands each question to the
 * main process, which forwards it to the PRIMARY renderer, which calls the same
 * `prompt.btw` RPC the inline `/btw` always called and relays the `btw.complete`
 * event back. There is no second submit path and no new gateway RPC — see
 * electron/side-chat.ts for why the window is a view rather than a client.
 *
 * What the backend guarantees is what makes this window honest: `prompt.btw`
 * answers from a SNAPSHOT of the conversation (agent/side_question.py), so the
 * parent chat's history, alternation and prompt cache are untouched and an
 * aside can be asked while the main turn is still streaming.
 */

/** The parent conversation a side chat is asking about. */
export interface SideChatContext {
  /** Optional first question, from `/btw <question>`. Blank for a bare `/btw`. */
  question: string
  /** Runtime session id of the parent chat. Every ask is scoped to it. */
  sessionId: string
  /** Parent chat's title, for the window header. */
  title: string
}

/** A question typed in the side chat, on its way to the primary renderer. */
export interface SideChatAsk {
  askId: string
  sessionId: string
  text: string
}

/** An answer (or failure) travelling back to the side window. */
export interface SideChatReply {
  askId: string
  /** Set when the ask could not be answered; `text` is then empty. */
  error: string
  text: string
}

/** True when the shell exposes the side chat (desktop only). */
export function canUseSideChat(): boolean {
  return typeof window !== 'undefined' && typeof window.hermesDesktop?.sideChat?.open === 'function'
}

/**
 * Open (or re-focus) the side chat on a conversation, optionally with its first
 * question. Returns whether the shell actually took it, so `/btw` can fall back
 * to its inline behavior instead of silently swallowing the question.
 */
export async function openSideChat(context: SideChatContext): Promise<boolean> {
  if (!canUseSideChat() || !context.sessionId) {
    return false
  }

  try {
    const result = await window.hermesDesktop.sideChat.open(context)

    return result?.ok === true
  } catch {
    return false
  }
}

// ── Side window state ───────────────────────────────────────────────────────

/** One question and its answer. `pending` until a reply lands. */
export interface SideChatTurn {
  answer: string
  askId: string
  error: string
  pending: boolean
  question: string
}

export interface SideChatState {
  /** null until main pushes the parent conversation. Blocks sending. */
  context: null | SideChatContext
  draft: string
  turns: SideChatTurn[]
}

export type SideChatEvent =
  | { type: 'context'; context: SideChatContext }
  | { type: 'edit'; draft: string }
  | { type: 'reply'; reply: SideChatReply }
  | { type: 'submit'; askId: string }

export interface SideChatTransition {
  /** Payload to hand to the shell, or null for none. */
  ask: null | SideChatAsk
  state: SideChatState
}

export const initialSideChatState: SideChatState = { context: null, draft: '', turns: [] }

const startTurn = (state: SideChatState, askId: string, question: string): SideChatState => ({
  ...state,
  draft: '',
  turns: [...state.turns, { answer: '', askId, error: '', pending: true, question }]
})

/**
 * The side window's whole behavior, as a pure reducer.
 *
 * The parts worth proving are the ones that would actually break a user:
 * - A blank submit sends nothing and clears nothing, so a stray Enter can't
 *   make a half-typed follow-up disappear.
 * - A submit before the context arrives sends nothing: `prompt.btw` snapshots
 *   one specific conversation, so an ask with no session could never be answered.
 * - Concurrent asks are allowed. Each `/btw` is its own backend side agent over
 *   its own snapshot, so blocking a second question while the first is thinking
 *   would be a restriction the backend does not impose.
 * - A reply is matched by `askId`, never by position — two asides in flight
 *   finish in whatever order their agents happen to, and an unmatched reply is
 *   dropped rather than landing on an unrelated question.
 * - A NEW parent conversation resets the thread; the SAME one keeps it, so
 *   re-running `/btw` in the chat you are already asking about reads as one
 *   continuing aside.
 */
export function sideChatReducer(state: SideChatState, event: SideChatEvent): SideChatTransition {
  switch (event.type) {
    case 'context': {
      const { context } = event
      const switched = state.context?.sessionId !== context.sessionId
      const base: SideChatState = { ...state, context, draft: switched ? '' : state.draft }
      const question = context.question.trim()

      if (switched) {
        base.turns = []
      }

      if (!question) {
        return { ask: null, state: base }
      }

      // `/btw <question>` opens the window with the question already asked. The
      // ask rides the same window → main → primary road as a typed follow-up,
      // so there is exactly one path a question can travel. The id is derived
      // from the conversation and the thread length — unique within this
      // window (turns reset when the conversation changes) and deterministic,
      // which is what lets the seeding path be tested at all.
      const askId = `seed-${context.sessionId}-${base.turns.length}`

      return {
        ask: { askId, sessionId: context.sessionId, text: question },
        state: startTurn(base, askId, question)
      }
    }

    case 'edit': {
      return { ask: null, state: { ...state, draft: event.draft } }
    }

    case 'reply': {
      const { reply } = event

      if (!state.turns.some(turn => turn.askId === reply.askId)) {
        return { ask: null, state }
      }

      return {
        ask: null,
        state: {
          ...state,
          turns: state.turns.map(turn =>
            turn.askId === reply.askId
              ? { ...turn, answer: reply.text, error: reply.error, pending: false }
              : turn
          )
        }
      }
    }

    case 'submit': {
      const text = state.draft.trim()

      if (!text || !state.context) {
        return { ask: null, state }
      }

      return {
        ask: { askId: event.askId, sessionId: state.context.sessionId, text },
        state: startTurn(state, event.askId, text)
      }
    }

    default: {
      return { ask: null, state }
    }
  }
}

// ── Primary-renderer bridge ─────────────────────────────────────────────────

/**
 * Live asides this window started, keyed by the backend task id `prompt.btw`
 * returned. It is what tells a `btw.complete` event apart from the inline
 * `/btw` the transcript still renders: an answer the user is watching in the
 * side window must NOT also be spliced into the chat it deliberately stayed
 * out of.
 */
const sideChatTasks = new Map<string, { askId: string; timer: number }>()

/**
 * How long an aside may stay unanswered before the window says so.
 *
 * `btw.complete` is the ONLY thing that ever settles a bubble, and it is an
 * event, not a reply: if the backend is recycled, the socket drops, or the side
 * agent dies, nothing arrives and the question sits "Thinking…" for the rest of
 * the session with no way for the user to learn it never will. Generous enough
 * that a slow side agent over a long conversation still lands — this is a
 * backstop for the answer that is never coming, not a latency budget.
 */
const SIDE_CHAT_ANSWER_TIMEOUT_MS = 600_000

/** Send a reply to the side window. No-ops outside the desktop shell. */
export function replyToSideChat(reply: SideChatReply): void {
  window.hermesDesktop?.sideChat?.reply?.(reply)
}

/** Forget a task and cancel its timeout. */
export function releaseSideChatTask(taskId: string): void {
  const entry = sideChatTasks.get(taskId)

  if (entry) {
    window.clearTimeout(entry.timer)
    sideChatTasks.delete(taskId)
  }
}

/** Remember that `taskId`'s answer belongs to the side window, not the transcript. */
export function claimSideChatTask(taskId: string, askId: string): void {
  if (!taskId || !askId) {
    return
  }

  releaseSideChatTask(taskId)

  const timer = window.setTimeout(() => {
    sideChatTasks.delete(taskId)
    replyToSideChat({ askId, error: 'No answer came back for this side question.', text: '' })
  }, SIDE_CHAT_ANSWER_TIMEOUT_MS)

  sideChatTasks.set(taskId, { askId, timer })
}

/**
 * Route a `btw.complete` answer to the side window when that window asked for
 * it. Returns true when the answer was consumed, so the caller leaves the
 * parent transcript alone.
 *
 * An empty answer still settles the bubble — as a failure, because a side agent
 * that returned nothing is indistinguishable to the reader from one still
 * thinking, and only one of those is worth waiting for.
 */
export function deliverSideChatAnswer(taskId: string, text: string): boolean {
  const entry = taskId ? sideChatTasks.get(taskId) : undefined

  if (!entry) {
    return false
  }

  releaseSideChatTask(taskId)
  replyToSideChat(
    text.trim()
      ? { askId: entry.askId, error: '', text }
      : { askId: entry.askId, error: 'The side question came back empty.', text: '' }
  )

  return true
}

let askHandler: ((ask: SideChatAsk) => void) | null = null
let unsubscribeAsk: (() => void) | null = null

/** Register the handler that turns a side-window ask into a real `prompt.btw`. */
export function setSideChatAskHandler(fn: ((ask: SideChatAsk) => void) | null): void {
  askHandler = fn
}

function normalizeAsk(raw: unknown): null | SideChatAsk {
  if (!raw || typeof raw !== 'object') {
    return null
  }

  const record = raw as Record<string, unknown>
  const askId = typeof record.askId === 'string' ? record.askId.trim() : ''
  const sessionId = typeof record.sessionId === 'string' ? record.sessionId.trim() : ''
  const text = typeof record.text === 'string' ? record.text : ''

  if (!askId || !sessionId || !text.trim()) {
    return null
  }

  return { askId, sessionId, text }
}

/**
 * Wire the side-window → primary-renderer ask channel once. Returns a disposer.
 * Idempotent — a second call while wired is a no-op.
 */
export function initSideChatBridge(): () => void {
  const api = typeof window === 'undefined' ? undefined : window.hermesDesktop?.sideChat

  if (!api?.onAsk || unsubscribeAsk) {
    return () => {}
  }

  unsubscribeAsk = api.onAsk(raw => {
    const ask = normalizeAsk(raw)

    if (ask) {
      askHandler?.(ask)
    }
  })

  return () => {
    unsubscribeAsk?.()
    unsubscribeAsk = null
  }
}
