import { atom } from 'nanostores'

/**
 * Live, in-flight turn state captured from the event stream (no backend).
 *
 * The chat runtime already receives `message.*` and `tool.*` events; here we
 * record just their timing + streamed reply per session. `buildLiveTrace`
 * (app/agents) stitches this into a TraceDoc so the waterfall can render the
 * current turn live, then the DB trace folds in once it settles.
 */

export interface LiveToolEvent {
  id: string
  name: string
  start: number
  end?: number
  status: 'error' | 'ok' | 'running'
}

export interface LiveTurn {
  busy: boolean
  turnStart: number
  tools: LiveToolEvent[]
  /** Streamed assistant text for the CURRENT round (reset each message.start).
   *  After the final round this is the turn's reply — shown on the trailing
   *  llm span so selecting it reveals what the turn produced. */
  replyText: string
}

export const $liveTurnBySession = atom<Record<string, LiveTurn>>({})

const REPLY_CAP = 4000

function patch(sid: string, fn: (turn: LiveTurn) => LiveTurn) {
  const map = $liveTurnBySession.get()
  const existing = map[sid] ?? { busy: false, turnStart: Date.now(), tools: [], replyText: '' }
  $liveTurnBySession.set({ ...map, [sid]: fn(existing) })
}

/** Mark the turn busy without wiping it. Called on every message.start — which
 *  fires per assistant message AND for synthetic re-entries (async-delegation
 *  completion, notifications), so it must accumulate, not reset. A new assistant
 *  message means a fresh reply run, so reset just the streamed text (not tools). */
export function liveTurnStart(sid: string) {
  patch(sid, turn => ({ ...turn, busy: true, replyText: '' }))
}

/** Reset for a brand-new user turn (the real boundary — call it on submit). */
export function liveTurnReset(sid: string, at = Date.now()) {
  $liveTurnBySession.set({
    ...$liveTurnBySession.get(),
    [sid]: { busy: true, turnStart: at, tools: [], replyText: '' }
  })
}

/** Accumulate streamed assistant text for the current round (message.delta). */
export function liveTurnAppendText(sid: string, delta: string) {
  if (!delta) {
    return
  }

  patch(sid, turn => ({ ...turn, replyText: (turn.replyText + delta).slice(-REPLY_CAP) }))
}

export function liveTurnEnd(sid: string) {
  const map = $liveTurnBySession.get()

  if (!map[sid]?.busy) {
    return
  }

  patch(sid, turn => ({ ...turn, busy: false }))
}

export function liveToolStart(sid: string, id: string, name: string, at = Date.now()) {
  patch(sid, turn => {
    if (turn.tools.some(t => t.id === id && t.end === undefined)) {
      return turn // already tracking this running tool
    }

    return { ...turn, busy: true, tools: [...turn.tools, { id, name, start: at, status: 'running' }] }
  })
}

export function liveToolComplete(sid: string, id: string, status: 'error' | 'ok', at = Date.now()) {
  patch(sid, turn => {
    let patched = false

    const tools = turn.tools.map(t => {
      if (!patched && t.id === id && t.end === undefined) {
        patched = true

        return { ...t, end: at, status }
      }

      return t
    })

    return { ...turn, tools }
  })
}

