import type { GatewayEvent, PersistedTurn } from '@hermes/shared'
import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { graftRefreshedTailOntoBackfill } from '@/app/chat/transcript-backfill'
import { useMessageStream } from '@/app/session/hooks/use-message-stream'
import {
  preserveLocalPendingTurnMessages,
  reconcileDurableHistory
} from '@/app/session/hooks/use-session-actions/utils'
import { useSessionStateCache } from '@/app/session/hooks/use-session-state-cache'
import { getLatestSessionMessages } from '@/hermes'
import { type ChatMessage, chatMessageText, toChatMessages } from '@/lib/chat-messages'
import { resetLiveSync } from '@/store/live-sync'
import {
  setActiveSessionId,
  setAwaitingResponse,
  setBusy,
  setMessages,
  setSelectedStoredSessionId
} from '@/store/session'
import { $sessionStates, clearAllSessionStates } from '@/store/session-states'
import { clearSessionTodos } from '@/store/todos'
import type { SessionMessage } from '@/types/hermes'

import { reconcileActiveTranscript } from './use-background-sync'

/**
 * A backend-started turn (an async delegation completion) draws no prompt: its
 * durable starter is a `user` row with `display_kind: async_delegation_complete`
 * that history paints as `system`, and it carries no submission id. The live
 * stream paints sealed narration and tool cards. A refresh of a long chat whose
 * newest page starts inside an older tool fold takes the stored-id merge; the
 * turn's live copy must give way to the page's stored copy exactly once, and
 * only where the producer proved which stored rows are that turn's: the stored
 * row each live event names (`row_id` / `message_uid`, agent/live_row_identity.py),
 * or, from an older runtime, the completion receipt's row ids.
 */

vi.mock('@/hermes', async original => ({
  ...(await original<Record<string, unknown>>()),
  getLatestSessionMessages: vi.fn()
}))

const RUNTIME = 'machine-runtime'
const STORED = 'machine-stored'
const ROUNDS = 6
const noop = async () => undefined

/** The turn whose current runtime wrote a row; null for an older runtime's rows. */
type Producer = { turnId: string } | null

/** The durable uid a current runtime's rows carry (`messages.message_uid`); older rows have none. */
const uidOf = (id: number, producer: Producer) => (producer ? { message_uid: `uid-${producer.turnId}-${id}` } : {})

const user = (id: number, text: string, extra: Partial<SessionMessage> = {}): SessionMessage =>
  ({ id, role: 'user', content: text, timestamp: id, ...extra }) as SessionMessage

const narration = (id: number, text: string, calls: string[], producer: Producer = null): SessionMessage =>
  ({
    id,
    role: 'assistant',
    content: text,
    timestamp: id,
    tool_calls: calls.map(call => ({ id: call, type: 'function', function: { name: 'terminal', arguments: '{}' } })),
    ...uidOf(id, producer)
  }) as SessionMessage

const tool = (id: number, call: string, producer: Producer = null, content = 'ok'): SessionMessage =>
  ({
    id,
    role: 'tool',
    content,
    tool_call_id: call,
    tool_name: 'terminal',
    timestamp: id,
    ...uidOf(id, producer)
  }) as SessionMessage

const answer = (id: number, text: string, producer: Producer = null): SessionMessage =>
  ({
    id,
    role: 'assistant',
    content: text,
    timestamp: id,
    ...uidOf(id, producer)
  }) as SessionMessage

/** A user turn of three narrated tool rounds: rows first..first+7. */
const olderTurn = (first: number, tag: string): SessionMessage[] => [
  user(first, `${tag} prompt`),
  narration(first + 1, `${tag} look one`, [`${tag}-1`]),
  tool(first + 2, `${tag}-1`),
  narration(first + 3, `${tag} look two`, [`${tag}-2`]),
  tool(first + 4, `${tag}-2`),
  narration(first + 5, `${tag} look three`, [`${tag}-3`]),
  tool(first + 6, `${tag}-3`),
  answer(first + 7, `${tag} answer`)
]

interface MachineTurn {
  tag: string
  first: number
  starter: SessionMessage
  rows: SessionMessage[]
  receipt: PersistedTurn
  narrations: string[]
  tools: string[]
  /** Each tool's result, by its index in `tools` ('ok' when absent). */
  results?: string[]
  final: string
}

interface TurnOptions {
  prose?: string
  kind?: string
  /** Rows written by a current runtime, which gives each a `message_uid` (default), or by an older one. */
  keyed?: boolean
  narrations?: (round: number) => string
}

/** A backend-started turn: starter, then ROUNDS narrated rounds of two tools, then the answer. */
function machineTurn(first: number, tag: string, options: TurnOptions = {}): MachineTurn {
  const { prose = tag, kind = 'async_delegation_complete', keyed = true } = options
  const producer: Producer = keyed ? { turnId: tag } : null
  const starter = user(first, `[Background work finished] ${tag}`, { display_kind: kind })
  const rows: SessionMessage[] = [starter]
  const narrations: string[] = []
  const tools: string[] = []
  let id = first + 1

  for (let round = 1; round <= ROUNDS; round += 1) {
    const calls = [`${tag}-r${round}-a`, `${tag}-r${round}-b`]
    const text = options.narrations?.(round) ?? `${prose} narration ${round}`
    narrations.push(text)
    tools.push(...calls)
    rows.push(narration(id++, text, calls, producer), ...calls.map(call => tool(id++, call, producer)))
  }

  const final = `${prose} final answer`
  rows.push(answer(id, final, producer))

  return {
    tag,
    first,
    starter,
    rows,
    narrations,
    tools,
    final,
    receipt: {
      row_ids: rows.map(row => row.id as number),
      complete: true,
      user_row_id: first,
      user_row_ids: [first],
      final_assistant_row_id: id
    }
  }
}

let cache: ReturnType<typeof useSessionStateCache>
let stream: ReturnType<typeof useMessageStream>
let refresh: () => Promise<void>

function Harness() {
  const busyRef = useRef(false)
  const queryClient = useRef(new QueryClient()).current
  const requestSequenceRef = useRef(0)
  const signatureRef = useRef(new Map<string, string>())
  cache = useSessionStateCache({
    activeSessionId: RUNTIME,
    selectedStoredSessionId: STORED,
    busyRef,
    setBusy,
    setAwaitingResponse,
    setMessages
  })
  stream = useMessageStream({
    ...cache,
    queryClient,
    hydrateFromStoredSession: noop,
    refreshHermesConfig: noop,
    refreshSessions: noop
  })
  refresh = () =>
    reconcileActiveTranscript({
      activeSessionIdRef: cache.activeSessionIdRef,
      selectedStoredSessionIdRef: cache.selectedStoredSessionIdRef,
      updateSessionState: cache.updateSessionState,
      busyRef,
      requestSequenceRef,
      signatureRef,
      resolveSession: () => ({ profile: 'default' })
    })

  return null
}

type Execution = Pick<GatewayEvent, 'admission_id' | 'authority_epoch' | 'execution_generation'>

function send(type: GatewayEvent['type'], payload: GatewayEvent['payload'] = {}, execution: Execution = {}) {
  act(() => stream.handleGatewayEvent({ session_id: RUNTIME, type, payload, ...execution }))
}

async function flush() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(100)
  })
}

const messages = () => $sessionStates.get()[RUNTIME].messages

type Receipt = 'complete' | 'partial' | 'without-final' | 'none' | 'running'

interface StreamOptions {
  /** `classic`: tui_gateway `_emit('message.start')` stamps `turn_id` on the start payload.
   *  `authority`: the session authority names its execution on the envelope and stamps only
   *  its reply frames (gateway/run_turn_runner.py `_stamp_segment`).
   *  `legacy`: an older runtime that names no stream turn on any frame. */
  owner?: 'classic' | 'authority' | 'legacy'
  /** False for a host that sends no `message.interim` for narration it streamed: the session
   *  authority (gateway/run_turn_runner.py) and a managed worker publish none. */
  seal?: boolean
  rounds?: number
  displayKind?: string
}

/** The reply-frame stamp `owner` puts on segment `segment` (none for an older runtime). */
const replyStamp = (turn: MachineTurn, owner: StreamOptions['owner'], segment: number, offset?: number) =>
  owner === 'legacy' ? {} : { turn_id: turn.tag, segment, ...(offset === undefined ? {} : { offset }) }

/** The stored row a current runtime names on round `round`'s interim and tool frames
 *  (agent/live_row_identity.py): the round's narration row. An older runtime names none. */
function storedRowOf(turn: MachineTurn, owner: StreamOptions['owner'], round: number) {
  const row = turn.rows.filter(entry => entry.role === 'assistant' && Array.isArray(entry.tool_calls))[round]

  return owner === 'legacy' || !row
    ? {}
    : { row_id: row.id, ...(row.message_uid ? { message_uid: row.message_uid } : {}) }
}

/** Replays the backend's frames of `turn`'s first `rounds` rounds; tool frames carry no turn id. */
async function streamRounds(
  turn: MachineTurn,
  { owner = 'classic', seal = true, rounds = ROUNDS, displayKind }: StreamOptions = {}
) {
  const execution: Execution =
    owner === 'authority' ? { authority_epoch: 1, execution_generation: 4, admission_id: turn.tag } : {}

  const emit = (type: GatewayEvent['type'], payload: GatewayEvent['payload']) => send(type, payload, execution)

  emit('message.start', {
    ...(owner === 'classic' ? { turn_id: turn.tag } : {}),
    ...(displayKind ? { display_kind: displayKind } : {})
  })

  for (const [index, text] of turn.narrations.slice(0, rounds).entries()) {
    const stored = storedRowOf(turn, owner, index)
    emit('message.delta', { text, ...replyStamp(turn, owner, index, 0) })
    await flush()

    if (seal) {
      emit('message.interim', { text, already_streamed: true, ...replyStamp(turn, owner, index), ...stored })
    }

    for (const [slot, call] of turn.tools.slice(index * 2, index * 2 + 2).entries()) {
      emit('tool.start', { name: 'terminal', tool_id: call, args: {}, ...stored })
      emit('tool.complete', {
        name: 'terminal',
        tool_id: call,
        result: turn.results?.[index * 2 + slot] ?? 'ok',
        ...stored
      })
    }
  }

  return emit
}

async function streamMachineTurn(turn: MachineTurn, receipt: Receipt, options: StreamOptions = {}) {
  const emit = await streamRounds(turn, options)
  emit('message.delta', { text: turn.final, ...replyStamp(turn, options.owner, ROUNDS, 0) })
  await flush()

  if (receipt === 'running') {
    return
  }

  // The final answer failed to store: the receipt addresses rows only through the last tool result.
  const coverage =
    receipt === 'without-final'
      ? { ...turn.receipt, row_ids: turn.receipt.row_ids.slice(0, -1), final_assistant_row_id: undefined }
      : turn.receipt

  // A current runtime names the stored reply on the completion whether or not a receipt rides along.
  const finalRow = turn.rows.at(-1)!

  const named =
    options.owner === 'legacy' || receipt === 'without-final'
      ? {}
      : { row_id: finalRow.id, ...(finalRow.message_uid ? { message_uid: finalRow.message_uid } : {}) }

  emit('message.complete', {
    status: 'complete',
    text: turn.final,
    persisted_turn: receipt === 'none' ? undefined : { ...coverage, complete: receipt === 'complete' },
    ...named
  })
}

async function refreshWith(rows: SessionMessage[]) {
  vi.mocked(getLatestSessionMessages).mockResolvedValue({ session_id: STORED, messages: rows })
  await act(async () => {
    await refresh()
  })

  return messages()
}

const allText = (list: ChatMessage[]) => list.map(chatMessageText).join('\n')
const occurrences = (haystack: string, needle: string) => haystack.split(needle).length - 1

const toolCards = (list: ChatMessage[], id: string) =>
  list
    .flatMap(message => message.parts)
    .flatMap(part => (part.type === 'tool-call' && part.toolCallId === id ? [part] : []))

const toolOccurrences = (list: ChatMessage[], id: string) => toolCards(list, id).length

/** Ordered narration as painted, so a reorder is as visible as a duplicate. */
const paintedNarrations = (list: ChatMessage[], turn: MachineTurn) =>
  list
    .flatMap(message => message.parts)
    .flatMap(part => (part.type === 'text' ? turn.narrations.filter(text => part.text.includes(text)) : []))

function expectDrawnOnce(list: ChatMessage[], turn: MachineTurn, rounds = ROUNDS) {
  expect(paintedNarrations(list, turn)).toEqual(turn.narrations.slice(0, rounds))
  const tools = turn.tools.slice(0, rounds * 2)
  expect(tools.map(id => toolOccurrences(list, id))).toEqual(tools.map(() => 1))
  expect(occurrences(allText(list), turn.final), 'final').toBe(1)
}

/** Nothing the turn drew is lost: every narration, tool card and its result, and the answer. */
function expectNothingLost(list: ChatMessage[], turn: MachineTurn) {
  const text = allText(list)

  for (const narrationText of turn.narrations) {
    expect(occurrences(text, narrationText), narrationText).toBeGreaterThanOrEqual(1)
  }

  for (const id of turn.tools) {
    expect(
      toolCards(list, id).some(card => card.result === 'ok'),
      id
    ).toBe(true)
  }

  expect(occurrences(text, turn.final)).toBeGreaterThanOrEqual(1)
}

beforeEach(() => {
  vi.useFakeTimers()
  vi.mocked(getLatestSessionMessages).mockReset()
  setActiveSessionId(RUNTIME)
  setSelectedStoredSessionId(STORED)
})

afterEach(() => {
  cleanup()
  clearSessionTodos(RUNTIME)
  clearAllSessionStates()
  resetLiveSync()
  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  setBusy(false)
  setAwaitingResponse(false)
  setMessages([])
  localStorage.clear()
  vi.useRealTimers()
  vi.restoreAllMocks()
})

// History: two older user turns (rows 1-16). The newest page starts at row 4,
// inside the first older turn's tool fold, so the refresh merges by stored id.
const history = [...olderTurn(1, 'old-a'), ...olderTurn(9, 'old-b')]
const machine = machineTurn(17, 'delegation')
const page = [...history.slice(3), ...machine.rows]

function seed(rows: SessionMessage[]) {
  render(<Harness />)
  act(() => cache.updateSessionState(RUNTIME, state => ({ ...state, messages: toChatMessages(rows) }), STORED))
}

/** Warm reopen / resume barrier (resume-barrier-reread.graftTranscriptPage). */
const reopen = (rows: SessionMessage[], live: ChatMessage[]) =>
  reconcileDurableHistory(graftRefreshedTailOntoBackfill(toChatMessages(rows), live), live)

/** Tile, active and post-turn refresh (use-background-sync) and the tile delegate. */
const refreshed = (rows: SessionMessage[], live: ChatMessage[]) =>
  preserveLocalPendingTurnMessages(graftRefreshedTailOntoBackfill(toChatMessages(rows), live), live)

describe('a refreshed page over a settled backend-started turn', () => {
  it.each(['complete', 'partial', 'none'] as const)(
    'draws a turn with no visible starter once (receipt: %s)',
    async receipt => {
      seed(history)
      await streamMachineTurn(machine, receipt)

      const list = await refreshWith(page)

      expectDrawnOnce(list, machine)
      expect(occurrences(allText(list), 'old-a look one')).toBe(1)
      expect(occurrences(allText(list), 'old-b answer')).toBe(1)
      // The page's own starter row now shows once, as the system card it is.
      expect(list.filter(message => message.rowId === machine.first).map(message => message.role)).toEqual(['system'])
    }
  )

  it('reconciles a session-authority turn whose start names its execution only on the envelope', async () => {
    seed(history)
    await streamMachineTurn(machine, 'complete', { owner: 'authority', displayKind: 'async_delegation_complete' })

    expectDrawnOnce(await refreshWith(page), machine)
  })

  it('draws a turn whose starter the window already paints as a system row once', async () => {
    // A reopen mid-turn hydrated the starter before the stream went on.
    seed([...history, machine.starter])
    await streamMachineTurn(machine, 'complete')

    const list = await refreshWith(page)

    expectDrawnOnce(list, machine)
    expect(list.filter(message => message.rowId === machine.first)).toHaveLength(1)
  })

  it.each([
    ['a hidden heartbeat starter', 'hidden'],
    ['a process completion starter', 'process_complete']
  ])('draws a turn with %s once', async (_, kind) => {
    const turn = machineTurn(17, `machine-${kind}`, { kind })
    seed(history)
    await streamMachineTurn(turn, 'complete')

    const list = await refreshWith([...history.slice(3), ...turn.rows])

    expectDrawnOnce(list, turn)
    expect(list.filter(message => message.rowId === turn.first)).toHaveLength(kind === 'hidden' ? 0 : 1)
  })

  it('gives the reply its uid when the receipt already bound its row', async () => {
    seed(history)
    await streamMachineTurn(machine, 'complete')
    const finalRow = machine.rows.at(-1)!

    const reply = messages()
      .flatMap(message => message.parts)
      .find(part => part.type === 'text' && part.text === machine.final)

    expect([reply?.sourceRowId, reply?.sourceMessageUid]).toEqual([finalRow.id, finalRow.message_uid])
  })

  it('is idempotent across repeated refreshes', async () => {
    seed(history)
    await streamMachineTurn(machine, 'complete')

    const once = await refreshWith(page)
    const twice = await refreshWith([...page])

    expectDrawnOnce(twice, machine)
    expect(twice.map(message => message.id)).toEqual(once.map(message => message.id))
  })

  it('keeps the live answer a partial receipt does not cover, once each piece', async () => {
    // The receipt ends at the last tool result and names no final row: the answer is live-only.
    seed(history)
    await streamMachineTurn(machine, 'without-final')

    const list = reopen(page.slice(0, -1), messages())

    expectDrawnOnce(list, machine)
    expect(list.find(message => chatMessageText(message).includes(machine.final))?.pending).not.toBe(true)
  })

  it('does not take a later turn reaching past the receipt for the turn’s missing final row', async () => {
    // Every row but the final answer is stored, and a later turn's rows follow: min/max coverage only.
    const later = [user(100, 'a later prompt'), answer(101, 'a later answer')]
    seed(history)
    await streamMachineTurn(machine, 'complete')

    const list = await refreshWith([...page.slice(0, -1), ...later])

    expectDrawnOnce(list, machine)
    expect(occurrences(allText(list), 'a later answer')).toBe(1)
  })

  it('keeps the error card of a turn whose receipt never arrived', async () => {
    seed(history)
    await streamRounds(machine)
    send('message.complete', { status: 'error', error: 'provider exploded', text: '' })

    // Every round is stored; the failed answer never was.
    const list = await refreshWith(page.slice(0, -1))

    expect(paintedNarrations(list, machine)).toEqual(machine.narrations)
    expect(machine.tools.map(id => toolOccurrences(list, id))).toEqual(machine.tools.map(() => 1))
    expect(list.filter(message => message.error === 'provider exploded')).toHaveLength(1)
  })

  it.each(['complete', 'none'] as const)('keeps two legitimate turns with equal prose (receipt: %s)', async receipt => {
    // The same delegation result twice: equal prose, distinct rows, tool ids and turn ids.
    const earlier = machineTurn(17, 'first-run', { prose: 'same' })
    const later = machineTurn(17 + earlier.rows.length, 'second-run', { prose: 'same' })

    seed([...history, ...earlier.rows])
    await streamMachineTurn(later, receipt)

    const list = await refreshWith([...history.slice(3), ...earlier.rows, ...later.rows])
    const text = allText(list)

    for (const narrationText of later.narrations) {
      expect(occurrences(text, narrationText), narrationText).toBe(2)
    }

    expect(occurrences(text, later.final)).toBe(2)

    for (const id of [...earlier.tools, ...later.tools]) {
      expect(toolOccurrences(list, id), id).toBe(1)
    }
  })

  it('keeps an earlier live turn whose rows the page does not hold', async () => {
    // The previous backend-started turn lost its receipt and never stored its
    // rows; the next one completed normally. Only the second one is the page's.
    const lost = machineTurn(17, 'lost')
    const next = machineTurn(17 + lost.rows.length, 'next')

    seed(history)
    await streamMachineTurn(lost, 'none')
    await streamMachineTurn(next, 'complete')

    const list = await refreshWith([...history.slice(3), ...next.rows])

    expectDrawnOnce(list, next)
    expectDrawnOnce(list, lost)
  })

  it('leaves a user-started turn to its own prompt identity', async () => {
    // Control: the user's optimistic prompt anchors the receipt as before.
    const prompt = user(17, 'run the checks', { display_metadata: { submission_id: 'sub-user' } })
    const turn = { ...machineTurn(17, 'user-run'), starter: prompt }
    turn.rows = [prompt, ...turn.rows.slice(1)]
    turn.receipt = { ...turn.receipt, submission_id: 'sub-user' }

    seed(history)
    act(() =>
      cache.updateSessionState(RUNTIME, state => ({
        ...state,
        messages: [
          ...state.messages,
          { id: 'user-sub-user', role: 'user', parts: [{ type: 'text', text: 'run the checks' }] }
        ]
      }))
    )
    await streamMachineTurn(turn, 'complete')

    const list = await refreshWith([...history.slice(3), ...turn.rows])

    expectDrawnOnce(list, turn)
    expect(occurrences(allText(list), 'run the checks')).toBe(1)
  })
})

describe('a host that streams narration without sealing it', () => {
  // The session authority and a managed worker publish no message.interim for streamed narration,
  // and a managed worker sends no receipt: the tool events name the round's row, narration included.
  const unsealed = { owner: 'authority', seal: false } as const

  it('names each round’s streamed narration by the row its tool events name', async () => {
    seed(history)
    await streamMachineTurn(machine, 'none', unsealed)

    const named = messages()
      .flatMap(message => message.parts)
      .flatMap(part => (part.type === 'text' && machine.narrations.includes(part.text) ? [part] : []))

    const rows = machine.rows.filter(row => Array.isArray(row.tool_calls))

    expect(named.map(part => [part.sourceRowId, part.sourceMessageUid])).toEqual(
      rows.map(row => [row.id, row.message_uid])
    )
  })

  it('keeps a narration’s row when a completion with no new text names another row', async () => {
    seed(history)
    const emit = await streamRounds(machine, unsealed)
    const lastRound = machine.rows.filter(row => Array.isArray(row.tool_calls)).at(-1)!
    // No reply text streamed after the last round: the bubble's last text part is that round's narration.
    emit('message.complete', { status: 'complete', text: '', row_id: 999, message_uid: 'uid-other' })

    const narration = messages()
      .flatMap(message => message.parts)
      .find(part => part.type === 'text' && part.text === machine.narrations.at(-1))

    expect([narration?.sourceRowId, narration?.sourceMessageUid]).toEqual([lastRound.id, lastRound.message_uid])
  })

  it.each(['complete', 'none'] as const)('draws the turn once after a refresh (receipt: %s)', async receipt => {
    seed(history)
    await streamMachineTurn(machine, receipt, unsealed)

    const list = await refreshWith(page)

    expectDrawnOnce(list, machine)
    expect(list.filter(message => message.rowId === machine.first).map(message => message.role)).toEqual(['system'])
  })

  it.each([
    ['refresh', refreshed],
    ['reopen', reopen]
  ] as const)('draws a receipt-less turn once through %s, idempotently', async (_, compose) => {
    seed(history)
    await streamMachineTurn(machine, 'none', unsealed)
    const live = messages()

    const once = compose(page, live)
    const twice = compose(page, once)

    expectDrawnOnce(once, machine)
    expect(twice.map(message => message.id)).toEqual(once.map(message => message.id))
  })
})

describe('an older runtime that names no stream turn anywhere', () => {
  // No `turn_id` and no stored row named on any frame, and no `message_uid` on any stored row: the
  // completion receipt is the only identity, and the window has no prompt of a backend-started turn to anchor it.
  const legacy = (first = 17, tag = 'legacy', options: TurnOptions = {}) =>
    machineTurn(first, tag, { keyed: false, ...options })

  async function streamLegacy(turn: MachineTurn, receipt: Receipt = 'complete') {
    await streamMachineTurn(turn, receipt, { owner: 'legacy' })
    const live = messages()
    // The witness: nothing on screen names a stream turn.
    expect(live.filter(message => message.streamTurnId)).toEqual([])

    return live
  }

  const unstampedPage = (rows: SessionMessage[]) => {
    const list = toChatMessages(rows)
    expect(list.flatMap(message => message.parts).filter(part => part.sourceMessageUid)).toEqual([])

    return rows
  }

  it.each([
    ['an async delegation completion', 'async_delegation_complete', 1],
    ['a process completion', 'process_complete', 1],
    ['a hidden heartbeat', 'hidden', 0]
  ])('draws %s turn once by its receipt alone', async (_, kind, starters) => {
    const turn = legacy(17, `legacy-${kind}`, { kind })
    seed(history)
    await streamLegacy(turn)

    const list = await refreshWith(unstampedPage([...history.slice(3), ...turn.rows]))

    expectDrawnOnce(list, turn)
    expect(occurrences(allText(list), 'old-a look one')).toBe(1)
    expect(occurrences(allText(list), 'old-b answer')).toBe(1)
    expect(list.filter(message => message.rowId === turn.first)).toHaveLength(starters)
  })

  it.each([
    ['refresh', refreshed],
    ['reopen', reopen]
  ] as const)('draws it once through %s, idempotently', async (_, compose) => {
    const turn = legacy()
    seed(history)
    const live = await streamLegacy(turn)
    const rows = [...history.slice(3), ...turn.rows]

    const once = compose(rows, live)
    const twice = compose(rows, once)

    expectDrawnOnce(once, turn)
    expect(twice.map(message => message.id)).toEqual(once.map(message => message.id))
  })

  it('keeps a local notice drawn inside the turn, in its turn', async () => {
    const turn = legacy()
    seed(history)
    await streamLegacy(turn)

    const notice: ChatMessage = { id: 'btw-notice', role: 'system', parts: [{ type: 'text', text: 'a btw answer' }] }
    // Drawn while round 4 ran: after round 3's narration bubble.
    act(() =>
      cache.updateSessionState(RUNTIME, state => {
        const at = state.messages.findIndex(message => chatMessageText(message).includes(turn.narrations[3]))

        return { ...state, messages: [...state.messages.slice(0, at), notice, ...state.messages.slice(at)] }
      })
    )

    const list = await refreshWith([...history.slice(3), ...turn.rows])

    expectDrawnOnce(list, turn)
    const at = list.findIndex(message => message.id === 'btw-notice')
    expect(at).toBeGreaterThan(list.findIndex(message => message.rowId === turn.first))
    expect(at).toBeLessThan(list.findIndex(message => chatMessageText(message).includes(turn.final)))
  })

  it('keeps an earlier id-less live turn the page does not hold', async () => {
    // The earlier backend-started turn lost its receipt and stored nothing.
    const lost = legacy(17, 'legacy-lost')
    const next = legacy(17 + lost.rows.length, 'legacy-next')

    seed(history)
    await streamLegacy(lost, 'none')
    await streamLegacy(next)

    const list = await refreshWith([...history.slice(3), ...next.rows])

    expectDrawnOnce(list, next)
    expectDrawnOnce(list, lost)
  })

  it.each([
    ['refresh', refreshed],
    ['reopen', reopen]
  ] as const)('keeps the live-only pieces past a page cut inside the turn (%s)', async (_, compose) => {
    // A stale read: the page holds the turn's first rows but ends inside round 5, short of the receipt.
    const turn = legacy()
    seed(history)
    const live = await streamLegacy(turn)

    const list = compose([...history.slice(3), ...turn.rows.slice(0, 1 + 3 * 4 + 1)], live)

    expectNothingLost(list, turn)
    expect(occurrences(allText(list), turn.final)).toBe(1)
    expect(occurrences(allText(list), turn.narrations[5])).toBe(1)
    expect(occurrences(allText(list), 'old-b answer')).toBe(1)
  })

  it('does not take a later turn reaching past the missing final row for a copy', async () => {
    const turn = legacy()
    const later = [user(100, 'a later prompt'), answer(101, 'a later answer')]
    seed(history)
    await streamLegacy(turn)

    const list = await refreshWith([...history.slice(3), ...turn.rows.slice(0, -1), ...later])

    expectNothingLost(list, turn)
    expect(occurrences(allText(list), turn.final)).toBe(1)
    expect(occurrences(allText(list), 'a later answer')).toBe(1)
  })

  it('keeps the live answer a receipt without the final row does not cover', async () => {
    const turn = legacy()
    seed(history)
    const live = await streamLegacy(turn, 'without-final')

    const list = reopen([...history.slice(3), ...turn.rows.slice(0, -1)], live)

    expectNothingLost(list, turn)
    expect(occurrences(allText(list), turn.final)).toBe(1)
  })

  it('keeps two legitimate turns with equal prose', async () => {
    const earlier = legacy(17, 'legacy-first', { prose: 'same' })
    const later = legacy(17 + earlier.rows.length, 'legacy-second', { prose: 'same' })

    seed([...history, ...earlier.rows])
    await streamLegacy(later)

    const list = await refreshWith([...history.slice(3), ...earlier.rows, ...later.rows])
    const text = allText(list)

    for (const narrationText of later.narrations) {
      expect(occurrences(text, narrationText), narrationText).toBe(2)
    }

    expect(occurrences(text, later.final)).toBe(2)

    for (const id of [...earlier.tools, ...later.tools]) {
      expect(toolOccurrences(list, id), id).toBe(1)
    }
  })

  it('leaves a user-started turn to its own prompt identity', async () => {
    const prompt = user(17, 'run the checks', { display_metadata: { submission_id: 'sub-legacy' } })
    const turn = legacy(17, 'legacy-user')
    turn.rows = [prompt, ...turn.rows.slice(1)]
    turn.receipt = { ...turn.receipt, submission_id: 'sub-legacy' }

    seed(history)
    act(() =>
      cache.updateSessionState(RUNTIME, state => ({
        ...state,
        messages: [
          ...state.messages,
          { id: 'user-sub-legacy', role: 'user', parts: [{ type: 'text', text: 'run the checks' }] }
        ]
      }))
    )
    await streamLegacy(turn)

    const list = await refreshWith([...history.slice(3), ...turn.rows])

    expectDrawnOnce(list, turn)
    expect(occurrences(allText(list), 'run the checks')).toBe(1)
  })

  it('keeps an id-less user prompt its receipt does not name, and its reply', async () => {
    // A user turn whose prompt the receipt cannot prove: the turn boundary is not provable.
    const prompt = user(17, 'an unnamed prompt')
    const turn = legacy(17, 'legacy-unnamed')
    turn.rows = [prompt, ...turn.rows.slice(1)]
    turn.receipt = { ...turn.receipt, user_row_id: undefined, user_row_ids: [] }

    seed(history)
    act(() =>
      cache.updateSessionState(RUNTIME, state => ({
        ...state,
        messages: [
          ...state.messages,
          { id: 'user-local', role: 'user', parts: [{ type: 'text', text: 'an unnamed prompt' }] }
        ]
      }))
    )
    const live = await streamLegacy(turn)

    const list = refreshed([...history.slice(3), ...turn.rows], live)
    const promptAt = list.findLastIndex(message => chatMessageText(message) === 'an unnamed prompt')

    expectNothingLost(list, turn)
    // Whatever stays, no reply of this turn is drawn above its last prompt copy.
    expect(list.findIndex(message => chatMessageText(message).includes(turn.final))).toBeGreaterThan(promptAt)
  })
})

describe('one tool-call id the same turn reuses in two rounds', () => {
  // A provider can hand out `call_0` again in a later round of the same turn; each round's call
  // has its own stored row and its own result, and neither result may stand in for the other.
  const reusedTurn = () => {
    const turn = machineTurn(17, 'reuse')
    const reused = 'call_0'
    // Round 1's first call and round 2's first call share the id.
    const tools = turn.tools.map((id, index) => (index === 0 || index === 2 ? reused : id))

    const results = tools.map((_, index) =>
      index === 0 ? 'first round result' : index === 2 ? 'second round result' : 'ok'
    )

    let call = 0

    const rows = turn.rows.map(row => {
      if (row.role === 'assistant' && Array.isArray(row.tool_calls) && row.tool_calls.length) {
        return {
          ...row,
          tool_calls: row.tool_calls.map(entry => ({ ...entry, id: tools[call++] }))
        } as SessionMessage
      }

      if (row.role === 'tool') {
        const index = turn.rows.filter(entry => entry.role === 'tool').indexOf(row)

        return { ...row, tool_call_id: tools[index], content: results[index] } as SessionMessage
      }

      return row
    })

    return { turn: { ...turn, rows, tools, results }, reused }
  }

  // A fold renames the second `call_0` it holds (withUniqueToolCallIds); it is still that call.
  const reusedResults = (list: ChatMessage[], reused: string) =>
    list
      .flatMap(message => message.parts)
      .flatMap(part =>
        part.type === 'tool-call' && (part.providerToolCallId ?? part.toolCallId) === reused ? [part.result] : []
      )

  it('hydrates both rounds with their own result', () => {
    const { turn, reused } = reusedTurn()

    expect(reusedResults(toChatMessages(turn.rows), reused)).toEqual(['first round result', 'second round result'])
  })

  it.each(['complete', 'none'] as const)('keeps both outcomes after the merge (receipt: %s)', async receipt => {
    const { turn, reused } = reusedTurn()
    seed(history)
    await streamMachineTurn(turn, receipt)
    expect(reusedResults(messages(), reused)).toEqual(['first round result', 'second round result'])

    const list = await refreshWith([...history.slice(3), ...turn.rows])

    expect(reusedResults(list, reused)).toEqual(['first round result', 'second round result'])
    expect(paintedNarrations(list, turn)).toEqual(turn.narrations)
  })

  it.each([
    ['refresh', refreshed],
    ['reopen', reopen]
  ] as const)('keeps both outcomes when the page holds only the first round’s result (%s)', async (_, compose) => {
    const { turn, reused } = reusedTurn()
    seed(history)
    await streamMachineTurn(turn, 'running')
    const live = messages()

    // Rounds 1-2 are stored, but round 2's result row for the reused id is not yet.
    const secondResultRow = turn.rows.findIndex(row => row.role === 'tool' && row.content === 'second round result')
    const page = [...history.slice(3), ...turn.rows.slice(0, secondResultRow)]

    const once = compose(page, live)
    const twice = compose(page, once)

    for (const list of [once, twice]) {
      expect(reusedResults(list, reused)).toEqual(['first round result', 'second round result'])
      expect(paintedNarrations(list, turn)).toEqual(turn.narrations)
    }
  })
})

describe('a provider tool id another turn reuses', () => {
  // Provider tool-call ids are not unique across turns (lib/chat-messages/reconciliation.ts).
  it.each([
    ['an older runtime (no uid on the rows)', null],
    ['a current runtime', { turnId: 'another-turn' }]
  ] as const)('keeps a live result when a stored turn written by %s reuses its id', async (_, producer) => {
    const lost = machineTurn(17, 'lost-identity')
    const reused = lost.tools[0]
    seed(history)
    await streamMachineTurn(lost, 'none')

    // The page holds the other turn and nothing of the live one.
    const unrelated = [
      user(200, '[Background work finished] another turn', { display_kind: 'async_delegation_complete' }),
      narration(201, 'another turn narration', [reused], producer),
      tool(202, reused, producer, 'the unrelated result'),
      answer(203, 'another turn answer', producer)
    ]

    const list = await refreshWith([...history.slice(3), ...unrelated])
    const results = toolCards(list, reused).map(card => card.result)

    expect(results).toHaveLength(2)
    expect(results).toEqual(expect.arrayContaining(['ok', 'the unrelated result']))
    expectNothingLost(list, lost)
  })
})

describe('a page read while the backend-started turn still runs', () => {
  // Rounds 1-3 are stored; round 4's narration is still streaming.
  const committed = [...history.slice(3), ...machine.rows.slice(0, 1 + 3 * 3)]

  async function runningWindow() {
    seed(history)
    await streamRounds(machine, { rounds: 3 })
    send('message.delta', { text: machine.narrations[3], turn_id: machine.tag, segment: 3, offset: 0 })
    await flush()

    return messages()
  }

  it.each([
    ['refresh', refreshed],
    ['reopen', reopen]
  ] as const)('draws committed rounds once and keeps the streaming remainder (%s)', async (_, compose) => {
    const live = await runningWindow()
    const merged = compose(committed, live)
    const stored = machine.tools.slice(0, 6)

    expect(paintedNarrations(merged, machine)).toEqual(machine.narrations.slice(0, 4))
    expect(stored.map(id => toolOccurrences(merged, id))).toEqual(stored.map(() => 1))
    expect(merged.filter(message => message.pending).map(chatMessageText)).toEqual([machine.narrations[3]])
    // Idempotent: the next read of the same page changes nothing.
    expect(compose(committed, merged).map(chatMessageText)).toEqual(merged.map(chatMessageText))
  })

  it('keeps every live row when the page holds none of its rows yet', async () => {
    const live = await runningWindow()
    // A stale page that ends before the turn wrote anything.
    const merged = refreshed(history.slice(3), live)

    expect(paintedNarrations(merged, machine)).toEqual(machine.narrations.slice(0, 4))
    expect(machine.tools.slice(0, 6).map(id => toolOccurrences(merged, id))).toEqual([1, 1, 1, 1, 1, 1])
  })

  it('keeps the only copy of a streaming answer when the page ends at tools, pending', async () => {
    seed(history)
    await streamMachineTurn(machine, 'running')

    const list = reopen(page.slice(0, -1), messages())

    expectDrawnOnce(list, machine)
    expect(list.find(message => chatMessageText(message).includes(machine.final))?.pending).toBe(true)
  })

  it('keeps a distinct later occurrence of the same narration beyond the stored page', async () => {
    // Round 6 says what round 1 said, with its own tools; only rounds 1-5 are stored.
    const turn = machineTurn(17, 'repeat', {
      narrations: round => (round === 6 ? 'repeat narration 1' : `repeat narration ${round}`)
    })

    seed(history)
    await streamMachineTurn(turn, 'running')

    const list = reopen([...history.slice(3), ...turn.rows.slice(0, 1 + 3 * (ROUNDS - 1))], messages())

    expect(occurrences(allText(list), 'repeat narration 1')).toBe(2)
    expect(turn.tools.map(id => toolOccurrences(list, id))).toEqual(turn.tools.map(() => 1))
    expect(occurrences(allText(list), turn.final)).toBe(1)
  })

  it('shows a tool card once when its result arrived only over the stream', async () => {
    seed(history)
    await streamMachineTurn(machine, 'running')
    const live = messages()

    // The last call is stored; its result row is not.
    const durable = page.slice(0, -2)
    const once = reopen(durable, live)
    const twice = reopen(durable, once)

    for (const list of [once, twice]) {
      expectDrawnOnce(list, machine)
      expect(toolCards(list, machine.tools.at(-1)!).map(card => card.result)).toEqual(['ok'])
    }
  })

  it('preserves every piece of a turn an older runtime wrote without its stream turn', async () => {
    // No receipt, no producer key: which stored rows are this turn's is unknown, so nothing goes.
    const legacy = machineTurn(17, 'legacy', { keyed: false })
    seed(history)
    await streamMachineTurn(legacy, 'running')

    expectNothingLost(reopen([...history.slice(3), ...legacy.rows.slice(0, -1)], messages()), legacy)
  })
})

describe('the stream turn a live row is stamped with', () => {
  const stamps = () => messages().flatMap(message => (message.streamTurnId ? [message.streamTurnId] : []))

  it('is never a late frame of the previous turn', async () => {
    const previous = machineTurn(17, 'previous')
    const current = machineTurn(100, 'current')
    seed(history)
    await streamMachineTurn(previous, 'complete')

    // The authority starts the next execution; a late delta of the previous turn arrives first.
    const execution = { authority_epoch: 1, execution_generation: 5, admission_id: 'current' }
    send('message.start', {}, execution)
    send('message.delta', { text: ' late', turn_id: previous.tag, segment: ROUNDS, offset: 999 }, execution)
    await flush()
    expect(stamps().filter(turnId => turnId !== previous.tag)).toEqual([])

    send('message.delta', { text: current.narrations[0], turn_id: current.tag, segment: 0, offset: 0 }, execution)
    await flush()
    expect(messages().at(-1)?.streamTurnId).toBe(current.tag)
  })
})

describe('a compaction during the backend-started turn', () => {
  // A post-tool compaction archives the active rows and writes copies under new ids, keeping each
  // row's message_uid (persistence/conversation/messages.py `compact`). The live events of the rounds
  // before it named the old ids; the page names the new ones.
  const REISSUE = 1000

  const compacted = (rows: SessionMessage[], from: number) =>
    rows.map(row =>
      (row.id as number) >= from ? ({ ...row, id: (row.id as number) + REISSUE } as SessionMessage) : row
    )

  it.each([
    ['refresh', refreshed],
    ['reopen', reopen]
  ] as const)('draws each piece once by its uid when the row ids were re-issued (%s)', async (_, compose) => {
    seed(history)
    await streamMachineTurn(machine, 'none')
    const live = messages()

    const list = compose(compacted(page, machine.first), live)

    expectDrawnOnce(list, machine)
    expect(occurrences(allText(list), 'old-b answer')).toBe(1)
  })

  it('needs the uid: a row id alone no longer names the page row', async () => {
    seed(history)
    await streamMachineTurn(machine, 'none')

    // The same events with their uids stripped: the old row ids are all that is left.
    const live = messages().map(message => ({
      ...message,
      parts: message.parts.map(part => ({ ...part, sourceMessageUid: undefined }))
    }))

    const list = refreshed(compacted(page, machine.first), live)

    expect(occurrences(allText(list), machine.narrations[0])).toBe(2)
  })
})
