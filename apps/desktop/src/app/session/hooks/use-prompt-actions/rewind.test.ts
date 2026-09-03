import { describe, expect, it } from 'vitest'

import { type ChatMessage, textPart } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'

import {
  appendMidTurnUserMessage,
  applyReloadOptimistic,
  applyRewindOptimistic,
  finalizeInterruptedMessages,
  planEdit,
  planReload,
  planRestore,
  rebindSurvivorRowIds,
  resolveDurableRowId,
  runRewindSubmit,
  survivorRowIdsFrom,
  truncateSubmitParams
} from './rewind'

const row = (id: string, role: ChatMessage['role'], text: string, extra: Partial<ChatMessage> = {}): ChatMessage => ({
  id,
  role,
  parts: [textPart(text)],
  ...extra
})

type MidTurnState = { interimBoundaryPending: boolean; messages: ChatMessage[]; streamId: null | string }

describe('appendMidTurnUserMessage', () => {
  // #73793: a message typed while a turn streams must land AFTER the assistant
  // output the user had already watched arrive — never spliced above it.
  it('appends the mid-turn message after the live assistant output, sealed in place', () => {
    const state: MidTurnState = {
      interimBoundaryPending: false,
      streamId: 'assistant-stream-1',
      messages: [
        row('user-1', 'user', 'long task'),
        row('assistant-stream-1', 'assistant', 'two screens of output', { pending: true })
      ]
    }

    const next = appendMidTurnUserMessage(state, row('user-2', 'user', 'urgently'))

    expect(next.messages.map(message => message.id)).toEqual(['user-1', 'assistant-stream-1', 'user-2'])
    // The sealed bubble stops streaming; the turn's next delta seeds a fresh
    // bubble BELOW the correction instead of mutating the one above it.
    expect(next.messages[1]).toMatchObject({ pending: false, interim: true })
    expect(next.streamId).toBeNull()
    expect(next.interimBoundaryPending).toBe(true)
  })

  // #83151: the retired insert-before splice fell back to the LAST assistant
  // row anywhere in the transcript when the stream id was stale, landing the
  // new prompt mid-thread. A stale/missing stream id must append at the tail.
  it('appends at the live tail when the stream id is stale or missing', () => {
    const state: MidTurnState = {
      interimBoundaryPending: false,
      streamId: 'assistant-stream-stale',
      messages: [
        row('user-1', 'user', 'old prompt'),
        row('assistant-1', 'assistant', 'old committed reply'),
        row('user-2', 'user', 'newer prompt'),
        row('assistant-2', 'assistant', 'newer committed reply')
      ]
    }

    const next = appendMidTurnUserMessage(state, row('user-3', 'user', 'mid-turn note'))

    expect(next.messages.map(message => message.id)).toEqual([
      'user-1',
      'assistant-1',
      'user-2',
      'assistant-2',
      'user-3'
    ])
    expect(next.messages.at(-1)?.id).toBe('user-3')
    expect(next.interimBoundaryPending).toBe(false)
  })

  it('drops an empty pending stream placeholder instead of sealing it', () => {
    const state: MidTurnState = {
      interimBoundaryPending: false,
      streamId: 'assistant-stream-1',
      messages: [
        row('user-1', 'user', 'task'),
        { id: 'assistant-stream-1', role: 'assistant', parts: [], pending: true }
      ]
    }

    const next = appendMidTurnUserMessage(state, row('user-2', 'user', 'correction'))

    expect(next.messages.map(message => message.id)).toEqual(['user-1', 'user-2'])
    expect(next.streamId).toBeNull()
    expect(next.interimBoundaryPending).toBe(false)
  })
})

describe('truncateSubmitParams', () => {
  it('omits truncation fields when no ordinal is set', () => {
    expect(truncateSubmitParams(undefined)).toEqual({})
  })

  it('confirms intent for every ordinal, and the empty edge only for ordinal 0', () => {
    expect(truncateSubmitParams(0)).toEqual({
      confirm_truncate: true,
      truncate_before_user_ordinal: 0,
      confirm_empty_truncate: true
    })
    expect(truncateSubmitParams(1)).toEqual({
      confirm_truncate: true,
      truncate_before_user_ordinal: 1
    })
  })

  it('never sends an ordinal without the intent flag the gateway requires', () => {
    // The gateway drops history only for a submit that declares itself a
    // rewind. An ordinal built here without confirm_truncate would be refused
    // (and, on an older gateway, would truncate silently).
    for (const ordinal of [0, 1, 2, 7]) {
      const params = truncateSubmitParams(ordinal)

      expect(params.truncate_before_user_ordinal).toBe(ordinal)
      expect(params.confirm_truncate).toBe(true)
    }
  })

  it('includes truncate_before_row_id when passed', () => {
    expect(truncateSubmitParams(1, 'msg-123', 456)).toEqual({
      confirm_truncate: true,
      truncate_before_user_ordinal: 1,
      truncate_before_message_id: 'msg-123',
      truncate_before_row_id: 456
    })
    expect(truncateSubmitParams(undefined, undefined, 456)).toEqual({
      confirm_truncate: true,
      truncate_before_row_id: 456
    })
  })

  it('drops renderer-synthetic message ids but keeps durable row ids', () => {
    // chat-messages/hydration.ts: `${timestamp}-${index}-${role}`
    expect(truncateSubmitParams(1, '1723456789-0-user', 456)).toEqual({
      confirm_truncate: true,
      truncate_before_user_ordinal: 1,
      truncate_before_row_id: 456
    })
    expect(truncateSubmitParams(0, 'user-1723456789-0', undefined)).toEqual({
      confirm_truncate: true,
      truncate_before_user_ordinal: 0,
      confirm_empty_truncate: true
    })
  })
})

describe('survivorRowIdsFrom', () => {
  it('returns undefined when the field is absent or not an array (older gateway)', () => {
    expect(survivorRowIdsFrom(undefined)).toBeUndefined()
    expect(survivorRowIdsFrom({ status: 'streaming' })).toBeUndefined()
    expect(survivorRowIdsFrom({ survivor_user_row_ids: 'nope' })).toBeUndefined()
  })

  it('keeps integer ids and nulls anything else', () => {
    expect(survivorRowIdsFrom({ survivor_user_row_ids: [7, null, 9.5, '11', 12] })).toEqual([7, null, null, null, 12])
  })

  it('prefers the explicit old-to-new row id map', () => {
    const parsed = survivorRowIdsFrom({
      survivor_user_row_ids: [100, 200],
      survivor_row_id_map: { '11': 101, '21': 201, '31': null }
    })

    expect(parsed).toEqual(
      new Map([
        [11, 101],
        [21, 201],
        [31, null]
      ])
    )
  })
})

describe('rebindSurvivorRowIds', () => {
  const user = (id: string, rowId?: number, hidden?: boolean): ChatMessage => ({
    id,
    role: 'user',
    parts: [textPart(`text ${id}`)],
    ...(rowId !== undefined ? { rowId } : {}),
    ...(hidden ? { hidden } : {})
  })

  const assistant = (id: string, rowId?: number): ChatMessage => ({
    id,
    role: 'assistant',
    parts: [textPart(`reply ${id}`)],
    ...(rowId !== undefined ? { rowId } : {})
  })

  it('rebinds surviving visible user turns positionally and clears the resubmitted turn', () => {
    // Post-rewind state: two survivors + the resubmitted turn (stale rowId 5).
    const messages = [user('u0', 1), assistant('a0', 2), user('u1', 3), assistant('a1', 4), user('u2', 5)]
    const rebound = rebindSurvivorRowIds(messages, [7, 9])

    expect(rebound[0].rowId).toBe(7)
    expect(rebound[2].rowId).toBe(9)
    // Resubmitted turn is past the survivor list — its durable id doesn't
    // exist yet, and keeping the stale one would 4018 the next rewind.
    expect(rebound[4].rowId).toBeUndefined()
    // Assistant rows are untouched (only user turns are rewind targets).
    expect(rebound[1].rowId).toBe(2)
  })

  it('clears the cached id for null entries instead of keeping a stale one', () => {
    const messages = [user('u0', 1), user('u1', 3)]
    const rebound = rebindSurvivorRowIds(messages, [null, 9])

    expect(rebound[0].rowId).toBeUndefined()
    expect(rebound[1].rowId).toBe(9)
  })

  it('skips hidden user turns — same visible-user filter as the ordinal math', () => {
    const messages = [user('u0', 1), user('hidden', 2, true), user('u1', 3)]
    const rebound = rebindSurvivorRowIds(messages, [7, 9])

    expect(rebound[0].rowId).toBe(7)
    expect(rebound[1].rowId).toBe(2) // hidden: untouched
    expect(rebound[2].rowId).toBe(9)
  })

  it('preserves object identity when nothing changes', () => {
    const messages = [user('u0', 7)]

    expect(rebindSurvivorRowIds(messages, [7])[0]).toBe(messages[0])
  })

  it('rebinds a paged tail by previous row id instead of global position', () => {
    const messages = [
      user('archived-u250', 800),
      user('tail-u298', 901),
      assistant('tail-a298', 902),
      user('edited', 903)
    ]

    const rebound = rebindSurvivorRowIds(
      messages,
      new Map([
        [901, 1001],
        [902, 1002],
        [903, null]
      ])
    )

    expect(rebound[0]).toBe(messages[0])
    expect(rebound[0].rowId).toBe(800)
    expect(rebound[1].rowId).toBe(1001)
    expect(rebound[2].rowId).toBe(1002)
    expect(rebound[3].rowId).toBeUndefined()
  })
})

describe('finalizeInterruptedMessages', () => {
  it('records when a stopped assistant turn and its active part ended', () => {
    const [message] = finalizeInterruptedMessages(
      [
        {
          id: 'assistant-live',
          role: 'assistant',
          parts: [{ type: 'text', text: 'partial answer', timestamp: 10 }],
          pending: true,
          timestamp: 10
        }
      ],
      'assistant-live',
      11.25
    )

    expect(message.completedAt).toBe(11.25)
    expect(message.parts[0].completedAt).toBe(11.25)
    expect(message.pending).toBe(false)
  })

  it('keeps and closes a tool-only assistant turn when it is stopped', () => {
    const [message] = finalizeInterruptedMessages(
      [
        {
          id: 'assistant-tool',
          role: 'assistant',
          parts: [
            {
              args: {} as never,
              argsText: '{}',
              timestamp: 10,
              toolCallId: 'call-1',
              toolName: 'terminal',
              type: 'tool-call'
            }
          ],
          pending: true,
          timestamp: 10
        }
      ],
      'assistant-tool',
      11.25
    )

    expect(message.parts).toHaveLength(1)
    expect(message.parts[0].completedAt).toBe(11.25)
    expect(message.completedAt).toBe(11.25)
  })
})

describe('failed-turn-aware ordinal space', () => {
  const user = (id: string, rowId?: number): ChatMessage => ({
    id,
    role: 'user',
    parts: [textPart(`text ${id}`)],
    ...(rowId !== undefined ? { rowId } : {})
  })

  const assistant = (id: string, rowId?: number): ChatMessage => ({
    id,
    role: 'assistant',
    parts: [textPart(`reply ${id}`)],
    ...(rowId !== undefined ? { rowId } : {})
  })

  const failedAssistant = (id: string): ChatMessage => ({
    id,
    role: 'assistant',
    parts: [textPart(`err ${id}`)],
    error: 'provider down'
  })

  it('planEdit/planReload skip failed turns when counting ordinals (#41275)', () => {
    // u0 failed (never persisted), u1/u2 succeeded. Backend counts u1 before
    // u2, so editing u2 must aim at ordinal 1, not 2.
    const messages: ChatMessage[] = [
      user('u0', undefined),
      failedAssistant('a0'),
      user('u1', 11),
      assistant('a1', 12),
      user('u2', 13),
      assistant('a2', 14)
    ]

    const reload = planReload(messages, 'a2')

    expect(reload?.truncateOrdinal).toBe(1)
    expect(reload?.truncateRowId).toBe(13)
  })

  it('planEdit still flags failed turns via the shared helper', () => {
    const messages: ChatMessage[] = [user('u0', 11), assistant('a0', 12), user('u1', undefined), failedAssistant('a1')]

    const plan = planEdit(messages, {
      role: 'user',
      sourceId: 'u1',
      parentId: null,
      content: [{ type: 'text', text: 'better text' }]
    } as never)

    expect(plan?.isFailedTurn).toBe(true)
    expect(plan?.truncateOrdinal).toBeUndefined()
    expect(plan?.sourceText).toBe('text u1')
  })

  it('planReload degrades a failed turn to a plain resubmit (#86623)', () => {
    const messages: ChatMessage[] = [user('u0', 11), assistant('a0', 12), user('u1', undefined), failedAssistant('a1')]

    const reload = planReload(messages, 'a1')

    expect(reload?.text).toBe('text u1')
    expect(reload?.truncateOrdinal).toBeUndefined()
    expect(reload?.truncateRowId).toBeUndefined()
    expect(reload?.truncateMessageId).toBeUndefined()
  })

  it('planRestore degrades a failed turn to a plain resubmit', () => {
    const messages: ChatMessage[] = [user('u0', 11), assistant('a0', 12), user('u1', undefined), failedAssistant('a1')]

    const plan = planRestore(messages, 'u1')

    expect(plan.truncateOrdinal).toBeUndefined()
    expect(plan.truncateRowId).toBeUndefined()
    expect(plan.truncateMessageId).toBeUndefined()
  })

  it('rebindSurvivorRowIds skips failed turns — they hold no survivor slot', () => {
    const messages: ChatMessage[] = [user('u0', 1), failedAssistant('a0'), user('u1', 3), assistant('a1', 4)]
    const rebound = rebindSurvivorRowIds(messages, [9])

    // u0 failed: untouched. u1 is survivor ordinal 0.
    expect(rebound[0].rowId).toBe(1)
    expect(rebound[2].rowId).toBe(9)
  })
})

describe('resolveDurableRowId', () => {
  const gatewayWith = (messages: unknown[]) => {
    const request = (async (method: string) => {
      expect(method).toBe('session.history')

      return { messages }
    }) as <T>(method: string, params?: Record<string, unknown>, timeoutMs?: number) => Promise<T>

    return request
  }

  it('resolves a unique content match to its durable row id', async () => {
    const request = gatewayWith([
      { role: 'user', text: 'first prompt', row_id: 11 },
      { role: 'assistant', text: 'reply', row_id: 12 },
      { role: 'user', text: 'typo prompt', row_id: 13 }
    ])

    expect(await resolveDurableRowId(request, 'sid', 'typo prompt', 1)).toBe(13)
  })

  it('ignores synthetic user-role injections (display_kind rows)', async () => {
    const request = gatewayWith([
      { role: 'user', text: 'real prompt', row_id: 11 },
      { role: 'user', text: 'real prompt', row_id: 12, display_kind: 'auto_continue' }
    ])

    expect(await resolveDurableRowId(request, 'sid', 'real prompt', 0)).toBe(11)
  })

  it('refuses ambiguous matches unless the target is provably the newest turn', async () => {
    const request = gatewayWith([
      { role: 'user', text: 'same text', row_id: 11 },
      { role: 'user', text: 'same text', row_id: 13 }
    ])

    // Ambiguous + target not the latest -> undefined (plain resubmit).
    expect(await resolveDurableRowId(request, 'sid', 'same text', 0)).toBeUndefined()
    // Ambiguous + target IS the latest persisted turn -> last match wins.
    expect(await resolveDurableRowId(request, 'sid', 'same text', 1)).toBe(13)
  })

  it('returns undefined on gateway failure or empty text', async () => {
    const failing = (async () => {
      throw new Error('boom')
    }) as <T>(method: string) => Promise<T>

    expect(await resolveDurableRowId(failing, 'sid', 'text', 0)).toBeUndefined()
    expect(await resolveDurableRowId(gatewayWith([]), 'sid', '   ', 0)).toBeUndefined()
  })
})

describe('runRewindSubmit durable-address discipline (#87059)', () => {
  interface Call {
    method: string
    params?: Record<string, unknown>
  }

  const historyMessages = [
    { role: 'user', text: 'first prompt', row_id: 11 },
    { role: 'assistant', text: 'ok', row_id: 12 },
    { role: 'user', text: 'typo prompt', row_id: 13 }
  ]

  const makeGateway = (calls: Call[]) =>
    (async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'session.history') {
        return { messages: historyMessages }
      }

      return { status: 'streaming' }
    }) as <T>(method: string, params?: Record<string, unknown>, timeoutMs?: number) => Promise<T>

  it('resolves a missing rowId by content before submitting, and drops the client ordinal', async () => {
    const calls: Call[] = []

    await runRewindSubmit(
      makeGateway(calls),
      'sid',
      'fixed prompt',
      5,
      undefined,
      false,
      undefined,
      undefined,
      'typo prompt'
    )

    const submit = calls.find(call => call.method === 'prompt.submit')

    expect(submit?.params?.truncate_before_row_id).toBe(13)
    expect(submit?.params?.truncate_before_user_ordinal).toBeUndefined()
    expect(submit?.params?.confirm_truncate).toBe(true)
  })

  it('degrades to a plain resubmit when the row id cannot be resolved', async () => {
    const calls: Call[] = []

    await runRewindSubmit(
      makeGateway(calls),
      'sid',
      'fixed prompt',
      5,
      undefined,
      false,
      undefined,
      undefined,
      'unknown text'
    )

    const submit = calls.find(call => call.method === 'prompt.submit')

    expect(submit?.params?.truncate_before_row_id).toBeUndefined()
    expect(submit?.params?.truncate_before_user_ordinal).toBeUndefined()
    expect(submit?.params?.confirm_truncate).toBeUndefined()
  })

  it('leaves a bound durable rowId untouched (no extra history call) and drops the client ordinal', async () => {
    const calls: Call[] = []

    await runRewindSubmit(
      makeGateway(calls),
      'sid',
      'fixed prompt',
      1,
      undefined,
      false,
      undefined,
      13,
      'typo prompt',
      [11, 12, 13]
    )

    expect(calls.some(call => call.method === 'session.history')).toBe(false)

    const submit = calls.find(call => call.method === 'prompt.submit')

    expect(submit?.params?.truncate_before_row_id).toBe(13)
    expect(submit?.params?.truncate_before_user_ordinal).toBeUndefined()
    expect(submit?.params?.confirm_empty_truncate).toBe(true)
    expect(submit?.params?.rebind_survivor_row_ids).toEqual([11, 12, 13])
  })

  it('drops the client ordinal whenever a durable row id is present, including a complete live transcript (#88082, #89244)', async () => {
    const calls: Call[] = []

    await runRewindSubmit(
      makeGateway(calls),
      'sid',
      'fixed prompt',
      1, // display-lineage / window-relative — not the gateway tip
      undefined,
      false,
      undefined,
      13,
      'typo prompt'
    )

    // No content-resolution read needed — the bubble already holds a durable id.
    expect(calls.some(call => call.method === 'session.history')).toBe(false)

    const submit = calls.find(call => call.method === 'prompt.submit')

    expect(submit?.params?.confirm_truncate).toBe(true)
    expect(submit?.params?.truncate_before_row_id).toBe(13)
    expect(submit?.params?.truncate_before_user_ordinal).toBeUndefined()
  })

  it('drops the client ordinal when the cut is addressed by durable message id alone', async () => {
    const calls: Call[] = []

    await runRewindSubmit(
      makeGateway(calls),
      'sid',
      'fixed prompt',
      1,
      'durable-platform-msg-1',
      false,
      undefined,
      undefined,
      'typo prompt'
    )

    const submit = calls.find(call => call.method === 'prompt.submit')

    expect(submit?.params?.confirm_truncate).toBe(true)
    expect(submit?.params?.truncate_before_message_id).toBe('durable-platform-msg-1')
    expect(submit?.params?.truncate_before_user_ordinal).toBeUndefined()
  })

  it('still confirms an empty truncate when the dropped ordinal was 0', async () => {
    const calls: Call[] = []

    await runRewindSubmit(
      makeGateway(calls),
      'sid',
      'fixed prompt',
      0,
      'durable-platform-msg-1',
      false,
      undefined,
      13,
      'typo prompt'
    )

    const submit = calls.find(call => call.method === 'prompt.submit')

    expect(submit?.params?.confirm_truncate).toBe(true)
    expect(submit?.params?.confirm_empty_truncate).toBe(true)
    expect(submit?.params?.truncate_before_row_id).toBe(13)
    expect(submit?.params?.truncate_before_user_ordinal).toBeUndefined()
  })
})

describe('optimistic rewind/reload turn-clock seeding (#86795)', () => {
  // The no-payload settle gate in gateway-event/session-info.ts holds a running=false
  // heartbeat off while an optimistically armed turn waits for the backend —
  // but only for a bounded grace window measured from turnStartedAt. These
  // transforms are the arm sites for restore/edit/regenerate, so they must
  // seed the clock (and reset turnLive): an armed state without a clock is
  // settled by the first heartbeat, and a STALE clock from a previous turn
  // would let a heartbeat settle the fresh arm immediately.
  const seeded = (extra: Partial<ReturnType<typeof createClientSessionState>>) => ({
    ...createClientSessionState(null, [row('u1', 'user', 'first'), row('a1', 'assistant', 'answer')]),
    ...extra
  })

  it('applyRewindOptimistic arms busy with a fresh turn clock', () => {
    const before = Date.now()
    const next = applyRewindOptimistic(seeded({ turnLive: true, turnStartedAt: before - 60_000 }), 0)

    expect(next.busy).toBe(true)
    expect(next.awaitingResponse).toBe(true)
    expect(next.turnLive).toBe(false)
    expect(next.turnStartedAt).toBeGreaterThanOrEqual(before)
  })

  it('applyRewindOptimistic invalidates persisted-display authority', () => {
    const next = applyRewindOptimistic(
      seeded({
        transcriptAuthorityEpoch: 3,
        transcriptProvenance: {
          connectionId: 'conn-1',
          coverage: 'latest-page',
          displayRevision: 7,
          lineageRootId: 'root-1',
          profile: 'default',
          resolvedTipId: 'tip-1',
          source: 'persisted-display',
          storedSessionId: 'stored-1'
        }
      }),
      0
    )

    expect(next.transcriptAuthorityEpoch).toBe(4)
    expect(next.transcriptProvenance).toBeUndefined()
  })

  it('applyReloadOptimistic arms busy with a fresh turn clock', () => {
    const state = seeded({ turnLive: true, turnStartedAt: Date.now() - 60_000 })
    const plan = planReload(state.messages, null)
    const before = Date.now()

    expect(plan).not.toBeNull()

    const next = applyReloadOptimistic(state, plan!)

    expect(next.busy).toBe(true)
    expect(next.turnLive).toBe(false)
    expect(next.turnStartedAt).toBeGreaterThanOrEqual(before)
  })

  it('applyReloadOptimistic invalidates persisted-display authority', () => {
    const state = seeded({
      transcriptAuthorityEpoch: 3,
      transcriptProvenance: {
        connectionId: 'conn-1',
        coverage: 'latest-page',
        displayRevision: 7,
        lineageRootId: 'root-1',
        profile: 'default',
        resolvedTipId: 'tip-1',
        source: 'persisted-display',
        storedSessionId: 'stored-1'
      }
    })

    const plan = planReload(state.messages, null)

    expect(plan).not.toBeNull()

    const next = applyReloadOptimistic(state, plan!)

    expect(next.transcriptAuthorityEpoch).toBe(4)
    expect(next.transcriptProvenance).toBeUndefined()
  })
})
