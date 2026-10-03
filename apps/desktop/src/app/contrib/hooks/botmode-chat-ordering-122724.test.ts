/**
 * RED-first repro for #122724: Bot Mode chat switching appends older sent
 * messages below newer in-progress activity.
 *
 * Scenario: Bot A is still working (live assistant stream + tool activity).
 * The user switches to Bot B and back. The switch-back runs
 * `resumeTile(storedA, { refreshTranscript: true })`, which REST-merges the
 * persisted page (now containing tool activity that landed while away) onto
 * the warm cached transcript. The already-sent user prompt must stay ABOVE
 * the in-progress assistant work it caused — never re-appended below it.
 */
import { renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { appendLiveSessionProjection } from '@/app/session/hooks/use-session-actions/utils'
import type * as HermesModule from '@/hermes'
import type { ChatMessage } from '@/lib/chat-messages'
import { chatMessageText } from '@/lib/chat-messages'
import { $sessionTiles, sessionTileDelegate } from '@/store/session-states'

import { useSessionTileDelegate } from './use-session-tile-delegate'

vi.mock('@/hermes', async importActual => ({
  ...(await importActual<typeof HermesModule>()),
  getLatestSessionMessages: vi.fn(async () => ({ messages: [], session_id: '' }))
}))
vi.mock('@/store/gateway', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn(),
  requestGatewayForProfile: vi.fn()
}))

const { getLatestSessionMessages } = await import('@/hermes')

function renderTile(
  requestGateway: ReturnType<typeof vi.fn>,
  refs: {
    runtimeIdByStoredSessionIdRef: { current: Map<string, string> }
    sessionStateByRuntimeIdRef: { current: Map<string, unknown> }
    updateSessionState: ReturnType<typeof vi.fn>
  }
) {
  renderHook(() =>
    useSessionTileDelegate({
      archiveSession: vi.fn(async () => undefined),
      branchStoredSession: vi.fn(async () => undefined),
      executeSlashCommand: vi.fn(async () => undefined) as never,
      removeSession: vi.fn(async () => undefined),
      requestGateway: requestGateway as never,
      runtimeIdByStoredSessionIdRef: refs.runtimeIdByStoredSessionIdRef as never,
      sessionStateByRuntimeIdRef: refs.sessionStateByRuntimeIdRef as never,
      updateSessionState: refs.updateSessionState as never
    })
  )
}

const labels = (messages: Array<{ id: string; role: string }>) =>
  messages.map(message => `${message.role}:${message.id}`)

const textMsg = (id: string, role: 'user' | 'assistant', text: string, rowId?: number) => ({
  id,
  role,
  ...(rowId !== undefined ? { rowId } : {}),
  parts: [{ type: 'text', text }]
})

describe('live projection keeps the turn prompt above mid-turn follow-ups (#122724)', () => {
  it('does not duplicate the already-persisted prompt below newer activity', () => {
    // Switch-back resume while Bot A still works on a steered turn. The
    // persisted page already holds the original prompt, the sealed
    // pre-correction output, AND the persisted follow-up; the activate
    // snapshot still carries the whole turn as inflight.
    const persisted = [
      textMsg('u1', 'user', 'run the migration', 1),
      textMsg('a2', 'assistant', 'checked part one', 2),
      textMsg('u2', 'user', 'also check the tests', 3)
    ] as unknown as ChatMessage[]

    const projected = appendLiveSessionProjection(persisted, {
      session_id: 's',
      inflight: {
        user: 'run the migration',
        assistant: 'checked part one\nchecked part two',
        streaming: true,
        corrections: ['also check the tests']
      }
    } as never)

    // The prompt sent BEFORE the work must not be re-appended below it.
    const promptPositions = projected
      .map((message, index) => ({ message, index }))
      .filter(({ message }) => message.role === 'user' && chatMessageText(message).includes('run the migration'))
      .map(({ index }) => index)

    expect(promptPositions).toHaveLength(1)
    expect(promptPositions[0]).toBe(0)
  })

  it('anchors on the newest matching follow-up, not an identical one from an earlier turn', () => {
    // The same follow-up text was already sent in turn 1, so a forward search
    // for the correction matches that older row first, walks back to turn 1's
    // opener, and decides the current prompt is unpersisted — re-appending it
    // below its own work. Only the newest matching row belongs to this turn.
    const persisted = [
      textMsg('u0', 'user', 'check the tests', 1),
      textMsg('a1', 'assistant', 'ok', 2),
      textMsg('u1', 'user', 'also check the tests', 3),
      textMsg('a2', 'assistant', 'done', 4),
      textMsg('u2', 'user', 'run the migration', 5),
      textMsg('a3', 'assistant', 'checked part one', 6),
      textMsg('u3', 'user', 'also check the tests', 7)
    ] as unknown as ChatMessage[]

    const projected = appendLiveSessionProjection(persisted, {
      session_id: 's',
      inflight: {
        user: 'run the migration',
        assistant: 'checked part one\nchecked part two',
        streaming: true,
        corrections: ['also check the tests']
      }
    } as never)

    const promptPositions = projected
      .map((message, index) => ({ message, index }))
      .filter(({ message }) => message.role === 'user' && chatMessageText(message).includes('run the migration'))
      .map(({ index }) => index)

    expect(promptPositions).toHaveLength(1)
    expect(promptPositions[0]).toBe(4)
  })
})

describe('bot-mode chat switching keeps chronological order (#122724)', () => {
  it('keeps the already-sent prompt above in-progress tool activity on switch-back', async () => {
    // Warm cache for Bot A as left when switching to Bot B: the prompt was
    // already sent (committed rowId 1) and the bot is still working — a live
    // assistant stream carrying the tool call issued so far.
    const cached = {
      busy: true,
      streamId: 'assistant-stream-live',
      storedSessionId: 'stored-botA',
      messages: [
        { id: 'u1', role: 'user', rowId: 1, parts: [{ type: 'text', text: 'run the migration' }] },
        {
          id: 'assistant-stream-live',
          role: 'assistant',
          pending: true,
          parts: [
            { type: 'tool-call', toolCallId: 'tc-1', toolName: 'terminal' },
            { type: 'text', text: 'running migration' }
          ]
        }
      ]
    }

    const states = { current: new Map<string, unknown>([['runtime-botA', cached]]) }

    const update = vi.fn((_id: string, updater: (state: unknown) => unknown) => {
      const next = updater(states.current.get(_id))
      states.current.set(_id, next)

      return next
    })

    renderTile(vi.fn(), {
      runtimeIdByStoredSessionIdRef: { current: new Map([['stored-botA', 'runtime-botA']]) },
      sessionStateByRuntimeIdRef: states,
      updateSessionState: update
    })

    // While away, the bot kept working: the persisted page now carries the
    // tool result plus follow-on activity under newer row ids.
    vi.mocked(getLatestSessionMessages).mockResolvedValueOnce({
      session_id: 'stored-botA',
      messages: [
        { role: 'user', content: 'run the migration', row_id: 1, timestamp: 1 },
        {
          role: 'assistant',
          content: 'running migration and more',
          row_id: 2,
          timestamp: 2,
          tool_calls: [{ id: 'tc-1', function: { name: 'terminal', arguments: '{}' } }]
        },
        { role: 'tool', tool_call_id: 'tc-1', tool_name: 'terminal', content: 'ok', row_id: 3, timestamp: 3 },
        { role: 'assistant', content: 'continuing with the next step', row_id: 4, timestamp: 4 }
      ]
    } as never)

    try {
      const runtimeId = await sessionTileDelegate()!.resumeTile('stored-botA', { refreshTranscript: true })
      expect(runtimeId).toBe('runtime-botA')

      const refreshed = states.current.get('runtime-botA')! as {
        messages: Array<{ id: string; role: string; parts: Array<{ text?: string }> }>
      }

      const order = labels(refreshed.messages)
      const texts = refreshed.messages.map(message => chatMessageText(message as never))

      // The prompt sent BEFORE the work started must render above it
      // (matched by text: hydration may re-id the row).
      const promptIndex = texts.findIndex(
        (text, index) => refreshed.messages[index].role === 'user' && text.includes('run the migration')
      )

      const workIndex = texts.findIndex(text => text.includes('continuing with the next step'))

      expect(promptIndex).toBeGreaterThanOrEqual(0)
      expect(workIndex).toBeGreaterThanOrEqual(0)
      expect(promptIndex).toBeLessThan(workIndex)

      // And exactly once — no duplicated prompt re-appended at the tail.
      expect(
        texts.filter(
          (text, index) => refreshed.messages[index].role === 'user' && text.includes('run the migration')
        )
      ).toHaveLength(1)

      // No trailing stale copy of the live stream below the persisted work.
      const streamCopies = order.filter(entry => entry === 'assistant:assistant-stream-live')
      expect(streamCopies.length).toBeLessThanOrEqual(1)

      if (streamCopies.length === 1) {
        expect(order.indexOf('assistant:assistant-stream-live')).toBeLessThanOrEqual(workIndex)
      }
    } finally {
      $sessionTiles.set([])
    }
  })
})
