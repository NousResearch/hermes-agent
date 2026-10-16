import { describe, expect, it } from 'vitest'

import { type ChatMessage, chatMessageText, toChatMessages } from '@/lib/chat-messages'
import type { SessionResumeResult } from '@/types/hermes'

import {
  appendLiveSessionProjection,
  dedupeInflightUserAgainstTranscript,
  removeRepresentedLocalLiveProjection
} from './live-session-projection'

const msg = (id: string, role: ChatMessage['role'], text: string, extra: Partial<ChatMessage> = {}): ChatMessage =>
  ({ id, role, parts: [{ type: 'text', text }], ...extra }) as ChatMessage

const runningProjection = (user: string): SessionResumeResult =>
  ({
    session_id: 'runtime-1',
    session_key: 'stored-1',
    resumed: 'stored-1',
    message_count: 2,
    messages: [],
    running: true,
    inflight: { user, assistant: 'partial answer', streaming: true }
  }) as SessionResumeResult

describe('appendLiveSessionProjection', () => {
  it('keeps runtime provenance and metadata when reconciling a persisted synthetic notice', () => {
    const inflight = {
      user: '[IMPORTANT: Background process finished] fixture',
      user_originated: false,
      display_kind: 'process_complete' as const,
      display_metadata: { display_text: 'Finished syncing the workspace' },
      assistant: 'The workspace is ready.',
      streaming: true
    }

    const projection = { session_id: 'runtime-1', turn_started_at: 20, inflight }
    const live = appendLiveSessionProjection([], projection)

    expect(live.map(message => [message.role, chatMessageText(message)])).toEqual([
      ['system', inflight.display_metadata.display_text],
      ['assistant', inflight.assistant]
    ])
    expect(live.every(message => message.runtimeTurnStartedAt === projection.turn_started_at)).toBe(true)

    const persisted = toChatMessages([{ role: 'user', content: inflight.user, timestamp: 21, ...inflight }])
    const hydrated = appendLiveSessionProjection(persisted, projection)

    expect(hydrated.map(message => [message.role, chatMessageText(message)])).toEqual(
      live.map(message => [message.role, chatMessageText(message)])
    )
    expect(hydrated[0].id).toBe(persisted[0].id)
    expect(hydrated[0].runtimeTurnStartedAt).toBe(projection.turn_started_at)
  })

  it.each([
    ['background-process notice', '[IMPORTANT: Background process proc_123 completed normally with exit code 0.]'],
    ['compaction task snapshot', '[Your active task list was preserved across context compression]\n- [>] current task']
  ])('keeps an attached inflight prompt anchored before a later %s', (_kind, syntheticNotice) => {
    const stored = [
      msg('stored-user', 'user', 'current running prompt', {
        attachmentRefs: ['@image:/tmp/screenshot.png'],
        timestamp: 11
      }),
      msg('stored-assistant', 'assistant', 'partial tool activity', { timestamp: 12 }),
      msg('synthetic-notice', 'user', syntheticNotice, { timestamp: 13 })
    ]

    const restored = appendLiveSessionProjection(stored, {
      session_id: 'runtime-1',
      turn_started_at: 10,
      inflight: {
        user: 'current running prompt',
        assistant: 'partial answer',
        streaming: true
      }
    })

    const promptRows = restored.filter(message => chatMessageText(message) === 'current running prompt')

    expect(promptRows).toHaveLength(1)
    expect(promptRows[0]).toMatchObject({
      id: 'stored-user',
      attachmentRefs: ['@image:/tmp/screenshot.png']
    })
    expect(restored.map(message => message.id)).toEqual([
      'stored-user',
      'stored-assistant',
      'synthetic-notice',
      'assistant-stream-runtime-1'
    ])
  })

  it('keeps a newly accepted repeated prompt after a completed turn and trailing synthetic notice', () => {
    const stored = [
      msg('stored-user', 'user', 'repeat this', { timestamp: 1 }),
      msg('stored-assistant', 'assistant', 'finished answer', { timestamp: 2 }),
      msg('synthetic-notice', 'user', '[IMPORTANT: Background process proc_123 completed normally with exit code 0.]', {
        timestamp: 3
      })
    ]

    const restored = appendLiveSessionProjection(stored, {
      session_id: 'runtime-1',
      turn_started_at: 4,
      inflight: {
        user: 'repeat this',
        assistant: 'new partial answer',
        streaming: true
      }
    })

    expect(restored.filter(message => chatMessageText(message) === 'repeat this')).toHaveLength(2)
    expect(restored.map(message => message.id)).toEqual([
      'stored-user',
      'stored-assistant',
      'synthetic-notice',
      'user-inflight-runtime-1',
      'assistant-stream-runtime-1'
    ])
  })
})

describe('dedupeInflightUserAgainstTranscript', () => {
  it('uses the local committed prefix when an older gateway omits runtime history and turn timing', () => {
    const committed = [
      msg('committed-user', 'user', 'earlier prompt', { timestamp: 1 }),
      msg('committed-assistant', 'assistant', 'earlier answer', { timestamp: 2 })
    ]

    const local = [
      ...committed,
      msg('user-optimistic', 'user', 'current prompt'),
      msg('assistant-stream-runtime-1', 'assistant', 'partial', { pending: true })
    ]

    const persisted = [
      ...committed,
      msg('persisted-current', 'user', 'current prompt', { timestamp: 3 }),
      msg(
        'persisted-background-notice',
        'user',
        '[IMPORTANT: Background process proc_123 completed normally with exit code 0.]',
        { timestamp: 4 }
      )
    ]

    const projection = runningProjection('current prompt')
    const deduped = dedupeInflightUserAgainstTranscript(persisted, [], projection, local)
    const restored = appendLiveSessionProjection(persisted, deduped)

    expect(restored.filter(message => chatMessageText(message) === 'current prompt')).toHaveLength(1)
    expect(restored.map(message => message.id)).not.toContain('user-inflight-runtime-1')
  })

  it('keeps a newly accepted repeated prompt when the local committed prefix ends after its historical twin', () => {
    const committed = [
      msg('committed-user', 'user', 'repeat this', { timestamp: 1 }),
      msg('committed-assistant', 'assistant', 'finished answer', { timestamp: 2 })
    ]

    const local = [
      ...committed,
      msg('user-optimistic', 'user', 'repeat this'),
      msg('assistant-stream-runtime-1', 'assistant', 'new partial answer', { pending: true })
    ]

    const persisted = [
      ...committed,
      msg(
        'persisted-background-notice',
        'user',
        '[IMPORTANT: Background process proc_123 completed normally with exit code 0.]',
        { timestamp: 3 }
      )
    ]

    const projection = {
      ...runningProjection('repeat this'),
      inflight: { user: 'repeat this', assistant: 'new partial answer', streaming: true }
    }

    const deduped = dedupeInflightUserAgainstTranscript(persisted, [], projection, local)
    const restored = appendLiveSessionProjection(persisted, deduped)

    expect(restored.filter(message => chatMessageText(message) === 'repeat this')).toHaveLength(2)
    expect(restored.map(message => message.id)).toContain('user-inflight-runtime-1')
  })

  it('uses a durable renderer-owned user row as the old-gateway prefix anchor', () => {
    const committed = [msg('user-previous', 'user', 'repeat this', { rowId: 41, timestamp: 1 })]

    const local = [
      ...committed,
      msg('user-optimistic', 'user', 'repeat this'),
      msg('assistant-stream-runtime-1', 'assistant', 'new partial answer', { pending: true })
    ]

    const persisted = [
      msg('persisted-previous', 'user', 'repeat this', { rowId: 41, timestamp: 1 }),
      msg(
        'persisted-background-notice',
        'user',
        '[IMPORTANT: Background process proc_123 completed normally with exit code 0.]',
        { rowId: 42, timestamp: 2 }
      )
    ]

    const projection = {
      ...runningProjection('repeat this'),
      inflight: { user: 'repeat this', assistant: 'new partial answer', streaming: true }
    }

    const deduped = dedupeInflightUserAgainstTranscript(persisted, [], projection, local)
    const restored = appendLiveSessionProjection(persisted, deduped)

    expect(restored.filter(message => chatMessageText(message) === 'repeat this')).toHaveLength(2)
    expect(restored.map(message => message.id)).toContain('user-inflight-runtime-1')
  })

  it('preserves a repeated in-flight prompt when the persisted match already has an answer', () => {
    const runtime = [
      msg('runtime-user', 'user', 'earlier prompt', { timestamp: 1 }),
      msg('runtime-assistant', 'assistant', 'earlier answer', { timestamp: 2 })
    ]

    const persisted = [
      ...runtime,
      msg('persisted-repeat', 'user', 'repeat this', { timestamp: 3 }),
      msg('persisted-repeat-answer', 'assistant', 'finished repeat answer', { timestamp: 4 })
    ]

    const local = [
      ...runtime,
      msg('user-optimistic', 'user', 'repeat this'),
      msg('assistant-stream-runtime-1', 'assistant', 'partial answer', { pending: true })
    ]

    const projection = runningProjection('repeat this')
    const deduped = dedupeInflightUserAgainstTranscript(persisted, runtime, projection, local)
    const restored = appendLiveSessionProjection(persisted, deduped)

    expect(deduped.inflight?.user).toBe('repeat this')
    expect(restored.filter(message => chatMessageText(message) === 'repeat this')).toHaveLength(2)
    expect(restored.map(message => message.id)).toContain('user-inflight-runtime-1')
  })
})

describe('removeRepresentedLocalLiveProjection', () => {
  it('matches a cached stream when the activation projection has advanced', () => {
    const previous = [
      msg('user-current', 'user', 'current prompt'),
      msg('assistant-stream-current', 'assistant', 'partial', { pending: true })
    ]

    expect(removeRepresentedLocalLiveProjection(previous, runningProjection('current prompt'))).toEqual([])
  })
})
