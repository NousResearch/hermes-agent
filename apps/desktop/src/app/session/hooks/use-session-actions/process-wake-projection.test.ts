import { expect, it } from 'vitest'

import { type ChatMessage, chatMessageText } from '@/lib/chat-messages'

import { appendLiveSessionProjection } from './utils'

const row = (id: string, role: ChatMessage['role'], text: string, timestamp: number): ChatMessage => ({
  id,
  role,
  parts: [{ type: 'text', text }],
  timestamp
})

it('does not replay a completed process wake from a stale activation snapshot (#131238)', () => {
  const persisted: ChatMessage[] = [
    row('old-user', 'user', 'Start the job.', 10),
    row('old-reply', 'assistant', 'Job running.', 20),
    row('wake-notice', 'system', 'Background Process Finished: fixture', 101),
    row('wake-reply', 'assistant', 'The build passed with 0 warnings.', 102)
  ]

  const inflight = {
    user: '[IMPORTANT: Background process finished] fixture',
    display_kind: 'process_complete' as const,
    display_metadata: { display_text: 'Background Process Finished: fixture' },
    assistant: 'The build passed',
    streaming: true
  }

  // REST completed the wake while session.activate still carried its old
  // streaming snapshot. The stored marker and answer must remain the only pair.
  const settled = appendLiveSessionProjection(persisted, {
    session_id: 'runtime-1',
    turn_started_at: 100,
    inflight
  })

  expect(settled).toBe(persisted)
  expect(
    settled.filter(
      message => message.role === 'assistant' && chatMessageText(message).startsWith('The build passed')
    )
  ).toHaveLength(1)

  // Identical notices from a later, genuinely running wake stay distinct.
  const nextWake = appendLiveSessionProjection(persisted, {
    session_id: 'runtime-1',
    turn_started_at: 200,
    inflight
  })
  expect(nextWake.filter(message => message.id === 'user-inflight-runtime-1')).toHaveLength(1)
  expect(nextWake.at(-1)).toMatchObject({ id: 'assistant-stream-runtime-1', pending: true })
})
