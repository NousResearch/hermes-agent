import { describe, expect, it } from 'vitest'

import { toChatMessages } from './hydration'

// A compaction carrier whose tail was a producer notice: the sessions route keeps the whole carrier in
// `content` and serves the notice alone in `display_content`. The expanded result must come from the
// notice, never the model-only summary handoff before it.
describe('toChatMessages with a compacted producer notice', () => {
  it('builds the async result from display_content', () => {
    const envelope = [
      '[ASYNC DELEGATION BATCH COMPLETE — deleg_x]',
      'A background fan-out unit you dispatched earlier — 1 subagent(s) — has finished.',
      '',
      '--- ✓ TASK 1/1: Summarize the notes  (status=completed) ---',
      'the result'
    ].join('\n')

    const [message] = toChatMessages([
      {
        role: 'user',
        content: `[CONTEXT SUMMARY]: old work\n--- END OF CONTEXT SUMMARY ---\n\n${envelope}`,
        display_content: envelope,
        display_kind: 'async_delegation_complete',
        display_metadata: { task_count: 1 },
        timestamp: 100
      }
    ] as never)

    expect(message.role).toBe('system')
    expect(message.asyncResult).toBe('the result')
  })
})
