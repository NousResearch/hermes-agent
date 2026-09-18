import { describe, expect, it } from 'vitest'

import { parseTurnStats, toChatMessages } from './index'

describe('parseTurnStats', () => {
  it('maps a snake_case dict, dropping negative, non-numeric and unrendered keys', () => {
    expect(
      parseTurnStats({
        duration_s: 12,
        input: 3124,
        output: 420,
        reasoning: -1,
        cache_read: 2890,
        cache_write: 0,
        calls: 6,
        cost_usd: 'n/a',
        model: 'kimi-k3',
        provider: 'kimi'
      })
    ).toEqual({
      durationS: 12,
      input: 3124,
      output: 420,
      cacheRead: 2890,
      cacheWrite: 0,
      calls: 6
    })
  })

  it('returns undefined for empty, null, or fully invalid payloads', () => {
    expect(parseTurnStats(undefined)).toBeUndefined()
    expect(parseTurnStats(null)).toBeUndefined()
    expect(parseTurnStats({ input: -4, output: Number.NaN })).toBeUndefined()
  })
})

describe('toChatMessages turn_stats hydration', () => {
  it('covers display_metadata.turn_stats onto the assistant ChatMessage', () => {
    const [assistant] = toChatMessages([
      {
        role: 'assistant',
        content: 'ok',
        timestamp: 1,
        display_metadata: {
          turn_stats: { duration_s: 12, input: 100, output: 20, cache_read: 80, cache_write: 0, calls: 2 }
        }
      }
    ])

    expect(assistant.turnStats).toEqual({
      durationS: 12,
      input: 100,
      output: 20,
      cacheRead: 80,
      cacheWrite: 0,
      calls: 2
    })
  })

  it('keeps the stats when the stamped row folds into a tool-calling bubble', () => {
    // The gateway stamps a turn's stats on its last row carrying text. On a tool-calling
    // turn that row merges into the bubble above it, which is where the strip reads from.
    const messages = toChatMessages([
      {
        content: 'looking that up',
        role: 'assistant',
        timestamp: 1,
        tool_calls: [{ function: { arguments: '{}', name: 'read' }, id: 'call-1' }]
      },
      { content: 'file contents', role: 'tool', timestamp: 2, tool_call_id: 'call-1' },
      {
        content: 'done',
        display_metadata: { turn_stats: { calls: 14, duration_s: 89, input: 53132 } },
        role: 'assistant',
        timestamp: 3
      }
    ])

    const assistants = messages.filter(message => message.role === 'assistant')

    expect(assistants).toHaveLength(1)
    expect(assistants[0].turnStats).toEqual({ calls: 14, durationS: 89, input: 53132 })
  })
})

