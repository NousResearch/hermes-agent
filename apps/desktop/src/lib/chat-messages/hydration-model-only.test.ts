import { describe, expect, it } from 'vitest'

import { toChatMessages } from './hydration'

// A micro-compaction merge (`display_metadata.model_only`) is skipped, but the ids of the rows after it
// keep their stored position: filtering first shifted every later message and tool-part id by one.
describe('toChatMessages with a model-only merge row', () => {
  it('skips the merge and keeps later ids keyed by stored position', () => {
    const messages = [
      { role: 'user', content: 'Q1\n\nQ2', timestamp: 100, display_metadata: { model_only: true } },
      { role: 'user', content: 'Q3', timestamp: 200 },
      { role: 'assistant', content: 'A3', timestamp: 201 }
    ]

    const result = toChatMessages(messages as never)

    expect(result.map(m => m.role)).toEqual(['user', 'assistant'])
    expect(result[0].id).toBe('200-1-user')
    expect(result[1].id).toBe('201-2-assistant')
  })
})
