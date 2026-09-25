import { describe, expect, it } from 'vitest'

import {
  appendReasoningPart,
  assistantTextPart,
  completeOpenTimelineParts,
  reasoningPart,
  upsertToolPart
} from './chat-messages'

describe('identified reasoning keeps the existing timeline boundary contract', () => {
  it.each([
    { name: 'text to native', prior: assistantTextPart('Checking.', 1000), sourceId: 'rs:summary:0' },
    { name: 'native to legacy', prior: reasoningPart('Inspect', 1000, 'rs:summary:0'), sourceId: undefined },
    { name: 'legacy to native', prior: reasoningPart('Inspect', 1000), sourceId: 'rs:summary:0' },
    { name: 'different native source', prior: reasoningPart('Inspect', 1000, 'rs:summary:0'), sourceId: 'rs:summary:1' }
  ])('seals $name at the transition, not at final settlement', ({ prior, sourceId }) => {
    const next = appendReasoningPart([prior], 'Compare', 2000, sourceId)

    expect(next).toHaveLength(2)
    expect(next[0]).toEqual({ ...prior, completedAt: 2000 })
    expect(next[1]).toEqual(reasoningPart('Compare', 2000, sourceId))
    expect(prior.completedAt).toBeUndefined()

    const settled = completeOpenTimelineParts(next, 11000)

    expect(settled[0].completedAt).toBe(2000)
    expect(settled[1].completedAt).toBe(11000)
  })

  it('coalesces an open same-source part without reopening completed parts or closing tools', () => {
    const first = reasoningPart('Inspect', 1000, 'rs:summary:0')

    expect(appendReasoningPart([first], ' files', 2000, first.sourceId)).toEqual([
      reasoningPart('Inspect files', 1000, 'rs:summary:0')
    ])

    const sealed = { ...first, completedAt: 1500 }
    const next = appendReasoningPart([sealed], 'Later', 2000, first.sourceId)

    expect(next).toHaveLength(2)
    expect(next[0]).toBe(sealed)
    expect(next[0].completedAt).toBe(1500)

    const tool = upsertToolPart([], { tool_id: 'in-flight', name: 'terminal', args: {} }, 'running')[0]
    const withTool = appendReasoningPart([tool], 'Reasoning during tool work', 2000, 'rs:summary:1')

    expect(withTool[0]).toBe(tool)
    expect(withTool[0].completedAt).toBeUndefined()
    expect(appendReasoningPart([first], 'No timestamp', undefined, 'rs:summary:1')[0]).toBe(first)
  })
})
