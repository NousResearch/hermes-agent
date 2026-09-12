import { describe, expect, it } from 'vitest'

import { summarizeToolRun } from '@/components/assistant-ui/tool/run-summary'
import { upsertToolPart } from '@/lib/chat-messages'

describe('D1 competing implementation contracts', () => {
  it('retains JSON object text as the canonical received value', () => {
    const result = '{"value":1}'
    const part = upsertToolPart([], { name: 'terminal', tool_id: 'a', result }, 'complete', 1)[0]
    expect(part.type === 'tool-call' && part.result).toBe(result)
  })
  it('keeps identified parallel running calls separate even with identical args', () => {
    let parts = upsertToolPart([], { name: 'terminal', tool_id: 'a', args: { command: 'pwd' } }, 'running', 1)
    parts = upsertToolPart(parts, { name: 'terminal', tool_id: 'b', args: { command: 'pwd' } }, 'running', 2)
    expect(parts).toHaveLength(2)
  })
  it('honors explicit successful skill results over stale envelope errors', () => {
    const part = { type: 'tool-call' as const, toolName: 'skill_view', args: { name: 'loaded-skill' }, result: { success: true }, isError: true }
    expect(summarizeToolRun([part], false)).toContain('Loaded skill')
  })
  it('does not report a failed skill as loaded in the settled summary', () => {
    const summary = summarizeToolRun([{ toolName: 'skill_view', args: { name: 'missing-skill' }, result: { error: 'missing' }, isError: true }], false)
    expect(summary.toLowerCase()).toContain('failed')
    expect(summary).not.toContain('Loaded')
  })
})
