import { describe, expect, it } from 'vitest'

import { splitWorkRange, type WorkPartLike } from './work-block-split'

const reasoning: WorkPartLike = { type: 'reasoning' }

const tool = (toolName: string): WorkPartLike => ({
  stays: toolName === 'clarify' || toolName === 'delegate_task',
  type: 'tool-call'
})

describe('splitWorkRange', () => {
  it('folds a whole stretch of thinking and activity into one block', () => {
    const parts = [reasoning, tool('read_file'), tool('patch'), tool('terminal'), reasoning]

    expect(splitWorkRange(parts, 0, 4)).toEqual([{ end: 4, kind: 'block', start: 0 }])
  })

  it('keeps questions and subagent cards in place and starts a new block after them', () => {
    const parts = [reasoning, tool('read_file'), tool('delegate_task'), tool('terminal'), tool('clarify')]

    expect(splitWorkRange(parts, 0, 4)).toEqual([
      { end: 1, kind: 'block', start: 0 },
      { index: 2, kind: 'part' },
      { end: 3, kind: 'block', start: 3 },
      { index: 4, kind: 'part' }
    ])
  })

  it('honors the range bounds', () => {
    const parts = [tool('clarify'), tool('read_file'), tool('terminal'), tool('clarify')]

    expect(splitWorkRange(parts, 1, 2)).toEqual([{ end: 2, kind: 'block', start: 1 }])
  })
})
