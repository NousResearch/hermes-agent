import { fromThreadMessageLike, getAutoStatus } from '@assistant-ui/core/internal'
import { afterEach, describe, expect, it } from 'vitest'

import { buildToolView } from '@/components/assistant-ui/tool/fallback-model'
import { toRuntimeMessage } from '@/lib/chat-runtime'
import { _resetToolCardsForTests, cardToolName } from '@/lib/tool-render-class'
import type { SessionMessage } from '@/types/hermes'

import { applyStoredToolResultToParts, storedToolMessagePart, upsertToolPart } from './tool-parts'
import type { ChatMessagePart } from './types'

describe('live tool result evidence', () => {
  it('preserves every JSON value and display hints through the real runtime adapter and replay', () => {
    const values = [
      'plain text\n',
      '',
      '{"answer":1}',
      '[1,false]',
      0,
      false,
      null,
      [],
      [1, false],
      { summary: 'original', output: 'ok' }
    ]

    for (const [index, result] of values.entries()) {
      const tool_id = `call-${index}`

      const parts = upsertToolPart(
        [],
        { name: 'terminal', tool_id, result, summary: 'abbreviated', duration_s: 0 },
        'complete',
        2
      )

      const [part] = parts
      expect(part.type).toBe('tool-call')

      if (part.type !== 'tool-call') {
        throw new Error('Missing tool call')
      }

      expect(part.result).toBe(result)
      expect(part.toolResultMetadata).toMatchObject({ summary: 'abbreviated', duration_s: 0 })

      const replayed = upsertToolPart(parts, { name: 'terminal', tool_id, message: 'completed' }, 'complete', 3)
      expect((replayed[0] as typeof part).result).toBe(result)

      const runtime = fromThreadMessageLike(
        toRuntimeMessage({ id: tool_id, role: 'assistant', parts: replayed }),
        tool_id,
        getAutoStatus(false, false, false, false, undefined)
      )

      const received = runtime.content[0] as typeof part
      expect(received.result).toEqual(result)
      expect(received.toolResultMetadata).toMatchObject({ summary: 'abbreviated', message: 'completed' })

      if (typeof result === 'object' && result && 'summary' in result) {
        expect((received.result as typeof result).summary).toBe(result.summary)
      }
    }
  })

  it('distinguishes a missing completion result from empty results and keeps parallel completions separate', () => {
    let parts: ChatMessagePart[] = []

    for (const [tool_id, command] of [
      ['a', 'echo a'],
      ['b', 'echo b']
    ]) {
      parts = upsertToolPart(parts, { name: 'terminal', tool_id, args: { command } }, 'running', 1)
    }

    parts = upsertToolPart(parts, { name: 'terminal', tool_id: 'b', result: '' }, 'complete', 2)
    parts = upsertToolPart(parts, { name: 'terminal', tool_id: 'a', summary: 'done' }, 'complete', 3)
    expect(parts).toHaveLength(2)
    const [missing, empty] = parts

    if (missing.type !== 'tool-call' || empty.type !== 'tool-call') {
      throw new Error('Missing tool call')
    }

    expect(missing.result).toBeUndefined()
    expect(missing.completedAt).toBe(3)
    expect(buildToolView(missing, '').status).toBe('warning')
    expect(empty.result).toBe('')
    expect(buildToolView(empty, '').status).toBe('success')
  })
})

describe('gateway-declared tool cards', () => {
  afterEach(() => {
    _resetToolCardsForTests()
  })

  it('learns the card from a live tool.start payload', () => {
    expect(cardToolName('agents_run')).toBeUndefined()

    upsertToolPart(
      [],
      { name: 'agents_run', tool_id: 'c1', card: 'delegate', args: { tasks: [{ goal: 'x' }] } },
      'running',
      1
    )

    expect(cardToolName('agents_run')).toBe('delegate_task')
  })

  it('learns the card from a stored tool row, paired or unpaired', () => {
    const row = {
      role: 'tool',
      tool_name: 'agents_run',
      tool_call_id: 'c1',
      card: 'delegate',
      content: '{}'
    } as SessionMessage

    storedToolMessagePart(row, 0)
    expect(cardToolName('agents_run')).toBe('delegate_task')

    _resetToolCardsForTests()

    const parts = upsertToolPart([], { name: 'agents_run', tool_id: 'c1' }, 'running', 1)

    applyStoredToolResultToParts(parts, row)
    expect(cardToolName('agents_run')).toBe('delegate_task')
  })

  it('learns the card stamped with the result on a cold REST load', () => {
    // The REST transcript prefetch carries no `card`; the stamp lives in display_metadata.
    const row = {
      role: 'tool',
      tool_name: 'agents_run',
      tool_call_id: 'c1',
      content: '{}',
      display_metadata: JSON.stringify({ tool_result_metadata: { card: 'delegate' } })
    } as SessionMessage

    storedToolMessagePart(row, 0)
    expect(cardToolName('agents_run')).toBe('delegate_task')
  })

  it('leaves a tool without a card unclassified', () => {
    upsertToolPart([], { name: 'terminal', tool_id: 'c2' }, 'running', 1)
    storedToolMessagePart({ role: 'tool', tool_name: 'terminal', content: '' } as SessionMessage, 0)

    expect(cardToolName('terminal')).toBeUndefined()
  })
})
