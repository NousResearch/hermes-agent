import { fromThreadMessageLike, getAutoStatus } from '@assistant-ui/core/internal'
import { describe, expect, it } from 'vitest'

import { buildToolView } from '@/components/assistant-ui/tool/fallback-model'
import { toRuntimeMessage } from '@/lib/chat-runtime'

import { upsertToolPart } from './tool-parts'
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

  it('does not falsely mark empty or non-error payloads as isError on stored result hydration', async () => {
    const { applyStoredToolResult } = await import('./tool-parts')
    const messages = [
      {
        id: 'msg-1',
        role: 'assistant',
        parts: [{ type: 'tool-call', toolCallId: 'call-1', toolName: 'linter', isError: false }]
      }
    ] as any

    // { error: [] } is a linter reporting 0 errors — should NOT be isError: true
    applyStoredToolResult(messages, {
      role: 'tool',
      tool_call_id: 'call-1',
      content: JSON.stringify({ error: [] }),
      timestamp: 10
    } as any)

    expect(messages[0].parts[0].isError).toBe(false)

    // Actual error message string should be isError: true
    const errorMessages = [
      {
        id: 'msg-2',
        role: 'assistant',
        parts: [{ type: 'tool-call', toolCallId: 'call-2', toolName: 'terminal', isError: false }]
      }
    ] as any

    applyStoredToolResult(errorMessages, {
      role: 'tool',
      tool_call_id: 'call-2',
      content: JSON.stringify({ error: 'Command failed with exit code 1' }),
      timestamp: 20
    } as any)

    expect(errorMessages[0].parts[0].isError).toBe(true)
  })
})
