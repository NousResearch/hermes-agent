import { describe, expect, it } from 'vitest'

import { buildToolView, resolveSealedToolResult } from './fallback-model'
import type { ToolPart } from './fallback-model'

function sealedPart(toolCallId: string): ToolPart {
  return {
    type: 'tool-call',
    toolCallId,
    toolName: 'terminal',
    args: { command: 'ls /tmp' },
    completedAt: 1_750_000_000
  }
}

function threadMessage(toolCallId: string, result: unknown) {
  return {
    parts: [{ type: 'tool-call', toolCallId, result }]
  }
}

describe('resolveSealedToolResult', () => {
  it('finds the result in a duplicate part from a later tail message', () => {
    const messages = [{ parts: [{ type: 'tool-call', toolCallId: 'call-1' }] }, threadMessage('call-1', 'ok')]

    expect(resolveSealedToolResult(messages, 'call-1')).toBe('ok')
  })

  it('finds the result in a duplicate part from an earlier message', () => {
    const messages = [threadMessage('call-1', { output: 'ok' }), { parts: [{ type: 'tool-call', toolCallId: 'call-1' }] }]

    expect(resolveSealedToolResult(messages, 'call-1')).toEqual({ output: 'ok' })
  })

  it('returns undefined when no sibling part carries a result', () => {
    const messages = [
      { parts: [{ type: 'tool-call', toolCallId: 'call-1' }] },
      { parts: [{ type: 'tool-call', toolCallId: 'call-1', result: undefined }] }
    ]

    expect(resolveSealedToolResult(messages, 'call-1')).toBeUndefined()
  })

  it('does not match a different tool_call_id', () => {
    const messages = [threadMessage('call-2', 'ok')]

    expect(resolveSealedToolResult(messages, 'call-1')).toBeUndefined()
  })

  it('returns undefined for missing messages or a missing id', () => {
    expect(resolveSealedToolResult(undefined, 'call-1')).toBeUndefined()
    expect(resolveSealedToolResult([], 'call-1')).toBeUndefined()
    expect(resolveSealedToolResult([threadMessage('call-1', 'ok')], undefined)).toBeUndefined()
  })
})

describe('sealed part repaint', () => {
  it('paints "Result unavailable" for a sealed resultless part', () => {
    const view = buildToolView(sealedPart('call-1'), '')

    expect(view.title).toBe('Result unavailable')
  })

  it('paints the real title once the thread result is re-attached', () => {
    const messages = [{ parts: [{ type: 'tool-call', toolCallId: 'call-1' }] }, threadMessage('call-1', 'ok')]
    const resolved = resolveSealedToolResult(messages, 'call-1')
    const view = buildToolView({ ...sealedPart('call-1'), result: resolved }, '')

    expect(view.title).not.toBe('Result unavailable')
  })
})
