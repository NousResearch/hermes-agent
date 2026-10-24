import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { buildToolView } from '@/components/assistant-ui/tool/fallback-model'

import { renderMessageStream } from './test-harness'

afterEach(cleanup)

describe('observed corrections and open tools', () => {
  it.each(['steer', 'redirect'] as const)('preserves %s tool semantics across the input boundary and later result', kind => {
    const h = renderMessageStream('s')
    const toolRows = () =>
      h.state().messages.flatMap(message =>
        message.parts.flatMap(part =>
          part.type === 'tool-call' && part.toolCallId === 'call' ? [{ messageId: message.id, part }] : []
        )
      )

    act(() => {
      h.handleEvent({
        type: 'message.start',
        session_id: 's',
        turn: { id: 'run', source: { kind: 'unknown' } }
      })
      h.handleEvent({
        type: 'tool.start',
        session_id: 's',
        payload: { name: 'terminal', tool_id: 'call', args: { command: 'run-task' }, timestamp: 101 }
      })
    })
    const originalMessageId = toolRows()[0]?.messageId
    expect(originalMessageId).toEqual(expect.any(String))
    expect(toolRows()[0]?.part.interrupted).toBeUndefined()

    act(() => {
      h.handleEvent({
        type: 'message.input',
        session_id: 's',
        turn: { id: 'run', source: { kind: 'unknown' } },
        payload: {
          kind,
          input: { role: 'user', text: 'Also check the log', display_kind: 'steer' },
          inputs: [{ id: 'correction' }]
        }
      })
    })
    expect(toolRows()).toHaveLength(1)
    expect(toolRows()[0]?.part.interrupted).toBe(kind === 'redirect' ? true : undefined)
    expect(h.state().messages.map(message => message.role)).toEqual(['assistant', 'user'])
    expect(h.state().busy).toBe(true)
    expect(h.state().interrupted).toBe(false)

    const result = { success: true, output: 'done' }
    act(() => {
      h.handleEvent({
        type: 'tool.complete',
        session_id: 's',
        payload: { name: 'terminal', tool_id: 'call', result, timestamp: 120 }
      })
    })
    expect(toolRows()).toHaveLength(1)
    expect(toolRows()[0]).toMatchObject({ messageId: originalMessageId, part: { result } })
    expect(buildToolView(toolRows()[0]!.part, '').status).toBe('success')
    expect(h.state().messages.map(message => message.role)).toEqual(['assistant', 'user'])
    expect(h.state().streamId).toBeNull()
  })
})
