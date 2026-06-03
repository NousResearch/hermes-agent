import { type ToolCallMessagePartProps } from '@assistant-ui/react'
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { $clarifyRequest } from '@/store/clarify'

import { ClarifyTool } from './clarify-tool'

function clarifyProps(toolCallId: string): ToolCallMessagePartProps {
  return {
    args: { question: 'How should Hermes continue?' },
    isError: false,
    result: undefined,
    toolCallId,
    toolName: 'clarify'
  } as ToolCallMessagePartProps
}

describe('ClarifyTool', () => {
  it('preserves freeform draft text across remounts for the same tool call', () => {
    $clarifyRequest.set({
      choices: null,
      question: 'How should Hermes continue?',
      requestId: 'clarify-request-1',
      sessionId: 'session-1'
    })

    const first = render(<ClarifyTool {...clarifyProps('tool-call-1')} />)

    fireEvent.change(screen.getByPlaceholderText('Type your answer…'), {
      target: { value: 'Use the official API endpoint.' }
    })

    first.unmount()

    render(<ClarifyTool {...clarifyProps('tool-call-1')} />)

    expect((screen.getByPlaceholderText('Type your answer…') as HTMLTextAreaElement).value).toBe(
      'Use the official API endpoint.'
    )
  })
})
