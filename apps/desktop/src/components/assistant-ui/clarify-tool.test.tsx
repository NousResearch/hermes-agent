import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $clarifyRequests, setClarifyRequest } from '@/store/clarify'
import { $gateway } from '@/store/gateway'
import { $activeSessionId } from '@/store/session'

import { ClarifyTool } from './clarify-tool'

vi.mock('@assistant-ui/react', () => ({
  useAuiState: () => true
}))

afterEach(() => {
  cleanup()
  $activeSessionId.set(null)
  $clarifyRequests.set({})
  $gateway.set(null)
})

describe('ClarifyTool', () => {
  it('keeps a typed answer when the pending panel remounts', () => {
    const question = 'What deployment context should I use?'
    $activeSessionId.set('session-a')
    setClarifyRequest({ choices: null, question, requestId: 'req-a', sessionId: 'session-a' })

    const props = { args: { question }, result: undefined } as Parameters<typeof ClarifyTool>[0]
    const first = render(<ClarifyTool {...props} />)

    fireEvent.change(screen.getByPlaceholderText(/Type your answer/), {
      target: { value: 'Use staging first, then promote to production after smoke tests.' }
    })

    first.unmount()
    render(<ClarifyTool {...props} />)

    expect(screen.getByDisplayValue('Use staging first, then promote to production after smoke tests.')).toBeTruthy()
  })
})
