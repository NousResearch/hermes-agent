import { cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $gateway } from '@/store/gateway'
import { $userInputRequests } from '@/store/user-input'

import { UserInputCard } from './user-input-card'

const request = {
  context: 'Need a decision',
  expiresAt: 0,
  questions: [{
    allowFreeText: false,
    defaultValue: undefined,
    id: 'choice',
    options: ['a', 'b'],
    text: 'Pick one'
  }],
  requestId: 'request-1',
  sessionId: 'session-1',
  status: 'pending' as const,
  turnId: 'turn-1'
}

afterEach(() => {
  cleanup()
  $gateway.set(null)
  $userInputRequests.set({})
})

describe('UserInputCard', () => {
  it('sends one request when submit is triggered twice before acknowledgement', async () => {
    let resolveResponse: ((value: unknown) => void) | undefined
    const requestCall = vi.fn(() => new Promise(resolve => { resolveResponse = resolve }))
    $gateway.set({ request: requestCall } as never)
    $userInputRequests.set({ 'session-1': [request] })

    const { container } = render(<UserInputCard sessionId="session-1" />)
    fireEvent.click(container.querySelector('input[type="radio"]') as HTMLInputElement)
    const form = container.querySelector('form') as HTMLFormElement

    fireEvent.submit(form)
    fireEvent.submit(form)

    expect(requestCall).toHaveBeenCalledTimes(1)
    resolveResponse?.({ accepted: true, status: 'answered' })
    await waitFor(() => expect($userInputRequests.get()['session-1']).toBeUndefined())
  })
})
