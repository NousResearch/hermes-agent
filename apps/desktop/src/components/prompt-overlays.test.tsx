import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesGateway } from '@/hermes'
import { $clarifyRequests, setClarifyRequest } from '@/store/clarify'
import { $gateway } from '@/store/gateway'
import { clearAllPrompts } from '@/store/prompts'
import { $activeSessionId } from '@/store/session'

import { PromptOverlays } from './prompt-overlays'

function mockGateway() {
  const request = vi.fn().mockResolvedValue({ ok: true })
  $gateway.set({ request } as unknown as HermesGateway)

  return request
}

afterEach(() => {
  cleanup()
  clearAllPrompts()
  $clarifyRequests.set({})
  $activeSessionId.set(null)
  $gateway.set(null)
  vi.restoreAllMocks()
})

describe('PromptOverlays clarify dialog', () => {
  it('renders the active clarify request as a modal and answers a choice', async () => {
    const request = mockGateway()
    $activeSessionId.set('sess-1')
    setClarifyRequest({
      requestId: 'req-1',
      question: 'Which option?',
      choices: ['Alpha', 'Beta'],
      sessionId: 'sess-1'
    })

    render(<PromptOverlays />)

    expect(screen.getByText('Which option?')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Alpha' }))

    await waitFor(() => {
      expect(request).toHaveBeenCalledWith('clarify.respond', {
        request_id: 'req-1',
        answer: 'Alpha'
      })
    })
    expect($clarifyRequests.get()['sess-1']).toBeUndefined()
  })

  it('advances to the next queued clarify for the same session after the first resolves', async () => {
    const request = mockGateway()
    $activeSessionId.set('sess-1')
    setClarifyRequest({
      requestId: 'req-1',
      question: 'First question?',
      choices: ['First answer'],
      sessionId: 'sess-1'
    })
    setClarifyRequest({
      requestId: 'req-2',
      question: 'Second question?',
      choices: ['Second answer'],
      sessionId: 'sess-1'
    })

    render(<PromptOverlays />)

    expect(screen.getByText('First question?')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'First answer' }))

    await waitFor(() => {
      expect(request).toHaveBeenCalledWith('clarify.respond', {
        request_id: 'req-1',
        answer: 'First answer'
      })
    })

    await waitFor(() => {
      expect(screen.getByText('Second question?')).toBeTruthy()
    })
    expect($clarifyRequests.get()['sess-1']?.[0]?.requestId).toBe('req-2')
  })
})
