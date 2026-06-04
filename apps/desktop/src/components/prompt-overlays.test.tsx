import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesGateway } from '@/hermes'
import { $gateway } from '@/store/gateway'
import { $approvalRequest } from '@/store/prompts'

import { PromptOverlays } from './prompt-overlays'

function setRequest(command = 'rm -rf /tmp/x', description = 'dangerous command') {
  $approvalRequest.set({ command, description, sessionId: 'sess-1' })
}

function mockGateway() {
  const request = vi.fn().mockResolvedValue({ resolved: true })
  $gateway.set({ request } as unknown as HermesGateway)

  return request
}

afterEach(() => {
  cleanup()
  $approvalRequest.set(null)
  $gateway.set(null)
})

describe('ApprovalOverlay', () => {
  it('renders nothing when there is no pending approval', () => {
    const { container } = render(<PromptOverlays />)

    expect(container.querySelector('[data-slot="tool-approval-overlay"]')).toBeNull()
  })

  it('renders the overlay when there is a pending approval', () => {
    setRequest()
    render(<PromptOverlays />)

    const overlay = document.querySelector('[data-slot="tool-approval-overlay"]')

    expect(overlay).not.toBeNull()
    expect(screen.getByText('dangerous command')).toBeTruthy()
  })

  it('shows Run and Reject buttons', () => {
    setRequest()
    render(<PromptOverlays />)

    expect(screen.getByRole('button', { name: /Run/ })).toBeTruthy()
    expect(screen.getByRole('button', { name: /Reject/ })).toBeTruthy()
  })

  it('sends approval.respond {choice: "once"} and clears the request on Run', async () => {
    const request = mockGateway()
    setRequest()
    render(<PromptOverlays />)

    fireEvent.click(screen.getByRole('button', { name: /Run/ }))

    await waitFor(() => {
      expect(request).toHaveBeenCalledWith('approval.respond', { choice: 'once', session_id: 'sess-1' })
    })
    expect($approvalRequest.get()).toBeNull()
  })

  it('sends choice "deny" on Reject', async () => {
    const request = mockGateway()
    setRequest()
    render(<PromptOverlays />)

    fireEvent.click(screen.getByRole('button', { name: /Reject/ }))

    await waitFor(() => {
      expect(request).toHaveBeenCalledWith('approval.respond', { choice: 'deny', session_id: 'sess-1' })
    })
  })

  it('shows command text when available', () => {
    setRequest('chmod -R 777 /tmp/x', 'recursive permission change')
    render(<PromptOverlays />)

    expect(screen.getByText('recursive permission change')).toBeTruthy()
  })

  it('auto-hides when approval request is cleared', () => {
    setRequest()
    const { rerender } = render(<PromptOverlays />)

    expect(document.querySelector('[data-slot="tool-approval-overlay"]')).not.toBeNull()

    $approvalRequest.set(null)
    rerender(<PromptOverlays />)

    expect(document.querySelector('[data-slot="tool-approval-overlay"]')).toBeNull()
  })
})
