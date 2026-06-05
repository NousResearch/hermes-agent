import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesGateway } from '@/hermes'
import { PromptOverlays } from '@/components/prompt-overlays'
import { $gateway } from '@/store/gateway'
import { clearAllPrompts, registerApprovalInline, setApprovalRequest } from '@/store/prompts'
import { $activeSessionId } from '@/store/session'

function setApproval(command = 'op item list --categories login --tags github') {
  $activeSessionId.set('sess-1')
  setApprovalRequest({ command, description: 'credential discovery command', sessionId: 'sess-1' })
}

function mockGateway() {
  const request = vi.fn().mockResolvedValue({ resolved: true })
  $gateway.set({ request } as unknown as HermesGateway)

  return request
}

afterEach(() => {
  cleanup()
  clearAllPrompts()
  $activeSessionId.set(null)
  $gateway.set(null)
})

describe('PromptOverlays approval fallback', () => {
  it('renders a modal approval fallback when no inline approval bar is mounted', () => {
    setApproval()

    render(<PromptOverlays />)

    expect(screen.getByRole('dialog')).toBeTruthy()
    expect(screen.getByText('Command approval required')).toBeTruthy()
    expect(screen.getByText(/op item list/)).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Run once' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Reject' })).toBeTruthy()
  })

  it('does not render the fallback while an inline approval bar is mounted for the active session', () => {
    setApproval()
    const unregister = registerApprovalInline('sess-1')

    try {
      render(<PromptOverlays />)

      expect(screen.queryByRole('dialog')).toBeNull()
    } finally {
      unregister()
    }
  })

  it('sends approval.respond from the fallback dialog and clears the request', async () => {
    const request = mockGateway()
    setApproval()

    render(<PromptOverlays />)
    fireEvent.click(screen.getByRole('button', { name: 'Run once' }))

    await waitFor(() => {
      expect(request).toHaveBeenCalledWith('approval.respond', { choice: 'once', session_id: 'sess-1' })
    })
    expect(screen.queryByRole('dialog')).toBeNull()
  })
})
