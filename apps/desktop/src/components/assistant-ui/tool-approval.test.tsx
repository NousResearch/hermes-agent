import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesGateway } from '@/hermes'
import { I18nProvider } from '@/i18n'
import { $gateway } from '@/store/gateway'
import { $approvalRequest, clearAllPrompts, setApprovalRequest } from '@/store/prompts'
import { $activeSessionId } from '@/store/session'

import { PendingToolApproval } from './tool-approval'
import type { ToolPart } from './tool-fallback-model'

function part(toolName: string): ToolPart {
  return { toolName, type: `tool-${toolName}` } as unknown as ToolPart
}

function setRequest(command = 'rm -rf /tmp/x') {
  $activeSessionId.set('sess-1')
  setApprovalRequest({ command, description: 'dangerous command', sessionId: 'sess-1' })
}

function mockGateway() {
  const request = vi.fn().mockResolvedValue({ resolved: true })
  $gateway.set({ request } as unknown as HermesGateway)

  return request
}

function renderWithI18n(ui: React.ReactElement) {
  return render(<I18nProvider configClient={null}>{ui}</I18nProvider>)
}

afterEach(() => {
  cleanup()
  clearAllPrompts()
  $activeSessionId.set(null)
  $gateway.set(null)
})

describe('PendingToolApproval', () => {
  it('renders nothing when there is no pending approval', () => {
    const { container } = renderWithI18n(<PendingToolApproval part={part('terminal')} />)

    expect(container.innerHTML).toBe('')
  })

  it('renders nothing for tools that never raise approval', () => {
    setRequest()
    const { container } = renderWithI18n(<PendingToolApproval part={part('read_file')} />)

    expect(container.innerHTML).toBe('')
  })

  it('renders the inline run/reject controls on the pending terminal row', () => {
    setRequest('chmod -R 777 /tmp/x')
    renderWithI18n(<PendingToolApproval part={part('terminal')} />)

    expect(screen.getByRole('button', { name: /Run/ })).toBeTruthy()
    expect(screen.getByRole('button', { name: /Reject/ })).toBeTruthy()
  })

  it('sends approval.respond {choice: "once"} and clears the request on Run', async () => {
    const request = mockGateway()
    setRequest()
    renderWithI18n(<PendingToolApproval part={part('terminal')} />)

    fireEvent.click(screen.getByRole('button', { name: /Run/ }))

    await waitFor(() => {
      expect(request).toHaveBeenCalledWith('approval.respond', { choice: 'once', session_id: 'sess-1' })
    })
    expect($approvalRequest.get()).toBeNull()
  })

  it('sends choice "deny" on Reject', async () => {
    const request = mockGateway()
    setRequest()
    renderWithI18n(<PendingToolApproval part={part('terminal')} />)

    fireEvent.click(screen.getByRole('button', { name: /Reject/ }))

    await waitFor(() => {
      expect(request).toHaveBeenCalledWith('approval.respond', { choice: 'deny', session_id: 'sess-1' })
    })
  })
})
