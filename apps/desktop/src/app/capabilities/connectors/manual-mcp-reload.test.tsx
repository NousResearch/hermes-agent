import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesGateway } from '@/hermes'
import { TRANSLATIONS } from '@/i18n'

import { ManualMcpReload } from './manual-mcp-reload'

afterEach(cleanup)

function setup(request = vi.fn().mockResolvedValue({ status: 'reloaded' })) {
  render(<ManualMcpReload gateway={{ request } as unknown as HermesGateway} sessionId="session-1" />)

  return request
}

describe('ManualMcpReload', () => {
  it('localizes the shared-pool warning for supported Connector locales', () => {
    expect(TRANSLATIONS.de.connectorsPage.manualReload.warning).toContain('Sitzungen')
    expect(TRANSLATIONS.fr.connectorsPage.manualReload.warning).toContain('sessions')
    expect(TRANSLATIONS.es.connectorsPage.manualReload.warning).toContain('sesiones')
  })

  it('requires an explicit confirmation before reloading the process-wide pool', async () => {
    const request = setup()
    fireEvent.click(screen.getByRole('button', { name: 'Reload MCP tools' }))
    expect(request).not.toHaveBeenCalled()
    expect(screen.getByRole('dialog').textContent).toContain('all open sessions')
    expect(screen.getByRole('dialog').textContent).toContain('pending turns')
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(request).not.toHaveBeenCalled()
  })

  it('calls the existing gateway RPC once and reports success only on a reloaded status', async () => {
    const request = setup()
    fireEvent.click(screen.getByRole('button', { name: 'Reload MCP tools' }))
    fireEvent.click(screen.getByRole('button', { name: 'Reload now' }))
    await waitFor(() => expect(request).toHaveBeenCalledWith('reload.mcp', { confirm: true, session_id: 'session-1' }))
    expect(request).toHaveBeenCalledTimes(1)
    await screen.findByText('MCP tools reloaded')
  })

  it('keeps the confirmation open with an error when the RPC did not reload', async () => {
    const request = setup(vi.fn().mockResolvedValue({ status: 'confirm_required' }))
    fireEvent.click(screen.getByRole('button', { name: 'Reload MCP tools' }))
    fireEvent.click(screen.getByRole('button', { name: 'Reload now' }))
    await screen.findByText('MCP tools were not reloaded')
    expect(request).toHaveBeenCalledTimes(1)
  })

  it('keeps the confirmation open and never announces success when the RPC rejects', async () => {
    const request = setup(vi.fn().mockRejectedValue(new Error('Host unavailable')))
    fireEvent.click(screen.getByRole('button', { name: 'Reload MCP tools' }))
    fireEvent.click(screen.getByRole('button', { name: 'Reload now' }))
    await screen.findByText('Host unavailable')
    expect(screen.getByRole('dialog')).toBeTruthy()
    expect(screen.queryByRole('status')).toBeNull()
    expect(request).toHaveBeenCalledTimes(1)
  })

  it('disables the action without a live gateway or a selected session', () => {
    render(<ManualMcpReload gateway={null} sessionId={null} />)
    expect((screen.getByRole('button', { name: 'Reload MCP tools' }) as HTMLButtonElement).disabled).toBe(true)
    cleanup()
    const request = vi.fn()
    render(<ManualMcpReload gateway={{ request } as unknown as HermesGateway} sessionId={null} />)
    const button = screen.getByRole('button', { name: 'Reload MCP tools' }) as HTMLButtonElement
    expect(button.disabled).toBe(true)
    fireEvent.click(button)
    expect(request).not.toHaveBeenCalled()
  })
})
