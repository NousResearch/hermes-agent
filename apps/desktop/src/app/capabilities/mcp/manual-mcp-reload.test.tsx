// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesGateway, SessionInfo } from '@/hermes'
import { $activeSessionId, $sessions } from '@/store/session'

import { ManualMcpReload } from './manual-mcp-reload'

const notify = vi.hoisted(() => vi.fn())

vi.mock('@/store/notifications', () => ({ notify, notifyError: vi.fn() }))

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  $sessions.set([])
  $activeSessionId.set(null)
})

const sessionFor = (profile: string): SessionInfo => ({ id: 'session-1', profile }) as unknown as SessionInfo

function setup({
  activeSessionId = 'session-1',
  profile = 'default',
  request = vi.fn().mockResolvedValue({ failed: [], status: 'reloaded' })
}: {
  activeSessionId?: null | string
  profile?: string
  request?: ReturnType<typeof vi.fn>
} = {}) {
  $activeSessionId.set(activeSessionId)

  render(<ManualMcpReload gateway={{ request } as unknown as HermesGateway} profile={profile} />)

  return request
}

const openAndConfirm = async () => {
  fireEvent.click(screen.getByRole('button', { name: 'Reload MCP servers' }))
  fireEvent.click(await screen.findByRole('button', { name: 'Reload now' }))
}

describe('ManualMcpReload', () => {
  it('disables the action without a live gateway', () => {
    render(<ManualMcpReload gateway={null} profile="default" />)

    expect((screen.getByRole('button', { name: 'Reload MCP servers' }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('requires confirmation and warns that the next message re-sends the full prompt', async () => {
    const request = setup()

    fireEvent.click(screen.getByRole('button', { name: 'Reload MCP servers' }))

    expect(request).not.toHaveBeenCalled()
    expect(screen.getByRole('dialog').textContent).toContain('prompt-cache')

    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(request).not.toHaveBeenCalled()
  })

  it('reloads the pane profile once confirmed and reports success', async () => {
    $sessions.set([sessionFor('default')])
    const request = setup()

    await openAndConfirm()

    await waitFor(() =>
      expect(request).toHaveBeenCalledWith(
        'reload.mcp',
        { confirm: true, profile: 'default', session_id: 'session-1' },
        expect.any(Number)
      )
    )
    expect(request).toHaveBeenCalledTimes(1)
    await waitFor(() => expect(notify).toHaveBeenCalledWith(expect.objectContaining({ kind: 'success' })))
  })

  it('omits session_id when the active session belongs to another profile', async () => {
    $sessions.set([sessionFor('other')])
    const request = setup()

    await openAndConfirm()

    await waitFor(() =>
      expect(request).toHaveBeenCalledWith('reload.mcp', { confirm: true, profile: 'default' }, expect.any(Number))
    )
  })

  it('works with no active session', async () => {
    const request = setup({ activeSessionId: null })

    await openAndConfirm()

    await waitFor(() =>
      expect(request).toHaveBeenCalledWith('reload.mcp', { confirm: true, profile: 'default' }, expect.any(Number))
    )
  })

  it('names the servers that did not connect', async () => {
    const request = setup({
      request: vi.fn().mockResolvedValue({ failed: ['broken', 'flaky'], status: 'reloaded' })
    })

    await openAndConfirm()

    await waitFor(() =>
      expect(notify).toHaveBeenCalledWith(
        expect.objectContaining({ kind: 'warning', message: expect.stringContaining('broken, flaky') })
      )
    )
    expect(request).toHaveBeenCalledTimes(1)
  })

  it('keeps the dialog open with the backend message when confirmation is still required', async () => {
    setup({ request: vi.fn().mockResolvedValue({ message: 'needs explicit approval', status: 'confirm_required' }) })

    await openAndConfirm()

    expect(await screen.findByText('needs explicit approval')).toBeTruthy()
    expect(notify).not.toHaveBeenCalled()
  })

  it('surfaces a busy turn error inline', async () => {
    setup({ request: vi.fn().mockRejectedValue(new Error('session busy — a turn is running in session-1')) })

    await openAndConfirm()

    expect(await screen.findByText(/session busy/)).toBeTruthy()
    expect(screen.getByRole('dialog')).toBeTruthy()
  })
})
