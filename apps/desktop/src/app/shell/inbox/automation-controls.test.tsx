import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $activeGatewayProfile } from '@/store/profile'

import { AutomationControls } from './automation-controls'

vi.mock('@/store/gateway', () => ({ $gateway: { get: vi.fn() } }))

vi.mock('@/store/profile', () => ({ $activeGatewayProfile: { get: vi.fn(() => 'test-profile') } }))

vi.mock('@/store/inbox', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  refreshInbox: vi.fn().mockResolvedValue({ published: true, snapshot: null })
}))

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

async function getGatewayMock() {
  const mod = await import('@/store/gateway')

  return vi.mocked(mod.$gateway.get)
}

describe('AutomationControls', () => {
  it('offers Pause for an active goal and sends the allowlisted action with the stored key', async () => {
    const gw = { request: vi.fn().mockResolvedValue({ status: 'ok' }) }
    ;(await getGatewayMock()).mockReturnValue(gw as never)
    const onChanged = vi.fn()

    render(
      <AutomationControls
        kind="goal"
        liveSessionId="live-1"
        onChanged={onChanged}
        sessionKey="sess-1"
        sessionLive
        status="active"
      />
    )

    fireEvent.click(screen.getByText('Pause goal'))

    await waitFor(() => {
      expect(gw.request).toHaveBeenCalledWith('session.control', {
        action: 'goal.pause',
        args: {},
        session_id: 'live-1',
        session_key: 'sess-1',
        profile: 'test-profile'
      })
    })
    expect(onChanged).toHaveBeenCalled()

    const { refreshInbox } = await import('@/store/inbox')

    expect(vi.mocked(refreshInbox)).toHaveBeenCalledWith('test-profile', expect.any(Function))
  })

  it('offers Resume for a paused loop', async () => {
    const gw = { request: vi.fn().mockResolvedValue({ status: 'ok' }) }
    ;(await getGatewayMock()).mockReturnValue(gw as never)

    render(
      <AutomationControls kind="loop" liveSessionId="live-1" sessionKey="sess-1" sessionLive status="paused" />
    )

    fireEvent.click(screen.getByText('Resume loop'))

    await waitFor(() => {
      expect(gw.request).toHaveBeenCalledWith('session.control', {
        action: 'loop.resume',
        args: {},
        session_id: 'live-1',
        session_key: 'sess-1',
        profile: 'test-profile'
      })
    })
  })

  it('renders nothing for a finished automation', async () => {
    ;(await getGatewayMock()).mockReturnValue({ request: vi.fn() } as never)

    const { container } = render(
      <AutomationControls kind="heartbeat" liveSessionId="live-1" sessionKey="sess-1" sessionLive status="done" />
    )

    expect(container.innerHTML).toBe('')
  })

  it('stays usable when the session is not running: stored-state pause via the key, no live id sent', async () => {
    const gw = { request: vi.fn().mockResolvedValue({ status: 'ok' }) }
    ;(await getGatewayMock()).mockReturnValue(gw as never)
    const onChanged = vi.fn()

    render(
      <AutomationControls
        kind="goal"
        liveSessionId=""
        onChanged={onChanged}
        sessionKey="stored-9"
        sessionLive={false}
        status="active"
      />
    )

    expect(screen.getByText("session isn't running — applies to stored state")).toBeTruthy()
    expect(screen.getByText('Pause goal').closest('button')!.disabled).toBe(false)

    fireEvent.click(screen.getByText('Pause goal'))

    await waitFor(() => {
      expect(gw.request).toHaveBeenCalledWith('session.control', {
        action: 'goal.pause',
        args: {},
        session_key: 'stored-9',
        profile: 'test-profile'
      })
    })
    const [, payload] = gw.request.mock.calls[0]
    expect('session_id' in payload).toBe(false)
    expect(onChanged).toHaveBeenCalled()

    const { refreshInbox } = await import('@/store/inbox')

    expect(vi.mocked(refreshInbox)).toHaveBeenCalledWith('test-profile', expect.any(Function))
  })

  it('makes no live-status claim while details are still loading', async () => {
    ;(await getGatewayMock()).mockReturnValue({ request: vi.fn() } as never)

    render(
      <AutomationControls kind="goal" liveSessionId="" sessionKey="stored-9" sessionLive={null} status="active" />
    )

    expect(screen.queryByText(/session isn't running/)).toBeNull()
    expect(screen.getByText('Pause goal').closest('button')!.disabled).toBe(false)
  })

  it('surfaces a failed action instead of pretending it paused', async () => {
    const gw = { request: vi.fn().mockRejectedValue(new Error('session is not live (4009)')) }
    ;(await getGatewayMock()).mockReturnValue(gw as never)
    const onChanged = vi.fn()

    render(
      <AutomationControls
        kind="heartbeat"
        liveSessionId="live-1"
        onChanged={onChanged}
        sessionKey="sess-1"
        sessionLive
        status="active"
      />
    )

    fireEvent.click(screen.getByText('Pause heartbeat'))

    await waitFor(() => {
      expect(screen.getByRole('alert').textContent).toContain('session is not live')
    })
    expect(onChanged).not.toHaveBeenCalled()

    const { refreshInbox } = await import('@/store/inbox')

    expect(vi.mocked(refreshInbox)).not.toHaveBeenCalled()
  })
})
