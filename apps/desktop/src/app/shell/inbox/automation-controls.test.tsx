import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $activeGatewayProfile } from '@/store/profile'

import { AutomationControls } from './automation-controls'

vi.mock('@/store/gateway', () => ({ $gateway: { get: vi.fn() } }))

vi.mock('@/store/profile', () => ({ $activeGatewayProfile: { get: vi.fn(() => 'test-profile') } }))

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

async function getGatewayMock() {
  const mod = await import('@/store/gateway')

  return vi.mocked(mod.$gateway.get)
}

describe('AutomationControls', () => {
  it('offers Pause for an active goal and sends the allowlisted action', async () => {
    const gw = { request: vi.fn().mockResolvedValue({ status: 'ok' }) }
    ;(await getGatewayMock()).mockReturnValue(gw as never)
    const onChanged = vi.fn()

    render(<AutomationControls kind="goal" liveSessionId="live-1" onChanged={onChanged} status="active" />)

    fireEvent.click(screen.getByText('Pause goal'))

    await waitFor(() => {
      expect(gw.request).toHaveBeenCalledWith('session.control', {
        action: 'goal.pause',
        args: {},
        session_id: 'live-1',
        profile: 'test-profile'
      })
    })
    expect(onChanged).toHaveBeenCalled()
  })

  it('offers Resume for a paused loop', async () => {
    const gw = { request: vi.fn().mockResolvedValue({ status: 'ok' }) }
    ;(await getGatewayMock()).mockReturnValue(gw as never)

    render(<AutomationControls kind="loop" liveSessionId="live-1" status="paused" />)

    fireEvent.click(screen.getByText('Resume loop'))

    await waitFor(() => {
      expect(gw.request).toHaveBeenCalledWith('session.control', {
        action: 'loop.resume',
        args: {},
        session_id: 'live-1',
        profile: 'test-profile'
      })
    })
  })

  it('renders nothing for a finished automation', async () => {
    ;(await getGatewayMock()).mockReturnValue({ request: vi.fn() } as never)

    const { container } = render(<AutomationControls kind="heartbeat" liveSessionId="live-1" status="done" />)

    expect(container.innerHTML).toBe('')
  })

  it('disables the control and says so when the session is not running', async () => {
    ;(await getGatewayMock()).mockReturnValue({ request: vi.fn() } as never)

    render(<AutomationControls kind="goal" liveSessionId="" status="active" />)

    expect(screen.getByText('Pause goal').closest('button')!.disabled).toBe(true)
    expect(screen.getByText('session is not running')).toBeTruthy()
  })

  it('surfaces a failed action instead of pretending it paused', async () => {
    const gw = { request: vi.fn().mockRejectedValue(new Error('session is not live (4009)')) }
    ;(await getGatewayMock()).mockReturnValue(gw as never)
    const onChanged = vi.fn()

    render(<AutomationControls kind="heartbeat" liveSessionId="live-1" onChanged={onChanged} status="active" />)

    fireEvent.click(screen.getByText('Pause heartbeat'))

    await waitFor(() => {
      expect(screen.getByRole('alert').textContent).toContain('session is not live')
    })
    expect(onChanged).not.toHaveBeenCalled()
  })
})
