import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  notifyError: vi.fn(),
  reconnectGateway: vi.fn<() => Promise<void>>()
}))

vi.mock('@/components/ui/tooltip', () => ({
  Tip: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock('@/hermes', () => ({
  getLogs: vi.fn().mockResolvedValue({ lines: [] })
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      commandCenter: { restartGateway: 'Restart gateway' },
      shell: {
        gatewayMenu: {
          checkingInference: 'Checking inference',
          connected: 'Connected',
          connecting: 'Connecting',
          disconnected: 'Disconnected',
          inferenceNotReady: 'Inference not ready',
          inferenceReady: 'Inference ready',
          messagingPlatforms: 'Messaging platforms',
          offline: 'Offline',
          openSystem: 'Open system panel',
          recentActivity: 'Recent activity',
          reconnectGateway: 'Reconnect gateway',
          viewAllLogs: 'View all logs'
        }
      }
    }
  })
}))

vi.mock('@/store/gateway-reconnect', () => ({
  reconnectGateway: mocks.reconnectGateway
}))

vi.mock('@/store/notifications', () => ({
  notifyError: mocks.notifyError
}))

vi.mock('@/store/system-actions', () => ({
  runGatewayRestart: vi.fn()
}))

import { getLogs } from '@/hermes'

import { GatewayMenuPanel } from './gateway-menu-panel'

const renderPanel = (gatewayState: string) =>
  render(
    <GatewayMenuPanel
      gatewayState={gatewayState}
      inferenceStatus={null}
      onClose={vi.fn()}
      onOpenSystem={vi.fn()}
      statusSnapshot={null}
    />
  )

describe('GatewayMenuPanel reconnect action', () => {
  beforeEach(() => {
    mocks.reconnectGateway.mockReset().mockResolvedValue(undefined)
    mocks.notifyError.mockReset()
  })

  afterEach(() => cleanup())

  it('shows reconnect while disconnected and disables it in flight', async () => {
    let finish: (() => void) | undefined
    mocks.reconnectGateway.mockImplementation(
      () =>
        new Promise<void>(resolve => {
          finish = resolve
        })
    )

    renderPanel('closed')

    const reconnect = screen.getByRole('button', { name: 'Reconnect gateway' })
    fireEvent.click(reconnect)
    fireEvent.click(reconnect)

    expect(mocks.reconnectGateway).toHaveBeenCalledOnce()
    expect((reconnect as HTMLButtonElement).disabled).toBe(true)

    await act(async () => finish?.())
  })

  it('allows recovery when an open socket stops delivering', async () => {
    renderPanel('open')
    await act(async () => undefined)

    const reconnect = screen.getByRole('button', { name: 'Reconnect gateway' })
    await act(async () => fireEvent.click(reconnect))
    expect(mocks.reconnectGateway).toHaveBeenCalledOnce()
  })
})

describe('GatewayMenuPanel recent activity source', () => {
  const logs = vi.mocked(getLogs)

  beforeEach(() => {
    logs.mockReset().mockResolvedValue({ file: 'gateway', lines: [] })
  })

  afterEach(() => {
    cleanup()
    logs.mockReset().mockResolvedValue({ file: 'gateway', lines: [] })
  })

  it('tails the live gateway log instead of a stale gui log (#155027)', async () => {
    // Under the gateway runtime, gui.log stopped when the backend last ran in
    // mode="gui": its last lines are days old, while gateway.log is live.
    logs.mockImplementation(async ({ file }) => ({
      file: file ?? 'gateway',
      lines:
        file === 'gateway'
          ? ['2026-11-13 13:02:11,004 INFO gateway is alive']
          : ['2026-11-06 08:14:02,871 WARNING gui stale line']
    }))

    renderPanel('open')

    expect(await screen.findByText(/gateway is alive/)).toBeTruthy()
    expect(logs).toHaveBeenCalledWith({ file: 'gateway', lines: 120 })
    expect(screen.queryByText(/gui stale line/)).toBeNull()
  })
})
