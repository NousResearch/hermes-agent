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
  afterEach(() => {
    cleanup()
    vi.mocked(getLogs).mockResolvedValue({ file: 'gui', lines: [] })
  })

  it('shows the fresher of gui.log and gateway.log (#146682)', async () => {
    // gui.log is frozen when the Desktop attaches to the messaging gateway; the panel must
    // not present its days-old lines as current activity.
    vi.mocked(getLogs).mockImplementation(async ({ file }) => ({
      file: file ?? 'gui',
      lines:
        file === 'gateway'
          ? ['2026-10-31 19:04:26,001 INFO gateway is alive']
          : ['2026-10-27 09:00:00,000 WARNING gui stale line']
    }))

    renderPanel('open')

    expect(await screen.findByText(/gateway is alive/)).toBeTruthy()
    expect(screen.queryByText(/gui stale line/)).toBeNull()
  })
})
