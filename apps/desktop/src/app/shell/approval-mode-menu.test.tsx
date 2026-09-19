import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { StatusbarControls } from '@/app/shell/statusbar-controls'
import { I18nProvider } from '@/i18n'
import { $approvalModes } from '@/store/approval-mode'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

import { useApprovalModeStatusbarItem } from './approval-mode-menu'

beforeAll(() => {
  stubResizeObserver()
  stubMenuDomApis()
})

afterEach(() => {
  cleanup()
  $approvalModes.set({})
})

function Harness({
  gatewayState = 'open',
  profile = 'default',
  requestGateway
}: {
  gatewayState?: string
  profile?: string
  requestGateway: (method: string, params?: Record<string, unknown>) => Promise<unknown>
}) {
  const item = useApprovalModeStatusbarItem(profile, requestGateway, gatewayState)

  return (
    <MemoryRouter>
      <StatusbarControls items={[item]} />
    </MemoryRouter>
  )
}

describe('approval mode statusbar item', () => {
  it('uses the shared statusbar menu trigger without a nested bespoke button', async () => {
    const response = new Promise<never>(() => undefined)
    render(<Harness requestGateway={vi.fn(() => response)} />)

    const statusbar = screen.getByRole('contentinfo')
    const trigger = within(statusbar).getByRole('button', { name: /smart/i })
    expect(within(statusbar).getAllByRole('button')).toHaveLength(1)

    fireEvent.pointerDown(trigger, { button: 0 })

    expect(await screen.findByRole('menuitemradio', { name: /manual/i })).toBeTruthy()
    expect(trigger.getAttribute('aria-haspopup')).toBe('menu')
    expect(screen.getByRole('menuitemradio', { name: /smart/i })).toBeTruthy()
    expect(screen.getByRole('menuitemradio', { name: /off/i })).toBeTruthy()
  })

  it('writes the selected mode through the gateway and updates its shared trigger label', async () => {
    const requestGateway = vi.fn(async (_method, params) => ({ value: params?.value ?? 'smart' }))
    render(<Harness profile="work" requestGateway={requestGateway} />)

    fireEvent.pointerDown(screen.getByRole('button', { name: /smart/i }), { button: 0 })
    fireEvent.click(await screen.findByRole('menuitemradio', { name: /manual/i }))

    await waitFor(() => {
      expect(requestGateway).toHaveBeenCalledWith('config.set', { key: 'approvals.mode', value: 'manual' })
      expect(screen.getByRole('button', { name: /manual/i })).toBeTruthy()
    })
  })

  it('renders the shared trigger and menu in the active locale', async () => {
    const response = new Promise<never>(() => undefined)
    render(
      <I18nProvider configClient={null} initialLocale="ja">
        <Harness requestGateway={vi.fn(() => response)} />
      </I18nProvider>
    )

    fireEvent.pointerDown(screen.getByRole('button', { name: 'スマート' }), { button: 0 })

    expect(await screen.findByText('必要な場合にのみ確認します')).toBeTruthy()
    expect(screen.getByText('承認プロンプトなしで実行します')).toBeTruthy()
  })

  it('syncs once the gateway opens instead of sitting on the pre-sync default', async () => {
    // Boot order: the shell (and this chip) mount while the socket is still
    // connecting, so the sync must wait for 'open' — and then actually run.
    // A sync attempted while connecting rejects ("Hermes gateway
    // unavailable") and, swallowed, would leave the chip on its optimistic
    // 'smart' default for the whole session even though the configured mode
    // is 'off'.
    const requestGateway = vi.fn(async () => ({ value: 'off' }))
    const { rerender } = render(
      <Harness gatewayState="connecting" profile="boot" requestGateway={requestGateway} />
    )

    expect(requestGateway).not.toHaveBeenCalled()
    expect(screen.getByRole('button', { name: /smart/i })).toBeTruthy()

    rerender(<Harness gatewayState="open" profile="boot" requestGateway={requestGateway} />)

    await waitFor(() => {
      expect(requestGateway).toHaveBeenCalledWith('config.get', { key: 'approvals.mode' })
      expect(screen.getByRole('button', { name: /off/i })).toBeTruthy()
    })
  })
})
