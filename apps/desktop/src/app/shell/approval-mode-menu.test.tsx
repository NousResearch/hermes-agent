import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { StatusbarControls } from '@/app/shell/statusbar-controls'
import { I18nProvider } from '@/i18n'
import { $approvalModes } from '@/store/approval-mode'

import { useApprovalModeStatusbarItem } from './approval-mode-menu'

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

beforeAll(() => {
  vi.stubGlobal('ResizeObserver', TestResizeObserver)
  Element.prototype.hasPointerCapture ??= () => false
  Element.prototype.setPointerCapture ??= () => undefined
  Element.prototype.releasePointerCapture ??= () => undefined
  HTMLElement.prototype.scrollIntoView ??= () => undefined
})

afterEach(() => {
  cleanup()
  $approvalModes.set({})
})

function Harness({
  profile = 'default',
  requestGateway
}: {
  profile?: string
  requestGateway: (method: string, params?: Record<string, unknown>) => Promise<unknown>
}) {
  const item = useApprovalModeStatusbarItem(profile, requestGateway)

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
    const trigger = within(statusbar).getByRole('button', { name: /manual/i })
    expect(within(statusbar).getAllByRole('button')).toHaveLength(1)

    fireEvent.pointerDown(trigger, { button: 0 })

    expect(await screen.findByRole('menuitemradio', { name: /manual/i })).toBeTruthy()
    expect(trigger.getAttribute('aria-haspopup')).toBe('menu')
    expect(screen.queryByRole('menuitemradio', { name: /smart/i })).toBeNull()
    expect(screen.getByRole('menuitemradio', { name: /off/i })).toBeTruthy()
  })

  it('writes the selected mode through the gateway and updates its shared trigger label', async () => {
    const requestGateway = vi.fn(async (_method, params) => ({ value: params?.value ?? 'manual' }))
    render(<Harness profile="work" requestGateway={requestGateway} />)

    fireEvent.pointerDown(screen.getByRole('button', { name: /manual/i }), { button: 0 })
    fireEvent.click(await screen.findByRole('menuitemradio', { name: /off/i }))

    await waitFor(() => {
      expect(requestGateway).toHaveBeenCalledWith('config.set', { key: 'approvals.mode', value: 'off' })
      expect(screen.getByRole('button', { name: /off/i })).toBeTruthy()
    })
  })

  it('renders the shared trigger and menu in the active locale', async () => {
    const response = new Promise<never>(() => undefined)
    render(
      <I18nProvider configClient={null} initialLocale="ja">
        <Harness requestGateway={vi.fn(() => response)} />
      </I18nProvider>
    )

    fireEvent.pointerDown(screen.getByRole('button', { name: '手動' }), { button: 0 })

    expect(await screen.findByText('承認が必要な操作の前に確認します')).toBeTruthy()
    expect(screen.getByText('承認プロンプトなしで実行します')).toBeTruthy()
  })
})
