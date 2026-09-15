import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { StatusbarControls } from '@/app/shell/statusbar-controls'
import { I18nProvider } from '@/i18n'
import { $approvalModes } from '@/store/approval-mode'
import { $statusbarHiddenIds, STATUSBAR_HIDDEN_BY_DEFAULT } from '@/store/statusbar-prefs'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

import { useApprovalModeStatusbarItem } from './approval-mode-menu'

beforeAll(() => {
  stubResizeObserver()
  stubMenuDomApis()
})

afterEach(() => {
  cleanup()
  $approvalModes.set({})
  $statusbarHiddenIds.set([...STATUSBAR_HIDDEN_BY_DEFAULT])
})

function Harness({
  profile = 'default',
  requestGateway,
  sessionYolo
}: {
  profile?: string
  requestGateway: (method: string, params?: Record<string, unknown>) => Promise<unknown>
  sessionYolo?: { active: boolean; onToggle: (enabled: boolean) => void }
}) {
  const item = useApprovalModeStatusbarItem(profile, requestGateway, sessionYolo)

  // The bar spreads this item with a `toggleLabel` (use-statusbar-items), which
  // is what makes it hideable via the context menu — mirror that shape.
  return (
    <MemoryRouter>
      <StatusbarControls items={[{ ...item, toggleLabel: 'Approval mode' }]} />
    </MemoryRouter>
  )
}

describe('approval mode statusbar item', () => {
  it('lights the zap and says YOLO when the session bypass is on but the profile mode is not off', async () => {
    $statusbarHiddenIds.set([])
    const requestGateway = vi.fn(async () => ({ value: 'smart' }))
    const onToggle = vi.fn()
    const { rerender } = render(<Harness requestGateway={requestGateway} sessionYolo={{ active: false, onToggle }} />)

    const before = await screen.findByRole('button', { name: /smart/i })
    expect(before.textContent).not.toMatch(/yolo/i)

    rerender(<Harness requestGateway={requestGateway} sessionYolo={{ active: true, onToggle }} />)

    const trigger = screen.getByRole('button', { name: /yolo/i })
    expect(trigger.textContent).not.toMatch(/smart/i)
    expect(trigger.className).toContain('bg-(--chrome-action-hover)')

    fireEvent.pointerDown(trigger, { button: 0 })
    const row = await screen.findByRole('menuitemcheckbox', { name: /yolo for this chat/i })
    expect(row.getAttribute('aria-checked')).toBe('true')
    expect(row.getAttribute('aria-disabled')).not.toBe('true')

    fireEvent.click(row)
    expect(onToggle).toHaveBeenCalledWith(false)
  })

  it('keeps the profile mode label when it is off and disables the redundant session row', async () => {
    $statusbarHiddenIds.set(['approval-mode'])
    const requestGateway = vi.fn(async () => ({ value: 'off' }))
    render(<Harness requestGateway={requestGateway} sessionYolo={{ active: true, onToggle: vi.fn() }} />)

    const trigger = await screen.findByRole('button', { name: /^off$/i })
    expect(trigger.textContent).not.toMatch(/yolo/i)

    fireEvent.pointerDown(trigger, { button: 0 })
    const row = await screen.findByRole('menuitemcheckbox', { name: /yolo for this chat/i })
    expect(row.getAttribute('aria-disabled')).toBe('true')
  })

  it('renders no session row when the surface has no session yolo control', async () => {
    $statusbarHiddenIds.set([])
    const response = new Promise<never>(() => undefined)
    render(<Harness requestGateway={vi.fn(() => response)} />)

    fireEvent.pointerDown(screen.getByRole('button', { name: /smart/i }), { button: 0 })
    await screen.findByRole('menuitemradio', { name: /manual/i })
    expect(screen.queryByRole('menuitemcheckbox')).toBeNull()
  })
  it('surfaces a user-hidden pill while a bypass is active, and hides it again after', async () => {
    $statusbarHiddenIds.set(['approval-mode'])
    const requestGateway = vi.fn(async () => ({ value: 'smart' }))
    const onToggle = vi.fn()
    const { rerender } = render(<Harness requestGateway={requestGateway} sessionYolo={{ active: false, onToggle }} />)

    await waitFor(() => expect(requestGateway).toHaveBeenCalled())
    expect(screen.queryByRole('button', { name: /smart/i })).toBeNull()

    rerender(<Harness requestGateway={requestGateway} sessionYolo={{ active: true, onToggle }} />)
    expect(screen.getByRole('button', { name: /yolo/i })).toBeTruthy()

    rerender(<Harness requestGateway={requestGateway} sessionYolo={{ active: false, onToggle }} />)
    expect(screen.queryByRole('button', { name: /yolo|smart/i })).toBeNull()
  })

  it('uses the shared statusbar menu trigger without a nested bespoke button', async () => {
    $statusbarHiddenIds.set([])
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
    $statusbarHiddenIds.set([])
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
    $statusbarHiddenIds.set([])
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
})
