import { act, cleanup, fireEvent, render, within } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

beforeEach(() => {
  vi.resetModules()
  localStorage.clear()
  window.document.documentElement.dataset.hermesDesktopHost = 'browser'
})

afterEach(() => {
  cleanup()
  window.document.body.replaceChildren()
  delete window.document.documentElement.dataset.hermesDesktopHost
})

it('renders the resolved browser binding in the first-run tip, tooltips and settings, including remaps', async () => {
  const { TipBubble } = await import('@/components/tips/tip-bubble')
  const { TipKeybindLabel } = await import('@/components/ui/tooltip')
  const { KeybindSettings } = await import('@/app/settings/keybind-settings')
  const { I18nProvider } = await import('@/i18n')
  const { TIP_CATALOG } = await import('@/lib/tips/catalog')
  const { setBinding } = await import('@/store/keybinds')
  const tip = TIP_CATALOG.find(entry => entry.id === 'new-session')!
  const anchor = window.document.createElement('button')
  window.document.body.append(anchor)

  const view = render(
    <I18nProvider configClient={null} initialLocale="en">
      <TipBubble anchor={anchor} keybind={tip.keybind} onClose={() => {}} side="right" text="Start a new session" />
      <div data-testid="shortcut-labels">
        <TipKeybindLabel actionId="session.new" />
        <TipKeybindLabel actionId="session.newTab" />
        <TipKeybindLabel actionId="view.closeTab" />
      </div>
      <div data-testid="settings">
        <KeybindSettings subpage="shortcuts" />
      </div>
    </I18nProvider>
  )

  const status = () => view.getByRole('status').textContent
  expect(status()).toContain('Alt')
  expect(within(view.getByRole('status')).getByText('N')).toBeTruthy()
  const labels = within(view.getByTestId('shortcut-labels'))
  const labelText = view.getByTestId('shortcut-labels').textContent

  for (const key of ['N', 'T', 'W']) {
    expect(labelText).toContain(`Alt+${key}`)
  }

  expect(labelText).not.toContain('Ctrl')

  const search = view.getByRole('textbox', { name: 'Search shortcuts…' })
  fireEvent.change(search, { target: { value: 'session.new' } })
  const row = within(view.getByTestId('settings')).getByText('New session').parentElement!
  expect(within(row).getByText('Alt')).toBeTruthy()
  expect(within(row).getByText('N')).toBeTruthy()
  expect(within(row).queryByRole('button', { name: 'Reset to default' })).toBeNull()

  act(() => setBinding('session.new', ['mod+shift+f6']))
  expect(within(view.getByRole('status')).getByText('F6')).toBeTruthy()
  expect(labels.getByText(/F6$/)).toBeTruthy()
  expect(within(row).getByText('F6')).toBeTruthy()
  fireEvent.click(within(row).getByRole('button', { name: 'Reset to default' }))
  expect(within(row).getByText('N')).toBeTruthy()

  act(() => setBinding('session.new', []))
  expect(view.getByRole('status').querySelector('kbd')).toBeNull()
  expect(view.getByTestId('shortcut-labels').textContent).not.toContain('Alt+N')
})
