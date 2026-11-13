// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { $chatTextScale } from '@/store/chat-text-scale'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { $shellChoice } from '@/store/shell-variant'
import { $tipsEnabled } from '@/store/tips'
import { $toursEnabled } from '@/store/tours'
import { modePref, ThemeProvider } from '@/themes/context'

import { APPEARANCE_AREAS } from './appearance-contrib'
import { AppearanceSettings } from './appearance-settings'
import { SETTING_IDS, settingElementId } from './settings-manifest'

const disposers: Array<() => void> = []
const initialChoice = $shellChoice.get()
const initialScale = $chatTextScale.get()
const initialTips = $tipsEnabled.get()
const initialTours = $toursEnabled.get()
const profile = normalizeProfileKey($activeGatewayProfile.get())
const initialMode = modePref.resolve(profile)

beforeEach(() => {
  $shellChoice.set(true)
  $chatTextScale.set(110)
  $tipsEnabled.set(true)
  $toursEnabled.set(true)
  Object.defineProperty(Element.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() })
})

afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
  $shellChoice.set(initialChoice)
  $chatTextScale.set(initialScale)
  $tipsEnabled.set(initialTips)
  $toursEnabled.set(initialTours)
  modePref.assign(profile, initialMode)
})

function renderPage(subpage?: string, route = '/settings?tab=appearance') {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <QueryClientProvider client={new QueryClient()}>
        <ThemeProvider>
          <AppearanceSettings subpage={subpage} />
        </ThemeProvider>
      </QueryClientProvider>
    </MemoryRouter>
  )
}

describe('AppearanceSettings extra slot', () => {
  it('mounts plugin extras on the top-level page only, not on deep-link subpages', () => {
    act(() => {
      disposers.push(
        registry.register({
          area: APPEARANCE_AREAS.extra,
          id: 'extra-controls',
          render: () => <span>Extra controls</span>,
          source: 'disk'
        })
      )
    })

    const { unmount } = renderPage('pet')

    expect(screen.queryByText('Extra controls')).toBeNull()
    unmount()

    renderPage()

    expect(screen.getByText('Extra controls')).toBeTruthy()
  })
})

describe('Appearance overview', () => {
  it('applies mode cards, text sizes and the combined tips preference through existing stores', () => {
    renderPage('general')

    for (const name of ['Light', 'Dark', 'System']) {
      const radio = screen.getByRole('radio', { name }) as HTMLInputElement
      fireEvent.click(radio)
      expect(radio.checked).toBe(true)
      expect(modePref.resolve(profile)).toBe(name.toLowerCase())
    }

    for (const [name, scale] of [
      ['Small', 100],
      ['Large', 125],
      ['Default', 110]
    ] as const) {
      fireEvent.click(screen.getByRole('button', { name }))
      expect($chatTextScale.get()).toBe(scale)
    }

    const tips = screen.getByRole('switch', { name: 'Tips & guided tours' })
    fireEvent.click(tips)
    expect($tipsEnabled.get()).toBe(false)
    expect($toursEnabled.get()).toBe(false)
    fireEvent.click(tips)
    expect($tipsEnabled.get()).toBe(true)
    expect($toursEnabled.get()).toBe(true)
  })

  it('keeps advanced controls collapsed until requested and retains custom size presets', () => {
    const { container } = renderPage('general')
    const advanced = container.querySelector('details')!
    expect(advanced.open).toBe(false)

    fireEvent.click(advanced.querySelector('summary')!)
    expect(advanced.open).toBe(true)
    fireEvent.click(screen.getAllByRole('button', { name: '175%' })[1]!)
    expect($chatTextScale.get()).toBe(175)
    // Advanced repeats neither half of the overview's one "Tips & guided tours" switch.
    expect(screen.queryByRole('switch', { name: 'In-App Tips' })).toBeNull()
    expect(screen.queryByRole('switch', { name: 'Guided Tours' })).toBeNull()
    expect(screen.getAllByRole('switch', { name: 'Tips & guided tours' })).toHaveLength(1)
  })

  it('lands the tips and tours deep links on the one tips switch', () => {
    const { container } = renderPage('general')
    const byId = (id: string) => container.ownerDocument.getElementById(settingElementId(id))
    const tips = byId(SETTING_IDS.appearance.tips)
    const tours = byId(SETTING_IDS.appearance.tours)

    expect(tips?.textContent).toContain('Tips & guided tours')
    expect(tours?.contains(tips!)).toBe(true)
  })

  it('reveals an advanced general-setting deep link before focusing its original target', async () => {
    const target = SETTING_IDS.appearance.introSplash
    const { container } = renderPage('general', `/settings?tab=appearance&setting=${target}`)

    expect(container.querySelector('details')!.open).toBe(true)
    await waitFor(() => expect(container.ownerDocument.activeElement?.id).toBe(settingElementId(target)))
  })

  it('preserves specialized subpages and the classic general page', () => {
    const { container, unmount } = renderPage('typography')
    expect(container.querySelector('[data-slot="appearance-overview"]')).toBeNull()
    fireEvent.click(screen.getAllByRole('button', { name: '175%' })[1]!)
    expect($chatTextScale.get()).toBe(175)
    unmount()

    $shellChoice.set(false)
    const classic = renderPage('general')
    expect(classic.container.querySelector('[data-slot="appearance-overview"]')).toBeNull()
    expect(screen.getByRole('switch', { name: 'In-App Tips' })).toBeTruthy()
    expect(screen.queryByRole('radio', { name: 'Light' })).toBeNull()
  })
})
