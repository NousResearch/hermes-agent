import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, expect, it, vi } from 'vitest'

import { RootTooltipProvider } from '@/components/ui/tooltip'
import { en } from '@/i18n/en'
import { Settings } from '@/lib/icons'

import type { OverlayNavGroup } from '../../overlays/overlay-split-layout'
import { redesignSettingsLabels, settingsNavigationPresentation } from '../../settings/redesign-navigation'
import { redesignNavigationCopy } from '../../settings/redesign-navigation-copy'

import { SimplifiedSettingsMenu } from './simplified-settings-menu'

afterEach(cleanup)

beforeAll(() => {
  Element.prototype.hasPointerCapture ??= () => false
  Element.prototype.releasePointerCapture ??= () => undefined
  Element.prototype.setPointerCapture ??= () => undefined
  HTMLElement.prototype.scrollIntoView ??= () => undefined
})

// The classic destinations the Settings page builds its column from (labels as classic names them).
const PAGE_GROUPS = [
  'config:appearance',
  'config:advanced',
  'notifications',
  'billing',
  'providers',
  'gateway',
  'web-access',
  'keybinds',
  'keys',
  'sessions',
  'plugins',
  'about'
].map((id): OverlayNavGroup => ({ active: false, icon: Settings, id, label: id, onSelect: vi.fn() }))

it('lists the Settings page destinations, in its order and under its names', () => {
  const navigate = vi.fn()
  render(
    <MemoryRouter>
      <RootTooltipProvider>
        <SimplifiedSettingsMenu active={false} onNavigate={navigate} />
      </RootTooltipProvider>
    </MemoryRouter>
  )

  const trigger = screen.getByRole('button', { name: en.commandCenter.settings })
  fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
  fireEvent.click(trigger)

  const page = settingsNavigationPresentation(
    PAGE_GROUPS,
    true,
    'billing',
    undefined,
    redesignSettingsLabels(en, redesignNavigationCopy('en'))
  ).groups.map(group => group.label)

  const menu = screen.getAllByRole('menuitem').map(item => item.textContent?.trim())
  expect(menu.slice(0, page.length)).toEqual(page)

  fireEvent.click(screen.getByRole('menuitem', { name: page[0] }))
  expect(navigate).toHaveBeenCalledWith(expect.objectContaining({ route: '/settings?tab=billing' }))
})
