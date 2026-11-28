import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it } from 'vitest'

import { RootTooltipProvider } from '@/components/ui/tooltip'
import { $commandPaletteOpen, $commandPalettePage, $commandPaletteSeed } from '@/store/command-palette'
import { $shellChoice } from '@/store/shell-variant'

import { SettingsView } from './index'

afterEach(() => {
  cleanup()
  $commandPaletteOpen.set(false)
  $commandPalettePage.set(null)
  $commandPaletteSeed.set('')
  $shellChoice.set(null)
})

describe.each([false, true])('Settings type-to-search, redesign %s', redesign => {
  it('opens the settings palette seeded with the typed character', () => {
    $shellChoice.set(redesign)

    render(
      <MemoryRouter initialEntries={['/settings?tab=about']}>
        <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
          <RootTooltipProvider>
            <SettingsView onClose={() => undefined} />
          </RootTooltipProvider>
        </QueryClientProvider>
      </MemoryRouter>
    )

    fireEvent.keyDown(window, { key: 'm' })

    expect($commandPaletteOpen.get()).toBe(true)
    expect($commandPalettePage.get()).toBe('settings')
    expect($commandPaletteSeed.get()).toBe('m')
  })

  it('leaves a printable key to a dialog opened over settings', () => {
    $shellChoice.set(redesign)

    render(
      <MemoryRouter initialEntries={['/settings?tab=about']}>
        <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
          <RootTooltipProvider>
            <SettingsView onClose={() => undefined} />
          </RootTooltipProvider>
        </QueryClientProvider>
      </MemoryRouter>
    )

    // Radix's Dialog.Content: role="dialog" + aria-labelledby, no aria-modal attribute.
    const modal = document.createElement('div')
    modal.setAttribute('aria-labelledby', 'test-modal-title')
    modal.setAttribute('role', 'dialog')
    const title = document.createElement('h2')
    title.id = 'test-modal-title'
    title.textContent = 'Pick a model'
    modal.append(title)
    document.body.append(modal)

    try {
      fireEvent.keyDown(window, { key: 'm' })

      expect($commandPaletteOpen.get()).toBe(false)
      expect($commandPaletteSeed.get()).toBe('')
    } finally {
      modal.remove()
    }
  })
})
