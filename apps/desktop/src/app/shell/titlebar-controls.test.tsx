import { cleanup, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import { setTitlebarAppActionsSide } from '@/store/titlebar-app-actions'

import { TitlebarControls } from './titlebar-controls'

function mount() {
  return render(
    <MemoryRouter>
      <I18nProvider configClient={null} initialLocale="en">
        <TitlebarControls onOpenSettings={() => undefined} />
      </I18nProvider>
    </MemoryRouter>
  )
}

afterEach(() => {
  setTitlebarAppActionsSide('right')
  cleanup()
})

describe('titlebar app-action cluster', () => {
  it('defaults settings, layout, and HUD to the right so the left titlebar stays free for tabs', () => {
    mount()

    const left = screen.getByLabelText('Window controls')
    const right = screen.getByLabelText('App controls')

    expect(within(right).getByLabelText('Open settings')).toBeTruthy()
    expect(within(right).getByLabelText('Layout editor')).toBeTruthy()
    expect(within(right).getByLabelText('HUD mode')).toBeTruthy()

    expect(within(left).queryByLabelText('Open settings')).toBeNull()
    expect(within(left).queryByLabelText('Layout editor')).toBeNull()
    expect(within(left).queryByLabelText('HUD mode')).toBeNull()
    expect(within(left).getByLabelText(/Hide sidebar|Show sidebar/)).toBeTruthy()
  })

  it('moves settings, layout, and HUD to the left when the appearance setting says left', () => {
    setTitlebarAppActionsSide('left')
    mount()

    const left = screen.getByLabelText('Window controls')
    const right = screen.getByLabelText('App controls')

    expect(within(left).getByLabelText('Open settings')).toBeTruthy()
    expect(within(left).getByLabelText('Layout editor')).toBeTruthy()
    expect(within(left).getByLabelText('HUD mode')).toBeTruthy()
    expect(within(right).queryByLabelText('Open settings')).toBeNull()
  })
})
