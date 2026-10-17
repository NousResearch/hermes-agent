// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'

import { APPEARANCE_AREAS } from './appearance-contrib'
import { AppearanceSettings } from './appearance-settings'

const disposers: Array<() => void> = []

afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
})

function renderPage(subpage?: string) {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <AppearanceSettings subpage={subpage} />
    </QueryClientProvider>
  )
}

describe('AppearanceSettings extra slot', () => {
  it('mounts plugin extras on the default (fallback) subpage only, not on other subpages', () => {
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

    // The router resolves every Appearance visit to a subpage, falling back to
    // the FIRST one (general) — resolveSettingsSubpage ends with
    // `?? pages[0]?.id`. A plain "open Appearance" visit therefore lands on
    // general, so the slot's one guaranteed home is that fallback page; the
    // router never hands this component undefined, and pinning the slot to a
    // later subpage (pet) hid the cards on the default entry.
    let view = renderPage('theme')

    expect(screen.queryByText('Extra controls')).toBeNull()
    view.unmount()

    view = renderPage('pet')

    expect(screen.queryByText('Extra controls')).toBeNull()
    view.unmount()

    view = renderPage('general')

    expect(screen.getByText('Extra controls')).toBeTruthy()
    view.unmount()

    view = renderPage()

    expect(screen.getByText('Extra controls')).toBeTruthy()
  })
})
