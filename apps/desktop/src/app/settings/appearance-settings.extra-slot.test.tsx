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
