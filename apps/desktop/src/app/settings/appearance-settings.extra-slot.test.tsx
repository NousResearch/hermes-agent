// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'

import { APPEARANCE_AREAS } from './appearance-contrib'
import { AppearanceSettings } from './appearance-settings'
import { buildSettingsPageSearch, resolveSettingsSubpage } from './subpages'

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

  it('shows the plugin row on the real top-level route (no ?page=), the way index.tsx resolves it', () => {
    act(() => {
      disposers.push(
        registry.register({
          area: APPEARANCE_AREAS.extra,
          id: 'extra-controls-routing',
          render: () => <span>Routing extra controls</span>,
          source: 'disk'
        })
      )
    })

    // The exact wiring from index.tsx: subpage comes from the URL, not a prop.
    const topSubpage = resolveSettingsSubpage('config:appearance', new URLSearchParams(''))
    const { unmount } = renderPage(topSubpage)

    expect(topSubpage).toBeUndefined()
    expect(screen.getByText('Routing extra controls')).toBeTruthy()
    unmount()
    cleanup()

    const deepSubpage = resolveSettingsSubpage('config:appearance', new URLSearchParams({ page: 'theme' }))

    expect(deepSubpage).toBe('theme')
    renderPage(deepSubpage)

    expect(screen.queryByText('Routing extra controls')).toBeNull()
  })

  it('reaches the top-level page through parent-row navigation (no ?page= synthesized)', () => {
    act(() => {
      disposers.push(
        registry.register({
          area: APPEARANCE_AREAS.extra,
          id: 'extra-controls-navigation',
          render: () => <span>Navigation extra controls</span>,
          source: 'disk'
        })
      )
    })

    // The parent Appearance row calls openSettingsPage(view) with no page:
    // it must not synthesize ?page=<first subpage>, so the resolver sees
    // "nothing requested" and the top-level extra slot renders.
    const parentSearch = buildSettingsPageSearch('', 'config:appearance')
    const parentParams = new URLSearchParams(parentSearch)

    expect(parentParams.get('page')).toBeNull()

    const parentSubpage = resolveSettingsSubpage('config:appearance', parentParams)

    expect(parentSubpage).toBeUndefined()

    const { unmount } = renderPage(parentSubpage)

    expect(screen.getByText('Navigation extra controls')).toBeTruthy()
    unmount()
    cleanup()

    // A child row passes its page explicitly and must keep deep-linking.
    const childSearch = buildSettingsPageSearch('', 'config:appearance', 'general')
    const childParams = new URLSearchParams(childSearch)

    expect(childParams.get('page')).toBe('general')
    expect(resolveSettingsSubpage('config:appearance', childParams)).toBe('general')
    renderPage(resolveSettingsSubpage('config:appearance', childParams))

    expect(screen.queryByText('Navigation extra controls')).toBeNull()
  })
})
