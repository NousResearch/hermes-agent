import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'

import { RouteTilePane } from './route-tile'

// #109063: route tiles subscribe to ROUTES_AREA but used to derive routes via
// an independently-called `contributedRoutes()`, which React Compiler can keep
// memoized across the delivered subscription update. These tests run through
// the real registry, the real `useContributions` hook and the compiler-enabled
// Vite config — mocking the route list would hide exactly this defect.

afterEach(cleanup)

/** Register (or hot-replace) the plugin route page mounted at `/resetwatch`. */
function registerResetwatch(label: string) {
  return registry.register({
    id: 'resetwatch',
    area: 'routes',
    title: `Resetwatch ${label}`,
    data: { path: '/resetwatch' },
    render: () => <div>{`PLUGIN-PAGE-${label}`}</div>
  })
}

describe('RouteTilePane route-contribution reactivity (#109063)', () => {
  it('mounts a plugin page whose route registers after the pane', () => {
    render(<RouteTilePane path="/resetwatch" />)
    expect(screen.getByText(/no page at \/resetwatch/)).toBeTruthy()

    let dispose = () => {}
    act(() => {
      dispose = registerResetwatch('A')
    })
    expect(screen.getByText('PLUGIN-PAGE-A')).toBeTruthy()

    act(() => {
      dispose()
    })
  })

  it('hot-replaces the page when the same contribution id re-registers', () => {
    let dispose = registerResetwatch('A')
    render(<RouteTilePane path="/resetwatch" />)
    expect(screen.getByText('PLUGIN-PAGE-A')).toBeTruthy()

    act(() => {
      dispose()
      dispose = registerResetwatch('B')
    })
    expect(screen.getByText('PLUGIN-PAGE-B')).toBeTruthy()
    expect(screen.queryByText('PLUGIN-PAGE-A')).toBeNull()

    act(() => {
      dispose()
    })
  })

  it('returns to the fallback after dispose and mounts again on re-register', () => {
    let dispose = registerResetwatch('A')
    render(<RouteTilePane path="/resetwatch" />)

    act(() => {
      dispose()
    })
    expect(screen.getByText(/no page at \/resetwatch/)).toBeTruthy()

    act(() => {
      dispose = registerResetwatch('A')
    })
    expect(screen.getByText('PLUGIN-PAGE-A')).toBeTruthy()

    act(() => {
      dispose()
    })
  })
})
