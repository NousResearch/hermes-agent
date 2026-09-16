import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'

import { ROUTES_AREA } from '../routes'

import { RouteTilePane } from './route-tile'

const disposers: Array<() => void> = []

afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
})

describe('RouteTilePane', () => {
  it('reattaches a missing route when the contribution registers again', () => {
    render(<RouteTilePane path="/reattach-probe" />)

    expect(screen.getByText('no page at /reattach-probe', { exact: true })).toBeTruthy()

    act(() => {
      disposers.push(
        registry.register({
          area: ROUTES_AREA,
          data: { path: '/reattach-probe' },
          id: 'reattach-page',
          render: () => <main data-testid="reattached-page">First page</main>,
          source: 'plugin:reattach-test'
        })
      )
    })

    expect(screen.getByTestId('reattached-page').textContent).toBe('First page')

    act(() => {
      disposers.shift()?.()
    })

    expect(screen.getByText('no page at /reattach-probe', { exact: true })).toBeTruthy()

    act(() => {
      disposers.push(
        registry.register({
          area: ROUTES_AREA,
          data: { path: '/reattach-probe' },
          id: 'reattach-page',
          render: () => <main data-testid="reattached-page">Reloaded page</main>,
          source: 'plugin:reattach-test'
        })
      )
    })

    expect(screen.getByTestId('reattached-page').textContent).toBe('Reloaded page')
  })
})
