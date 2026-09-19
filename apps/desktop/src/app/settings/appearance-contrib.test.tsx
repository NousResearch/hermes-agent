// @vitest-environment jsdom
import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'

import { APPEARANCE_AREAS, AppearanceExtraSlot } from './appearance-contrib'

const disposers: Array<() => void> = []

afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
})

describe('AppearanceExtraSlot', () => {
  it('renders nothing when empty, then mounts a late registration in place', () => {
    const { container, unmount } = render(<AppearanceExtraSlot />)

    expect(container.firstChild).toBeNull()
    unmount()

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

    render(<AppearanceExtraSlot />)

    expect(screen.getByText('Extra controls')).toBeTruthy()

    act(() => {
      disposers.splice(0).forEach(dispose => dispose())
    })

    expect(screen.queryByText('Extra controls')).toBeNull()
  })

  it('contains a throwing contribution instead of taking the page down', () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)

    act(() => {
      disposers.push(
        registry.register({
          area: APPEARANCE_AREAS.extra,
          id: 'broken-extra',
          render: () => {
            throw new Error('broken appearance contribution')
          },
          source: 'disk'
        })
      )
    })

    render(<AppearanceExtraSlot />)

    // The boundary's inline fallback names the failed contribution.
    expect(screen.getByRole('button', { name: 'broken-extra' })).toBeTruthy()
  })
})
