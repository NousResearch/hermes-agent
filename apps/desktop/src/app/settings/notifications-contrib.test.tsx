// @vitest-environment jsdom
import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'

import { NOTIFICATIONS_AREAS, NotificationsExtraSlot } from './notifications-contrib'
import { NotificationsSettings } from './notifications-settings'

const disposers: Array<() => void> = []

afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
})

describe('NotificationsExtraSlot', () => {
  it('renders nothing when empty, then mounts a registration in place', () => {
    const { container, unmount } = render(<NotificationsExtraSlot />)

    expect(container.firstChild).toBeNull()
    unmount()

    act(() => {
      disposers.push(
        registry.register({
          area: NOTIFICATIONS_AREAS.extra,
          id: 'plugin-row',
          render: () => <span>Plugin alerts row</span>,
          source: 'disk'
        })
      )
    })

    render(<NotificationsExtraSlot />)

    expect(screen.getByText('Plugin alerts row')).toBeTruthy()

    act(() => {
      disposers.splice(0).forEach(dispose => dispose())
    })

    expect(screen.queryByText('Plugin alerts row')).toBeNull()
  })

  it('contains a throwing contribution instead of taking the page down', () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)

    act(() => {
      disposers.push(
        registry.register({
          area: NOTIFICATIONS_AREAS.extra,
          id: 'broken-row',
          render: () => {
            throw new Error('broken notifications contribution')
          },
          source: 'disk'
        })
      )
    })

    render(<NotificationsExtraSlot />)

    expect(screen.getByText('“broken-row” failed to render')).toBeTruthy()
    expect(screen.getByRole('button', { name: /retry/i })).toBeTruthy()
  })

  it('is mounted on the Alerts page and not on Sounds', () => {
    act(() => {
      disposers.push(
        registry.register({
          area: NOTIFICATIONS_AREAS.extra,
          id: 'page-row',
          render: () => <span>Plugin alerts row</span>,
          source: 'disk'
        })
      )
    })

    const { unmount } = render(<NotificationsSettings subpage="sounds" />)

    expect(screen.queryByText('Plugin alerts row')).toBeNull()
    unmount()

    render(<NotificationsSettings subpage="alerts" />)

    expect(screen.getByText('Plugin alerts row')).toBeTruthy()
  })
})
