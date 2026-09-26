import { renderToStaticMarkup } from 'react-dom/server'

import { describe, expect, it } from 'vitest'

import { NotificationDeck } from './notifications'
import type { AppNotification } from '@/store/notifications'

const notification = (over: Partial<AppNotification>): AppNotification =>
  ({
    detail: '',
    id: 'n1',
    kind: 'warning',
    message: 'hello',
    placement: 'default',
    title: 'Title',
    ...over
  }) as AppNotification

describe('NotificationDeck overflow guard', () => {
  it('collapses an oversized collapsed-card message so controls stay reachable', () => {
    // The model-switch confirmation (~800 chars) previously blew the collapsed
    // card past the viewport bottom: its Confirm/Dismiss buttons rendered off
    // screen with no way to scroll the toast itself.
    const huge = 'x'.repeat(4_000)

    const html = renderToStaticMarkup(
      <NotificationDeck
        expanded={false}
        notifications={[notification({ id: 'huge', message: huge })]}
      />
    )

    expect(html).toContain('notification-collapsed-clamp')
  })

  it('does not clamp when expanded (the stack scrolls instead)', () => {
    const huge = 'x'.repeat(4_000)

    const html = renderToStaticMarkup(
      <NotificationDeck
        expanded
        notifications={[notification({ id: 'huge', message: huge })]}
      />
    )

    expect(html).not.toContain('notification-collapsed-clamp')
  })

  it('keeps the clamp on every collapsed card (harmless for short messages)', () => {
    // The clamp lives on the collapsed card surface itself, so short messages
    // carry it too — it only takes effect past 48dvh.
    const html = renderToStaticMarkup(
      <NotificationDeck
        expanded={false}
        notifications={[notification({ id: 'small', message: 'short' })]}
      />
    )

    expect(html).toContain('notification-collapsed-clamp')
  })
})
