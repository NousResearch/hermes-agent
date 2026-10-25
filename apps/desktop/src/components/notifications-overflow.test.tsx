// @vitest-environment jsdom
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { AppNotification } from '@/store/notifications'
import { stubResizeObserver } from '@/test/jsdom'

import { NotificationDeck } from './notifications'

const SRC = dirname(fileURLToPath(import.meta.url))

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
  beforeEach(() => {
    stubResizeObserver()
    vi.spyOn(HTMLElement.prototype, 'offsetHeight', 'get').mockReturnValue(96)
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  // The clamp is real CSS in styles.css, not just a class-name string. Parse
  // the actual rule so emptying the rule body or swapping it for a no-op
  // (e.g. `max-height: 100%`) fails here instead of shipping the off-screen
  // defect again — renderToStaticMarkup can never observe any of this.
  it('clamps the collapsed card via a real max-height + scroll rule in styles.css', () => {
    const css = readFileSync(join(SRC, '../styles.css'), 'utf8')
    const rule = css.match(/\.notification-collapsed-clamp\s*\{([^}]*)\}/)

    expect(rule).not.toBeNull()
    expect(rule![1]).toMatch(/max-height\s*:\s*\d/)
    expect(rule![1]).toMatch(/overflow-y\s*:\s*auto/)
    expect(rule![1]).not.toMatch(/max-height\s*:\s*100%/)
  })

  // The class must land on the CardStack surface — the element the clamp
  // constrains, and the element whose height CardStack reserves — not on some
  // wrapper above or below it.
  it('applies the clamp class to the collapsed card surface', () => {
    const { container } = render(
      <NotificationDeck expanded={false} notifications={[notification({ id: 'huge', message: 'x'.repeat(4_000) })]} />
    )

    const surface = container.querySelector('[data-slot="card-stack-surface"]')

    expect(surface).not.toBeNull()
    expect(surface!.className).toContain('notification-collapsed-clamp')
  })

  it('does not clamp when expanded (the stack scrolls instead)', () => {
    const { container } = render(
      <NotificationDeck expanded notifications={[notification({ id: 'huge', message: 'x'.repeat(4_000) })]} />
    )

    const surface = container.querySelector('[data-slot="card-stack-surface"]')

    expect(surface).not.toBeNull()
    expect(surface!.className).not.toContain('notification-collapsed-clamp')
  })
})
