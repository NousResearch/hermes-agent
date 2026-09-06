import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import { showAgentNotice } from '@/store/agent-notices'
import { $notifications, clearNotifications, notify } from '@/store/notifications'

import { NotificationStack, toastTitleClassName } from './notifications'

const LONG_TITLE = 'This turn is no longer in server history (it may have been compressed away).'
const DETAIL = 'target user message is no longer in session history'

describe('persistent notice stacks', () => {
  afterEach(() => {
    cleanup()
    clearNotifications()
  })

  it.each(['warn', 'info'])('keeps all %s notices reachable in a bounded stack', level => {
    clearNotifications()

    for (let index = 0; index < 8; index += 1) {
      showAgentNotice({ key: `persistent-${index}`, kind: 'sticky', level, text: `Persistent notice ${index}` })
    }

    render(
      <I18nProvider configClient={null} initialLocale="en">
        <NotificationStack />
      </I18nProvider>
    )

    if (level === 'warn') {
      fireEvent.click(screen.getByRole('button', { name: /^Show/ }))
    }

    expect(screen.getAllByRole('status')).toHaveLength(8)
    expect(screen.getByRole('region').className).toContain('overflow-y-auto')
    expect(screen.getByRole('region').className).toContain('max-h-[calc(100dvh-')

    if (level === 'warn') {
      fireEvent.click(screen.getByRole('button', { name: 'Clear all' }))
      expect($notifications.get()).toEqual([])
    }
  })
})

describe('toast titles', () => {
  beforeEach(() => {
    clearNotifications()
  })

  afterEach(() => {
    cleanup()
    clearNotifications()
  })

  it('drops the one-line clamp so a long error title can wrap', () => {
    const className = toastTitleClassName()

    expect(className).toMatch(/\bline-clamp-none\b/)
    expect(className).not.toMatch(/\bline-clamp-1\b/)
    expect(className).toMatch(/\bwhitespace-normal\b/)
    expect(className).toContain('max-h-[4.5em]')
    expect(className).toMatch(/\boverflow-y-auto\b/)
  })

  it('renders the full title and body instead of truncating them', () => {
    notify({ kind: 'error', title: LONG_TITLE, message: DETAIL })

    render(
      <I18nProvider configClient={null} initialLocale="en">
        <NotificationStack />
      </I18nProvider>
    )

    const title = screen.getByText(LONG_TITLE)

    expect(title.textContent).toBe(LONG_TITLE)
    expect(title.getAttribute('title')).toBe(LONG_TITLE)
    expect(title.className).toMatch(/\bline-clamp-none\b/)
    expect(title.className).not.toMatch(/\bline-clamp-1\b/)
    expect(title.className).toMatch(/\boverflow-y-auto\b/)
    expect(screen.getByText(DETAIL)).toBeTruthy()
  })
})
