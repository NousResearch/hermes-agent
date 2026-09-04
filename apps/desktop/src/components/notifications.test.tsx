import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import { clearNotifications, notify } from '@/store/notifications'
import { setToastSessionOpener } from '@/store/toast-focus'

import { NotificationStack, toastTitleClassName } from './notifications'

const LONG_TITLE = 'This turn is no longer in server history (it may have been compressed away).'
const DETAIL = 'target user message is no longer in session history'

describe('toast session click', () => {
  const opened: string[] = []

  beforeEach(() => {
    opened.length = 0
    clearNotifications()
    setToastSessionOpener(sessionId => opened.push(sessionId))
  })

  afterEach(() => {
    cleanup()
    clearNotifications()
    setToastSessionOpener(null)
  })

  it('clicking a session toast opens the session and dismisses the toast', () => {
    notify({ kind: 'error', message: 'turn failed', sessionId: 'runtime-1' })

    render(
      <I18nProvider configClient={null} initialLocale="en">
        <NotificationStack />
      </I18nProvider>
    )

    fireEvent.click(screen.getByText('turn failed'))

    expect(opened).toEqual(['runtime-1'])
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('the dismiss button does not navigate', () => {
    notify({ kind: 'error', message: 'turn failed', sessionId: 'runtime-1' })

    render(
      <I18nProvider configClient={null} initialLocale="en">
        <NotificationStack />
      </I18nProvider>
    )

    fireEvent.click(screen.getByRole('button', { name: 'Dismiss notification' }))

    expect(opened).toEqual([])
    expect(screen.queryByRole('alert')).toBeNull()
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
