import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $notifications, clearNotifications, notify, notifyError } from '@/store/notifications'
import { $poolLimitsSettingsRequest } from '@/store/pool-limits'

import { NotificationStack, toastTitleClassName } from './notifications'

const LONG_TITLE = 'This turn is no longer in server history (it may have been compressed away).'
const DETAIL = 'target user message is no longer in session history'

describe('toast titles', () => {
  beforeEach(() => {
    clearNotifications()
    $poolLimitsSettingsRequest.set(0)
  })

  afterEach(() => {
    cleanup()
    clearNotifications()
    $poolLimitsSettingsRequest.set(0)
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

  it('makes a local pool-slot timeout actionable without changing ordinary errors', () => {
    notifyError(
      new Error(
        `Error invoking remote method 'hermes:connection': Error: Local backend start for "research" timed out while waiting for a free slot.`
      ),
      'Failed to switch to profile "research"'
    )

    render(
      <I18nProvider configClient={null} initialLocale="en">
        <NotificationStack />
      </I18nProvider>
    )

    expect(screen.getByText(/All local profile backend slots are busy/)).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Open Advanced Settings' }))

    expect($poolLimitsSettingsRequest.get()).toBe(1)
    expect($notifications.get()).toHaveLength(0)

    notifyError(new Error('gateway unavailable'), 'Failed to switch profile')
    expect($notifications.get()[0]?.action).toBeUndefined()
  })

  it('keeps background pool-slot timeouts quiet if they reach the renderer', () => {
    notifyError(
      new Error('Local backend start for "background" timed out while waiting for a free slot. (background)'),
      'Background profile warm-up failed'
    )

    expect($notifications.get()[0]?.action).toBeUndefined()
  })
})
