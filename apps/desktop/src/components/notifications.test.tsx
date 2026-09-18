import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import { setInAppToastCorner } from '@/store/in-app-toast-corner'
import { $notifications, clearNotifications, notify, notifyError } from '@/store/notifications'
import { $poolLimitsSettingsRequest } from '@/store/pool-limits'
import { stubResizeObserver } from '@/test/jsdom'

import { ambientStackClassName, NotificationStack, toastTitleClassName } from './notifications'

const LONG_TITLE = 'This turn is no longer in server history (it may have been compressed away).'
const DETAIL = 'target user message is no longer in session history'

beforeAll(stubResizeObserver)

describe('toast titles', () => {
  beforeEach(() => {
    clearNotifications()
    setInAppToastCorner('bottom-right')
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

  it.each(['default', 'bottom-right'] as const)(
    'caps the %s toast stack at one back edge and keeps older notifications reachable',
    async placement => {
      for (let index = 0; index < 7; index++) {
        notify({ id: `notice-${index}`, message: `Notice ${index}`, placement, durationMs: 0 })
      }

      render(<NotificationStack />)
      expect(screen.getAllByRole('status')).toHaveLength(1)
      const edgeButton = screen.getByRole('button', { name: /Show.*6/ })
      const stack = edgeButton.closest('[data-slot="card-stack"]')
      expect(stack?.querySelector('[data-slot="card-stack-edge"]')).not.toBeNull()
      fireEvent.click(edgeButton)
      expect(screen.getByText('Notice 0')).toBeTruthy()
      expect(screen.getAllByRole('status')).toHaveLength(7)
      fireEvent.click(screen.getAllByRole('button', { name: /Dismiss/ })[0])
      await waitFor(() => expect(screen.queryByText('Notice 6')).toBeNull())
    }
  )

  it.each([
    ['top-left', 'ml-4', 'self-start'],
    ['top-right', 'mr-4', 'self-end'],
    ['bottom-left', 'left-4', 'bottom-4'],
    ['bottom-right', 'right-4', 'bottom-4']
  ] as const)('places routine toasts in the selected %s corner', (corner, horizontal, vertical) => {
    setInAppToastCorner(corner)
    notify({ id: `notice-${corner}`, message: corner, placement: 'bottom-right', durationMs: 0 })

    render(<NotificationStack />)

    const region = screen.getByText(corner).closest('[role="region"]')
    expect(region?.className).toContain(horizontal)
    expect(region?.className).toContain(vertical)
    expect(ambientStackClassName(corner)).toContain(horizontal)
  })

  it('flows top-corner routine toasts below urgent notices in the same vertical lane', () => {
    setInAppToastCorner('top-right')
    notify({ id: 'urgent', kind: 'error', message: 'Urgent notice', durationMs: 0 })
    notify({ id: 'routine', message: 'Routine notice', placement: 'bottom-right', durationMs: 0 })

    render(<NotificationStack />)

    const urgentRegion = screen.getByText('Urgent notice').closest('[role="region"]')
    const routineRegion = screen.getByText('Routine notice').closest('[role="region"]')
    const lane = urgentRegion?.parentElement

    expect(lane?.getAttribute('data-slot')).toBe('top-notification-lane')
    expect(routineRegion?.parentElement).toBe(lane)
    expect(urgentRegion?.nextElementSibling).toBe(routineRegion)
    expect(routineRegion?.className).not.toContain('fixed')
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

    expect(screen.getByText(/Too many bots are running at once/)).toBeTruthy()

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
