import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $notifications, clearNotifications, notify, notifyError } from '@/store/notifications'
import { $poolLimitsSettingsRequest } from '@/store/pool-limits'
import { stubResizeObserver } from '@/test/jsdom'

import { NotificationStack } from './notifications'

beforeAll(stubResizeObserver)

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

  it.each(['default', 'bottom-right'] as const)(
    'caps the %s toast stack at one back edge and keeps older notifications reachable',
    async placement => {
      for (let index = 0; index < 7; index++) {
        notify({ id: `notice-${index}`, message: `Notice ${index}`, placement, durationMs: 0 })
      }

      render(<NotificationStack />)
      expect(screen.getAllByRole('status')).toHaveLength(1)
      expect(document.querySelectorAll('[data-slot="card-stack-edge"]')).toHaveLength(1)
      fireEvent.click(screen.getByRole('button', { name: /Show.*6/ }))
      expect(screen.getByText('Notice 0')).toBeTruthy()
      expect(screen.getAllByRole('status')).toHaveLength(7)
      fireEvent.click(screen.getAllByRole('button', { name: /Dismiss/ })[0])
      await waitFor(() => expect(screen.queryByText('Notice 6')).toBeNull())
    }
  )

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
