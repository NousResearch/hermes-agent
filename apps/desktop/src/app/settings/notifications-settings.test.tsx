import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $inAppToastCorner, setInAppToastCorner } from '@/store/in-app-toast-corner'

import { NotificationsSettings } from './notifications-settings'

const STORAGE_KEY = 'hermes.desktop.inAppToastCorner'

describe('NotificationsSettings', () => {
  beforeEach(() => {
    window.localStorage.clear()
    setInAppToastCorner('bottom-right')
  })

  afterEach(cleanup)

  it('lets the user move routine in-app toasts to another corner', () => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <NotificationsSettings />
      </I18nProvider>
    )

    fireEvent.click(screen.getByRole('combobox', { name: 'In-app toast position' }))
    fireEvent.click(screen.getByRole('option', { name: 'Top right' }))

    expect($inAppToastCorner.get()).toBe('top-right')
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe('top-right')
  })
})
