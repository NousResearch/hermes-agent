import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { composerFocusBlockedBySurface } from '@/lib/keybinds/composer-focus-keys'

import { BackworkspacePage } from './back-page'
import { $backworkspaceOpen, toggleBackworkspace } from './store'

// The page stays loading: the textarea is disabled, which is exactly when
// focus used to fall through to <body> and the hidden composer.
vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestGatewayForAgent: () => new Promise(() => {})
}))
vi.mock('./store', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  toggleBackworkspace: vi.fn()
}))

afterEach(() => {
  cleanup()
  $backworkspaceOpen.set(false)
})

describe('BackworkspacePage', () => {
  it('owns the keyboard while turned over and shows the shell again on any unmount', () => {
    $backworkspaceOpen.set(true)

    const { unmount } = render(
      <I18nProvider configClient={null} initialLocale="en">
        <BackworkspacePage />
      </I18nProvider>
    )

    const sheet = screen.getByRole('region', { name: 'Back workspace' })
    const root = sheet.ownerDocument.documentElement

    expect(root.hasAttribute('data-backworkspace')).toBe(true)
    expect(sheet).toBe(sheet.ownerDocument.activeElement)
    expect(composerFocusBlockedBySurface()).toBe(true)

    fireEvent.keyDown(sheet, { key: 'Escape' })
    expect(toggleBackworkspace).toHaveBeenCalledTimes(1)

    // An error boundary replacing the tree unmounts the page the same way.
    unmount()
    expect(root.hasAttribute('data-backworkspace')).toBe(false)
  })
})
