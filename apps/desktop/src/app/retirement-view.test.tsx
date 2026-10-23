import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

import { Dialog, DialogContent } from '@/components/ui/dialog'
import type { DesktopUpdateStatus } from '@/global'
import { I18nProvider } from '@/i18n'
import { en } from '@/i18n/en'
import { $updateStatus, resetUpdateApplyState } from '@/store/updates'

import { DiscontinuedNotice } from './retirement-view'

afterEach((): void => {
  cleanup()
  resetUpdateApplyState()
  $updateStatus.set(null)
  Reflect.deleteProperty(window, 'hermesDesktop')
})

test('discontinued retirement shows the uninstall notice and persists dismissal per revision', async (): Promise<void> => {
  const dismissed: string[] = []
  const stored = new Map<string, string>()

  const original = {
    getItem: window.localStorage.getItem.bind(window.localStorage),
    setItem: window.localStorage.setItem.bind(window.localStorage)
  }

  vi.spyOn(window.localStorage, 'getItem').mockImplementation((key: string) => stored.get(key) ?? original.getItem(key))
  vi.spyOn(window.localStorage, 'setItem').mockImplementation((key: string, value: string) => {
    stored.set(key, value)
  })

  const retirement: NonNullable<DesktopUpdateStatus['retirement']> = {
    state: 'discontinued',
    destination: 'stable',
    version: '1.0.0'
  }

  render(
    <I18nProvider configClient={null} initialLocale="en">
      <Dialog open>
        <DialogContent>
          <DiscontinuedNotice
            onDismiss={(): void => {
              dismissed.push('dismissed')
            }}
            retirement={retirement}
          />
        </DialogContent>
      </Dialog>
    </I18nProvider>
  )

  // The notice carries the "no longer supported — uninstall" copy, and offers
  // no download or install action — only the dismissal.
  expect(screen.getByText(en.updates.discontinuedTitle)).toBeTruthy()
  expect(screen.getByText(en.updates.discontinuedBody)).toBeTruthy()
  expect(screen.queryByRole('button', { name: en.updates.updateNow })).toBeNull()
  expect(screen.queryByRole('checkbox')).toBeNull()

  fireEvent.click(screen.getByRole('button', { name: en.updates.maybeLater }))
  await waitFor((): void => {
    expect(dismissed).toEqual(['dismissed'])
  })
})
