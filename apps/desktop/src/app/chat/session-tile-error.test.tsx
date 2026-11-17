import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { resolveTranslations } from '@/i18n'
import { I18nProvider } from '@/i18n/context'
import { $sessionTiles } from '@/store/session-states'
import { $shellChoice } from '@/store/shell-variant'

import { SessionTilePane } from './session-tile'

afterEach(() => {
  cleanup()
  $sessionTiles.set([])
  $shellChoice.set(null)
})

it('reads a failed tile in fixed English in classic, and wholly in the app language in the redesign', () => {
  $shellChoice.set(false)
  $sessionTiles.set([{ storedSessionId: 'broken', error: 'boom' }])
  render(
    <I18nProvider configClient={null} initialLocale="de">
      <SessionTilePane storedSessionId="broken" />
    </I18nProvider>
  )

  expect(screen.getByText("Couldn't open this session")).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()

  act(() => $shellChoice.set(true))
  const de = resolveTranslations('de', 'redesign')
  expect(screen.getByText(de.desktop.sessionOpenFailed)).toBeTruthy()
  expect(screen.getByRole('button', { name: de.common.retry })).toBeTruthy()
  expect(de.common.retry).not.toBe('Retry')
})
