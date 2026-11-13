import { act, cleanup, render } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, expect, it } from 'vitest'

import { PRIMARY_SESSION_VIEW, SessionViewProvider } from '@/app/chat/session-view'
import { I18nProvider } from '@/i18n'
import { setSessionCompacting } from '@/store/compaction'

import { CompactingNotice } from './compacting-notice'

afterEach(() => {
  cleanup()
  setSessionCompacting('owner', false)
})

function renderNotice(ownerBusy: boolean) {
  const $ownerBusy = atom(ownerBusy)
  const view = { ...PRIMARY_SESSION_VIEW, $runtimeId: atom<string | null>('owner'), $busy: $ownerBusy }

  return {
    $ownerBusy,
    ...render(
      <I18nProvider configClient={null} initialLocale="en">
        <SessionViewProvider value={view}>
          <CompactingNotice />
        </SessionViewProvider>
      </I18nProvider>
    )
  }
}

it('warns that sends will queue while a standalone compression runs', () => {
  setSessionCompacting('owner', true)

  const { container } = renderNotice(false)

  const notice = container.querySelector('[data-slot="composer-compacting"]')
  expect(notice).toBeTruthy()
  expect(notice?.textContent).toContain('Compacting context — sent messages will queue')
})

it('stays out of the way mid-turn and disappears when the compaction ends', () => {
  setSessionCompacting('owner', true)

  const { container, $ownerBusy } = renderNotice(true)
  expect(container.querySelector('[data-slot="composer-compacting"]')).toBeNull()

  act(() => $ownerBusy.set(false))
  expect(container.querySelector('[data-slot="composer-compacting"]')).toBeTruthy()

  act(() => setSessionCompacting('owner', false))
  expect(container.querySelector('[data-slot="composer-compacting"]')).toBeNull()
})
