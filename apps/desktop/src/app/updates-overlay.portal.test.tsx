import { act, cleanup, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, expect, it } from 'vitest'

import type { DesktopUpdateCommit } from '@/global'
import { I18nProvider } from '@/i18n/context'
import { en } from '@/i18n/en'
import { $shellChoice } from '@/store/shell-variant'
import {
  $updateApply,
  $updateOverlayOpen,
  $updateOverlayTarget,
  $updateStatus,
  resetUpdateApplyState
} from '@/store/updates'

import { UpdatesOverlay } from './updates-overlay'

const commits = ['feat: clearer names', 'fix: faster start'].map((summary, index): DesktopUpdateCommit => ({
  sha: `${index}abc`,
  summary,
  author: 'a',
  at: 1
}))

beforeEach((): void => {
  $shellChoice.set(true)
})

afterEach((): void => {
  cleanup()
  $shellChoice.set(false)
  $updateOverlayOpen.set(false)
  $updateStatus.set(null)
  resetUpdateApplyState()
})

async function open(): Promise<void> {
  $updateStatus.set({ behind: 2, commits, mechanism: 'posix-handoff', supported: true, targetSha: 'f00' })
  $updateOverlayTarget.set('client')
  $updateOverlayOpen.set(true)
  await act(async (): Promise<void> => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <UpdatesOverlay />
      </I18nProvider>
    )
  })
}

it('pitches an update in the redesign as the changes it really brings, numbered, with Update now and Maybe later', async () => {
  await open()

  const dialog = screen.getByRole('dialog')
  const rows = within(dialog).getAllByRole('listitem')
  expect(rows.length).toBeGreaterThan(0)
  expect(rows[0].textContent).toContain('01')
  expect(within(dialog).getByRole('button', { name: en.updates.updateNow })).toBeTruthy()
  expect(within(dialog).getByRole('button', { name: en.updates.maybeLater })).toBeTruthy()
})

it('shows each updater stage the install has reached, the newest under way', async () => {
  $updateApply.set({
    applying: true,
    stage: 'pull',
    message: 'Pulling',
    percent: 40,
    error: null,
    command: null,
    log: [
      { stage: 'fetch', message: 'Fetched', at: 1 },
      { stage: 'pull', message: 'Pulling', at: 2 }
    ]
  })
  await open()

  const rows = within(screen.getByRole('dialog')).getAllByRole('listitem')
  expect(rows.map(row => row.textContent)).toEqual([
    `${en.updates.stages.fetch}Fetched`,
    `${en.updates.stages.pull}Pulling`
  ])
  expect(rows[1].hasAttribute('data-current')).toBe(true)
  expect(rows[0].hasAttribute('data-current')).toBe(false)
})
