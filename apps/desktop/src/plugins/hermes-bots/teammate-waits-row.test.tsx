/**
 * The strip renders in the composer's `composer.top` slot, which sits inside the composer's
 * <form>: its Open and Dismiss buttons must never act as Send for whatever draft is typed.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { translateBotsIn } from './i18n-test-helper'
import { DeliveryRow, type TeammateDelivery } from './teammate-waits'
import type { RosterRow } from './types'

const { openRosterBot } = vi.hoisted(() => ({ openRosterBot: vi.fn(async () => true) }))

vi.mock('@hermes/plugin-sdk', async importOriginal => ({
  ...(await importOriginal<typeof HermesSdk>()),
  usePluginI18n: () => translateBotsIn('en')
}))

vi.mock('./roster-actions', () => ({ openRosterBot }))

const en = translateBotsIn('en')

afterEach(cleanup)

it('Open and Dismiss inside the composer form never submit the draft', () => {
  const sender = { name: 'alpha' } as RosterRow
  const scribe = { name: 'scribe' } as RosterRow

  const row: TeammateDelivery = {
    delivery_id: 'd1',
    target: 'scribe',
    target_handle: 'scribe',
    state: 'failed',
    since: 0,
    preview: '',
    reason: 'target_error'
  }

  const submit = vi.fn((event: Event) => event.preventDefault())

  render(
    <form onSubmit={event => submit(event.nativeEvent)}>
      <textarea defaultValue="a draft the user has not sent" />
      <ul>
        <DeliveryRow now={0} roster={[sender, scribe]} row={row} sender={sender} />
      </ul>
    </form>
  )

  fireEvent.click(screen.getByText(en('waiting.open')))
  fireEvent.click(screen.getByLabelText(en('waiting.dismiss')))

  expect(openRosterBot).toHaveBeenCalledTimes(1)
  expect(submit).not.toHaveBeenCalled()
})
