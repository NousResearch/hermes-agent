/**
 * Edit Profile's Role field: seeded from the bot, saved through saveBotMeta
 * only when changed, and independent of Title and the avatar.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as DataModule from './data'
import { EditProfileDialog } from './edit-profile-dialog'
import { translateBots } from './i18n-test-helper'
import type { RosterRow } from './types'

const mocks = vi.hoisted(() => ({
  notify: vi.fn(),
  notifyError: vi.fn(),
  requestForBot: vi.fn(async () => ({})),
  saveBotMeta: vi.fn(async () => ({ serverOutcome: 'persisted' }))
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const original = await importOriginal<typeof HermesSdk>()

  return {
    ...original,
    host: { ...original.host, notify: mocks.notify, notifyError: mocks.notifyError },
    usePluginI18n: () => translateBots
  }
})

vi.mock('./data', async importOriginal => {
  const original = await importOriginal<typeof DataModule>()

  return { ...original, saveBotMeta: mocks.saveBotMeta }
})

vi.mock('./routing', async importOriginal => {
  const original = await importOriginal<Record<string, unknown>>()

  return { ...original, requestForBot: mocks.requestForBot }
})

const bot = {
  connectionId: 'local',
  name: 'scout',
  ui_meta: { 'hermes-bots': { role: 'Deep Research', title: 'Scout' } }
} as RosterRow

function open(row: RosterRow = bot) {
  return render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <EditProfileDialog bot={row} onClose={() => undefined} open />
    </QueryClientProvider>
  )
}

beforeEach(() => {
  vi.clearAllMocks()
})

afterEach(() => {
  cleanup()
})

describe('Edit Profile role field', () => {
  it('opens with the bot’s current role', () => {
    open()

    expect((screen.getByPlaceholderText('e.g. Deep Research') as HTMLInputElement).value).toBe('Deep Research')
  })

  it('saves only the role when only the role changed', async () => {
    open()

    fireEvent.change(screen.getByPlaceholderText('e.g. Deep Research'), { target: { value: 'Scouting' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    await waitFor(() => expect(mocks.saveBotMeta).toHaveBeenCalledTimes(1))
    expect(mocks.saveBotMeta).toHaveBeenCalledWith(bot, { role: 'Scouting' })
  })

  it('writes an empty role when it is cleared', async () => {
    open()

    fireEvent.change(screen.getByPlaceholderText('e.g. Deep Research'), { target: { value: '' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    await waitFor(() => expect(mocks.saveBotMeta).toHaveBeenCalledWith(bot, { role: '' }))
  })

  it('does not write at all when nothing changed', async () => {
    open()

    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    await waitFor(() => expect(mocks.notify).toHaveBeenCalledWith({ kind: 'success', message: 'Scout updated' }))
    expect(mocks.saveBotMeta).not.toHaveBeenCalled()
  })
})
