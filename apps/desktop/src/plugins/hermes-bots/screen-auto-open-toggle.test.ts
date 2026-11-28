/**
 * The "open Screen when the bot uses it" switch: it flips what is stored for that bot (the same
 * BotMeta field auto-raise reads, so a setting made in either shell is the other's too) and says
 * which way it went.
 */
import { beforeEach, expect, it, vi } from 'vitest'

import type { BotMeta, RosterRow } from './types'

const calls = vi.hoisted(() => ({
  notify: vi.fn(),
  notifyError: vi.fn(),
  save: vi.fn(),
  stored: {} as BotMeta,
  fail: false
}))

vi.mock('@hermes/plugin-sdk', async original => ({
  ...(await original<object>()),
  host: { notify: calls.notify, notifyError: calls.notifyError }
}))
vi.mock('./screen-pane', () => ({ BotScreenPane: () => null }))
vi.mock('./screen-autoraise', () => ({ noteScreenTabClosed: () => {}, noteScreenTabOpened: () => {} }))
vi.mock('./data', () => ({ botSelectionKey: (bot: RosterRow) => bot.name, saveBotMeta: calls.save }))
vi.mock('./canonical-chat', () => ({
  ensureBotMetadata: async () => {
    if (calls.fail) {
      throw new Error('offline')
    }

    return calls.stored
  }
}))

import { botsText } from './i18n'
import { displayName } from './labels'
import { toggleBotScreenAutoOpen } from './screen-open'

const parker = { connectionId: 'lab', name: 'parker' } as RosterRow

beforeEach(() => {
  calls.notify.mockReset()
  calls.notifyError.mockReset()
  calls.save.mockReset()
  calls.stored = {}
  calls.fail = false
})

it('turns it on for a bot that has it off, and off again from what is stored', async () => {
  await toggleBotScreenAutoOpen(parker)
  expect(calls.save).toHaveBeenLastCalledWith(parker, { screenAutoOpen: true })
  expect(calls.notify).toHaveBeenLastCalledWith({ kind: 'info', message: botsText().screen.autoOpenOnToast(displayName(parker, null)) })

  calls.stored = { screenAutoOpen: true }
  await toggleBotScreenAutoOpen(parker)
  expect(calls.save).toHaveBeenLastCalledWith(parker, { screenAutoOpen: false })
  expect(calls.notify).toHaveBeenLastCalledWith({ kind: 'info', message: botsText().screen.autoOpenOffToast(displayName(parker, null)) })
})

it('changes nothing when the bot’s stored settings cannot be read, and says so', async () => {
  calls.fail = true

  await toggleBotScreenAutoOpen(parker)

  expect(calls.save).not.toHaveBeenCalled()
  expect(calls.notifyError).toHaveBeenCalledTimes(1)
})
