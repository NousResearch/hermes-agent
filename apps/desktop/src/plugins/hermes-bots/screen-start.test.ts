/** Starting a bot's screen from outside its pane: the same request the pane makes, and the status
 *  the backend answers with becomes what every surface shows. */
import { beforeEach, expect, it, vi } from 'vitest'

import type * as ScreenConnection from './screen-connection'
import type { RosterRow } from './types'

const display = vi.hoisted(() => ({ request: vi.fn() }))

vi.mock('./screen-pane', () => ({ BotScreenPane: () => null }))
vi.mock('./screen-connection', async original => ({
  ...(await original<typeof ScreenConnection>()),
  displayRequest: display.request
}))

import { startBotScreen } from './screen-open'
import { $screenState, screenStateFor } from './screen-state'

const parker = { connectionId: 'lab', name: 'parker', sourceScoped: true } as RosterRow

beforeEach(() => {
  display.request.mockReset()
  $screenState.set({})
})

it('asks the bot’s backend to start its screen and takes the status it answers with', async () => {
  const running = { installed: true, profile_key: 'k', running: true, supported: true }
  display.request.mockResolvedValue(running)

  await startBotScreen(parker)

  expect(display.request).toHaveBeenCalledWith(parker, 'display.start')
  expect(screenStateFor($screenState.get(), parker)?.status).toEqual(running)
})

it('leaves the status alone and passes on the backend’s reason when it will not start', async () => {
  display.request.mockRejectedValue(new Error('not enough memory'))

  await expect(startBotScreen(parker)).rejects.toThrow('not enough memory')
  expect(screenStateFor($screenState.get(), parker)).toBeNull()
})
