import { host } from '@hermes/plugin-sdk'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $lastRoster } from '@/plugins/hermes-bots/data'
import { $notifications } from '@/store/notifications'

import { openAgentScreen } from './portal-agent-screen'

const opened = vi.hoisted(() => vi.fn())

vi.mock('@/plugins/hermes-bots/screen-open', () => ({ openBotScreen: opened }))
vi.mock('@/plugins/hermes-bots/screen-hero', () => ({ ScreenHero: () => null }))

beforeEach(() => {
  opened.mockReset()
  vi.spyOn(host, 'profileRoutes').mockResolvedValue([])
})

afterEach(() => {
  vi.restoreAllMocks()
  $lastRoster.set([])
  $notifications.set([])
})

it('opens the agent’s own roster row, not an alias listed ahead of it that serves the same profile', async () => {
  $lastRoster.set([
    { connectionId: 'lab', name: 'nightly', sourceScoped: true, targetProfile: 'ops' },
    { connectionId: 'lab', name: 'ops', sourceScoped: true }
  ] as never)

  expect(await openAgentScreen('lab', 'ops')).toBe(true)

  expect(opened).toHaveBeenCalledTimes(1)
  expect(opened.mock.calls[0][0]).toMatchObject({ connectionId: 'lab', name: 'ops' })
})

it('opens it under the profile the machine serves it from, when the routes name one', async () => {
  vi.spyOn(host, 'profileRoutes').mockResolvedValue([
    { connectionId: 'lab', profile: 'ops', targetProfile: 'ops-main' }
  ] as never)

  await openAgentScreen('lab', 'ops')

  expect(opened.mock.calls[0][0]).toMatchObject({ name: 'ops', targetProfile: 'ops-main' })
})

it('opens nothing and says so when the agent’s route cannot be looked up', async () => {
  vi.spyOn(host, 'profileRoutes').mockRejectedValue(new Error('no route to that agent'))

  expect(await openAgentScreen('lab', 'ops')).toBe(false)

  expect(opened).not.toHaveBeenCalled()
  expect($notifications.get().some(note => note.kind === 'error')).toBe(true)
})
