import { host } from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { botsText } from '@/plugins/hermes-bots/i18n'
import type * as ScreenConnection from '@/plugins/hermes-bots/screen-connection'
import { stubMenuDomApis } from '@/test/jsdom'

import type { SimplifiedAgent } from '../use-simplified-agents'

import { AgentScreenRow } from './portal-agent-screen'

const plugin = vi.hoisted(() => ({
  frames: vi.fn(async () => ({ data_url: 'data:image/jpeg;base64,AA==' })),
  open: vi.fn(),
  tone: 'live'
}))

vi.mock('@/plugins/hermes-bots/i18n', async original => (await import('@/test/bots-screen-mocks')).botsI18nMock(original))
vi.mock('@/plugins/hermes-bots/screen-portal', async original =>
  (await import('@/test/bots-screen-mocks')).screenPortalMock(original, plugin)
)
vi.mock('@/plugins/hermes-bots/screen-open', () => ({ openBotScreen: plugin.open }))
vi.mock('@/plugins/hermes-bots/screen-connection', async original => ({
  ...(await original<typeof ScreenConnection>()),
  displayRequest: plugin.frames
}))

const OPS: SimplifiedAgent = { connectionId: 'lab', gateway: 'Lab box', id: 'lab/ops', label: 'ops', profile: 'ops' }

// The preview's tile is a second control named for the Screen: the row is the one in the list.
const row = () =>
  screen.getAllByRole('button', { name: new RegExp(`^${botsText().screen.portalTitle}`) }).find(button => button.closest('.portal-session-row'))!

const preview = () => document.querySelector('.portal-screen-preview')
const wait = (ms: number) => act(() => vi.advanceTimersByTimeAsync(ms))
const rest = () => fireEvent.pointerEnter(row().closest('.portal-session-row')!, { pointerType: 'mouse' })
const leave = () => fireEvent.pointerLeave(row().closest('.portal-session-row')!, { pointerType: 'mouse' })

beforeEach(() => {
  stubMenuDomApis()
  vi.useFakeTimers()
  plugin.frames.mockClear()
  plugin.open.mockReset()
  vi.spyOn(host, 'profileRoutes').mockResolvedValue([])
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.restoreAllMocks()
})

it('shows nothing and asks for no picture until the pointer rests on the row', async () => {
  render(<AgentScreenRow agent={OPS} />)
  await wait(0)

  rest()
  await wait(100)
  leave()
  await wait(2000)

  expect(preview()).toBeNull()
  expect(plugin.frames).not.toHaveBeenCalled()
})

it('shows the screen while the pointer rests on the row, and stops asking once it has gone', async () => {
  render(<AgentScreenRow agent={OPS} />)
  rest()
  await wait(400)

  expect(preview()?.querySelector('img')?.getAttribute('src')).toBe('data:image/jpeg;base64,AA==')
  await wait(4000)
  const asked = plugin.frames.mock.calls.length
  expect(asked).toBeGreaterThan(1)

  leave()
  await wait(400)
  expect(preview()).toBeNull()

  await wait(20_000)
  expect(plugin.frames).toHaveBeenCalledTimes(asked)
})

it('stays while the pointer is on the preview itself', async () => {
  render(<AgentScreenRow agent={OPS} />)
  rest()
  await wait(400)

  leave()
  fireEvent.pointerEnter(preview()!, { pointerType: 'mouse' })
  await wait(1000)

  expect(preview()).not.toBeNull()
})

it('opens for keyboard focus without taking it, and Escape or leaving the row closes it', async () => {
  render(<AgentScreenRow agent={OPS} />)

  act(() => row().focus())
  fireEvent.focus(row())
  await wait(0)
  expect(preview()).not.toBeNull()
  expect(document.activeElement).toBe(row())

  fireEvent.keyDown(row(), { key: 'Escape' })
  await wait(0)
  expect(preview()).toBeNull()

  fireEvent.focus(row())
  await wait(0)
  fireEvent.blur(row())
  await wait(0)
  expect(preview()).toBeNull()
})

it('still opens the Screen from the row, with or without the preview up', async () => {
  vi.useRealTimers()
  render(<AgentScreenRow agent={OPS} />)

  fireEvent.click(row())

  await vi.waitFor(() => expect(plugin.open).toHaveBeenCalledTimes(1))
})
