import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type * as ScreenConnection from '@/plugins/hermes-bots/screen-connection'

import { AgentScreenThumbnail } from './portal-agent-screen'

const plugin = vi.hoisted(() => ({
  frames: vi.fn(async () => ({ data_url: 'data:image/jpeg;base64,AA==' })),
  open: vi.fn(),
  route: null as unknown,
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

const onOpened = vi.fn()
const mount = () => render(<AgentScreenThumbnail connectionId="local" onOpened={onOpened} profile="default" />)
const flush = () => act(() => vi.advanceTimersByTimeAsync(0))

beforeEach(() => {
  vi.useFakeTimers()
  plugin.frames.mockClear()
  plugin.open.mockReset()
  onOpened.mockReset()
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

it.each(['live', 'human', 'other'])('shows the agent’s screen while it is running (%s)', async tone => {
  plugin.tone = tone
  const { container } = mount()
  await flush()

  expect(container.querySelector('img')?.getAttribute('src')).toBe('data:image/jpeg;base64,AA==')
})

it.each(['off', 'missing', 'unknown', 'unsupported', 'unavailable'])('shows nothing and asks for no frame while it is %s', async tone => {
  plugin.tone = tone
  const { container } = mount()
  await flush()

  expect(container.firstChild).toBeNull()
  expect(plugin.frames).not.toHaveBeenCalled()
})

it('stops asking for frames once it is gone', async () => {
  plugin.tone = 'live'
  const view = mount()
  await flush()
  await act(() => vi.advanceTimersByTimeAsync(4000))
  const asked = plugin.frames.mock.calls.length
  expect(asked).toBeGreaterThan(1)

  view.unmount()
  await act(() => vi.advanceTimersByTimeAsync(20_000))

  expect(plugin.frames).toHaveBeenCalledTimes(asked)
})

it('opens the Screen and lets the page step aside for it', async () => {
  plugin.tone = 'live'
  render(<AgentScreenThumbnail connectionId="lab" onOpened={onOpened} profile="ops" />)
  await flush()

  fireEvent.click(screen.getByRole('button'))

  // The tile opens the row it was given, with no route lookup: it must be this agent on its own machine.
  expect(plugin.open).toHaveBeenCalledTimes(1)
  expect(plugin.open.mock.calls[0][0]).toMatchObject({ connectionId: 'lab', name: 'ops' })
  expect(onOpened).toHaveBeenCalledTimes(1)
})
