import { host } from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type * as ScreenConnection from '@/plugins/hermes-bots/screen-connection'
import type { RosterRow } from '@/plugins/hermes-bots/types'

import { AgentScreenThumbnail } from './portal-agent-screen'

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

const OPS = { connectionId: 'lab', name: 'ops', sourceScoped: true } as RosterRow
const onOpened = vi.fn()
const mount = () => render(<AgentScreenThumbnail bot={OPS} onOpened={onOpened} />)
const flush = () => act(() => vi.advanceTimersByTimeAsync(0))

beforeEach(() => {
  vi.useFakeTimers()
  plugin.tone = 'live'
  plugin.frames.mockClear()
  plugin.open.mockReset()
  onOpened.mockReset()
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.restoreAllMocks()
})

it('shows the agent’s screen', async () => {
  const { container } = mount()
  await flush()

  expect(container.querySelector('img')?.getAttribute('src')).toBe('data:image/jpeg;base64,AA==')
})

it('stops asking for frames once it is gone', async () => {
  const view = mount()
  await flush()
  await act(() => vi.advanceTimersByTimeAsync(4000))
  const asked = plugin.frames.mock.calls.length
  expect(asked).toBeGreaterThan(1)

  view.unmount()
  await act(() => vi.advanceTimersByTimeAsync(20_000))

  expect(plugin.frames).toHaveBeenCalledTimes(asked)
})

it('opens the row it was given, and tells the page a Screen opened', async () => {
  mount()
  await flush()

  fireEvent.click(screen.getByRole('button'))

  expect(plugin.open).toHaveBeenCalledTimes(1)
  expect(plugin.open.mock.calls[0][0]).toBe(OPS)
  expect(onOpened).toHaveBeenCalledTimes(1)
})

it('tells the page nothing where this app cannot open a Screen', async () => {
  // An older Desktop has no workspace tabs: the plugin's open then only says so.
  const shell = host as { openWorkspace?: unknown }
  const openWorkspace = shell.openWorkspace
  shell.openWorkspace = undefined

  try {
    mount()
    await flush()

    fireEvent.click(screen.getByRole('button'))

    expect(onOpened).not.toHaveBeenCalled()
  } finally {
    shell.openWorkspace = openWorkspace
  }
})
