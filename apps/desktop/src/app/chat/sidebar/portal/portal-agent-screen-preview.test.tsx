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
// jsdom never reports :focus-visible, so each test says whether focus arrived by keyboard.
const focusKind = { keyboard: false }
const matches = Element.prototype.matches

function focusRow(keyboard: boolean) {
  focusKind.keyboard = keyboard
  act(() => row().focus())
}

const rest = () => fireEvent.pointerEnter(row().closest('.portal-session-row')!, { pointerType: 'mouse' })
const leave = () => fireEvent.pointerLeave(row().closest('.portal-session-row')!, { pointerType: 'mouse' })

beforeEach(() => {
  stubMenuDomApis()
  vi.useFakeTimers()
  plugin.frames.mockClear()
  plugin.tone = 'live'
  focusKind.keyboard = false
  plugin.open.mockReset()
  vi.spyOn(host, 'profileRoutes').mockResolvedValue([])
  vi.spyOn(Element.prototype, 'matches').mockImplementation(function (this: Element, selector: string) {
    return selector === ':focus-visible' ? focusKind.keyboard && this === document.activeElement : matches.call(this, selector)
  })
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

  focusRow(true)
  await wait(0)
  expect(preview()).not.toBeNull()
  expect(document.activeElement).toBe(row())

  fireEvent.keyDown(row(), { key: 'Escape' })
  await wait(0)
  expect(preview()).toBeNull()

  act(() => row().blur())
  focusRow(true)
  await wait(0)
  act(() => row().blur())
  await wait(0)
  expect(preview()).toBeNull()
})

it('does not open for focus the mouse left on the row, however often focus comes back to it', async () => {
  render(<AgentScreenRow agent={OPS} />)

  // A click focuses the row; the window losing and regaining focus then blurs and focuses it again.
  focusRow(false)
  act(() => row().blur())
  focusRow(false)
  await wait(1000)

  expect(preview()).toBeNull()
  expect(plugin.frames).not.toHaveBeenCalled()
})

it('lets a keyboard-opened card be clicked: focus moving into the card does not close it', async () => {
  render(<AgentScreenRow agent={OPS} />)
  focusRow(true)
  await wait(0)
  const tile = preview()!.querySelector('button')!

  fireEvent.blur(row(), { relatedTarget: tile })
  await wait(0)

  expect(preview()).not.toBeNull()
})

it('closes a keyboard-opened card when its picture is pressed to open the Screen', async () => {
  render(<AgentScreenRow agent={OPS} />)
  focusRow(true)
  await wait(0)
  const tile = preview()!.querySelector('button')!

  fireEvent.blur(row(), { relatedTarget: tile })
  fireEvent.click(tile)
  await wait(0)

  expect(preview()).toBeNull()
})

it('starts closed again when the screen stops and later runs again', async () => {
  const view = render(<AgentScreenRow agent={OPS} />)
  rest()
  await wait(400)
  expect(preview()).not.toBeNull()

  plugin.tone = 'off'
  view.rerender(<AgentScreenRow agent={OPS} />)
  await wait(0)
  plugin.tone = 'live'
  view.rerender(<AgentScreenRow agent={OPS} />)
  await wait(1000)

  expect(preview()).toBeNull()
})

it('still opens the Screen from the row, with or without the preview up', async () => {
  vi.useRealTimers()
  render(<AgentScreenRow agent={OPS} />)

  fireEvent.click(row())

  await vi.waitFor(() => expect(plugin.open).toHaveBeenCalledTimes(1))
})
