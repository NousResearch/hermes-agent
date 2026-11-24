import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $lastRoster } from '@/plugins/hermes-bots/data'
import { botsText } from '@/plugins/hermes-bots/i18n'
import type * as ScreenPortal from '@/plugins/hermes-bots/screen-portal'
import { $gatewayState } from '@/store/session'

import type { SimplifiedAgent } from '../use-simplified-agents'

import { AgentScreenRow } from './portal-agent-screen'

const plugin = vi.hoisted(() => ({
  bots: [] as unknown[],
  tone: 'live' as ScreenPortal.PortalTone,
  open: vi.fn(),
  route: null as unknown
}))

vi.mock('@/plugins/hermes-bots/i18n', async original => (await import('@/test/bots-screen-mocks')).botsI18nMock(original))
vi.mock('@/plugins/hermes-bots/screen-portal', async original =>
  (await import('@/test/bots-screen-mocks')).screenPortalMock(original, plugin)
)

const copy = botsText().screen
const OPS: SimplifiedAgent = { connectionId: 'lab', gateway: 'Lab box', id: 'lab/ops', label: 'ops', profile: 'ops' }
const row = () => screen.queryByRole('button', { name: new RegExp(`^${copy.portalTitle}`) })

beforeEach(() => plugin.open.mockReset())
afterEach(() => {
  cleanup()
  $lastRoster.set([])
  $gatewayState.set('idle')
})

it.each([
  ['live', copy.portalWatching, copy.holderAgent, 'working'],
  ['human', copy.portalYouControl, copy.holderYou, 'needs-input'],
  ['other', copy.portalOtherControls, copy.holderViewer, 'needs-input']
] as const)('shows a running screen (%s): who holds it in a word, and the mark for agent or person', (tone, phrase, holder, mark) => {
  plugin.tone = tone
  const { container } = render(<AgentScreenRow agent={OPS} />)

  expect(row()?.getAttribute('aria-label')).toBe(`${copy.portalTitle}: ${phrase}`)
  expect(row()?.textContent).toBe(copy.portalTitle)
  expect(container.querySelector('.portal-session-meta')?.textContent).toBe(holder)
  expect(container.querySelector<HTMLElement>('.portal-row-mark')?.dataset.state).toBe(mark)
})

it.each(['off', 'missing', 'unknown', 'unsupported', 'unavailable'] as const)('adds no row for a screen that is %s', tone => {
  plugin.tone = tone
  render(<AgentScreenRow agent={OPS} />)

  expect(row()).toBeNull()
})

it('opens that agent’s Screen', () => {
  plugin.tone = 'live'
  render(<AgentScreenRow agent={OPS} />)

  fireEvent.click(row()!)

  expect(plugin.route).toEqual({ connectionId: 'lab', profile: 'ops' })
  expect(plugin.open).toHaveBeenCalledTimes(1)
})

it('keeps reading one row for its agent when the roster is published again', () => {
  plugin.tone = 'unknown'
  plugin.bots.length = 0
  render(<AgentScreenRow agent={OPS} />)

  act(() => $lastRoster.set([{ connectionId: 'lab', name: 'ops' }] as never))
  act(() => $lastRoster.set([{ connectionId: 'lab', name: 'ops' }] as never))

  // The plugin's hook re-reads an unsettled status whenever the row object it is given changes.
  expect(new Set(plugin.bots.slice(-2)).size).toBe(1)
})

it('asks again when the gateway’s state changes: a read made before it opened has failed', () => {
  plugin.tone = 'unknown'
  $gatewayState.set('connecting')
  plugin.bots.length = 0
  render(<AgentScreenRow agent={OPS} />)
  const before = plugin.bots.at(-1)

  act(() => $gatewayState.set('open'))

  expect(plugin.bots.at(-1)).not.toBe(before)
})
