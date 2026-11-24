import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $lastRoster } from '@/plugins/hermes-bots/data'
import { botsText } from '@/plugins/hermes-bots/i18n'
import type * as BotsI18n from '@/plugins/hermes-bots/i18n'
import type * as ScreenPortal from '@/plugins/hermes-bots/screen-portal'

import type { SimplifiedAgent } from '../use-simplified-agents'

import { AgentScreenRow } from './portal-agent-screen'

const plugin = vi.hoisted(() => ({
  bots: [] as unknown[],
  tone: 'live' as ScreenPortal.PortalTone,
  open: vi.fn(),
  route: null as unknown
}))

vi.mock('@/plugins/hermes-bots/i18n', async original => {
  const module = await original<typeof BotsI18n>()

  return { ...module, useBots: () => module.botsText() }
})
vi.mock('@/plugins/hermes-bots/screen-portal', async original => ({
  ...(await original<typeof ScreenPortal>()),
  profileScreenCommand: (route: unknown) => {
    plugin.route = route

    return { label: '', run: plugin.open }
  },
  useScreenPortalState: (bot: unknown) => {
    plugin.bots.push(bot)

    return { lease: null, status: null, tone: plugin.tone }
  }
}))

const copy = botsText().screen
const OPS: SimplifiedAgent = { connectionId: 'lab', gateway: 'Lab box', id: 'lab/ops', label: 'ops', profile: 'ops' }
const row = () => screen.queryByRole('button', { name: new RegExp(`^${copy.portalTitle}`) })

beforeEach(() => plugin.open.mockReset())
afterEach(() => {
  cleanup()
  $lastRoster.set([])
})

it.each([
  ['live', copy.portalWatching],
  ['human', copy.portalYouControl],
  ['other', copy.portalOtherControls]
] as const)('shows a running screen (%s) and says who holds it', (tone, holder) => {
  plugin.tone = tone
  render(<AgentScreenRow agent={OPS} />)

  expect(row()?.textContent).toBe(`${copy.portalTitle} · ${holder}`)
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
