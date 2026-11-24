import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { botsText } from '@/plugins/hermes-bots/i18n'
import type * as BotsI18n from '@/plugins/hermes-bots/i18n'
import type * as ScreenPortal from '@/plugins/hermes-bots/screen-portal'

import type { SimplifiedAgent } from '../use-simplified-agents'

import { AgentScreenRow } from './portal-agent-screen'

const plugin = vi.hoisted(() => ({ tone: 'live' as ScreenPortal.PortalTone, open: vi.fn(), route: null as unknown }))

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
  useScreenPortalState: () => ({ lease: null, status: null, tone: plugin.tone })
}))

const copy = botsText().screen
const OPS: SimplifiedAgent = { connectionId: 'lab', gateway: 'Lab box', id: 'lab/ops', label: 'ops', profile: 'ops' }
const row = () => screen.queryByRole('button', { name: new RegExp(`^${copy.portalTitle}`) })

beforeEach(() => plugin.open.mockReset())
afterEach(cleanup)

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
