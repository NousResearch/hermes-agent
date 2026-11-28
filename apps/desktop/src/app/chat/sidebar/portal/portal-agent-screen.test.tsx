import { host } from '@hermes/plugin-sdk'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $lastRoster } from '@/plugins/hermes-bots/data'
import { botsText } from '@/plugins/hermes-bots/i18n'
import type * as ScreenPortal from '@/plugins/hermes-bots/screen-portal'

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

vi.mock('@/plugins/hermes-bots/screen-open', () => ({ openBotScreen: plugin.open }))
vi.mock('@/plugins/hermes-bots/screen-hero', () => ({ ScreenHero: () => null }))

const copy = botsText().screen
const OPS: SimplifiedAgent = { connectionId: 'lab', gateway: 'Lab box', id: 'lab/ops', label: 'ops', profile: 'ops' }
const row = () => screen.queryByRole('button', { name: new RegExp(`^${copy.portalTitle}`) })

beforeEach(() => {
  plugin.open.mockReset()
  vi.spyOn(host, 'profileRoutes').mockResolvedValue([])
})
afterEach(() => {
  cleanup()
  $lastRoster.set([])
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

it('opens that agent’s Screen', async () => {
  plugin.tone = 'live'
  render(<AgentScreenRow agent={OPS} />)

  fireEvent.click(row()!)

  await waitFor(() => expect(plugin.open).toHaveBeenCalledTimes(1))
  expect(plugin.open.mock.calls[0][0]).toMatchObject({ connectionId: 'lab', name: 'ops' })
})

it('stands on the agent’s own roster row, not on an alias that serves the same profile', () => {
  plugin.tone = 'unknown'
  // "nightly" is an alias served from ops, listed first; ops itself follows.
  $lastRoster.set([
    { connectionId: 'lab', name: 'nightly', sourceScoped: true, targetProfile: 'ops' },
    { connectionId: 'lab', name: 'ops', sourceScoped: true }
  ] as never)
  plugin.bots.length = 0
  render(<AgentScreenRow agent={OPS} />)

  expect(plugin.bots.at(-1)).toMatchObject({ name: 'ops' })
})

it('falls back to the alias that serves the profile when it has no row of its own', () => {
  plugin.tone = 'unknown'
  $lastRoster.set([{ connectionId: 'lab', name: 'nightly', sourceScoped: true, targetProfile: 'ops' }] as never)
  plugin.bots.length = 0
  render(<AgentScreenRow agent={OPS} />)

  expect(plugin.bots.at(-1)).toMatchObject({ name: 'nightly' })
})
