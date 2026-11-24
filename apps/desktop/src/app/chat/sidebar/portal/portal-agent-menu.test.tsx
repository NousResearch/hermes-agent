import { host } from '@hermes/plugin-sdk'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { RootTooltipProvider } from '@/components/ui/tooltip'
import { $botMeta, botMetaKey } from '@/plugins/hermes-bots/data'
import { botsText } from '@/plugins/hermes-bots/i18n'
import type * as ScreenOpen from '@/plugins/hermes-bots/screen-open'
import type { RosterRow } from '@/plugins/hermes-bots/types'
import { stubMenuDomApis } from '@/test/jsdom'

import type { SimplifiedAgent } from '../use-simplified-agents'

import { PortalAgentMenu } from './portal-agent-menu'

const plugin = vi.hoisted(() => ({ bots: [] as unknown[], loaded: true, open: vi.fn(), toggle: vi.fn() }))

vi.mock('../simplified-bot-chat-row', () => ({ useBotModeLoaded: () => plugin.loaded }))
vi.mock('@/plugins/hermes-bots/i18n', async original => (await import('@/test/bots-screen-mocks')).botsI18nMock(original))
vi.mock('@/plugins/hermes-bots/screen-portal', async original =>
  (await import('@/test/bots-screen-mocks')).screenPortalMock(original, plugin)
)
vi.mock('@/plugins/hermes-bots/screen-open', async original => ({
  ...(await original<typeof ScreenOpen>()),
  openBotScreen: plugin.open,
  toggleBotScreenAutoOpen: plugin.toggle
}))
vi.mock('@/plugins/hermes-bots/screen-hero', () => ({ ScreenHero: () => null }))

const copy = botsText().screen
const LAB_OPS = { connectionId: 'lab', connectionKind: 'remote', name: 'ops', sourceScoped: true } as RosterRow
const OPS: SimplifiedAgent = { connectionId: 'lab', gateway: 'Lab box', id: 'lab/ops', label: 'ops', profile: 'ops' }

function openMenu() {
  render(
    <RootTooltipProvider>
      <PortalAgentMenu agent={OPS} hidden={false} onOpen={() => {}} pending={false} />
    </RootTooltipProvider>
  )
  fireEvent.pointerDown(screen.getByRole('button', { name: /ops · Lab box$/ }), { button: 0, ctrlKey: false })
}

beforeEach(() => {
  stubMenuDomApis()
  plugin.loaded = true
  plugin.open.mockReset()
  vi.spyOn(host, 'profileRoutes').mockResolvedValue([])
  plugin.toggle.mockReset()
})

afterEach(() => {
  cleanup()
  $botMeta.set({})
})

it('opens this agent’s Screen, whichever chat is in front', async () => {
  openMenu()

  fireEvent.click(await screen.findByRole('menuitem', { name: copy.menu }))

  await waitFor(() => expect(plugin.open).toHaveBeenCalledTimes(1))
  expect(plugin.open.mock.calls[0][0]).toMatchObject({ connectionId: 'lab', name: 'ops' })
})

it('turns "open Screen when it is used" on and off for this agent, showing what is stored', async () => {
  openMenu()
  // What classic stored for this agent, under the roster's own key for a row on another machine.
  $botMeta.set({ [botMetaKey(LAB_OPS)]: { screenAutoOpen: true } })
  const item = await screen.findByRole('menuitemcheckbox', { name: copy.autoOpenMenu })
  expect(item.getAttribute('aria-checked')).toBe('true')

  fireEvent.click(item)

  expect(plugin.toggle).toHaveBeenCalledTimes(1)
  expect(plugin.toggle.mock.calls[0][0]).toMatchObject({ connectionId: 'lab', name: 'ops' })
})

it('offers no Screen items while Bot Mode is not loaded', async () => {
  plugin.loaded = false
  openMenu()

  await screen.findByRole('menu')
  expect(screen.queryByRole('menuitem', { name: copy.menu })).toBeNull()
  expect(screen.queryByRole('menuitemcheckbox', { name: copy.autoOpenMenu })).toBeNull()
})
