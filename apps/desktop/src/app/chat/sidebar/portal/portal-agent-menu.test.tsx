import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { RootTooltipProvider } from '@/components/ui/tooltip'
import { $botMeta } from '@/plugins/hermes-bots/data'
import { botsText } from '@/plugins/hermes-bots/i18n'
import type * as BotsI18n from '@/plugins/hermes-bots/i18n'
import type * as ScreenOpen from '@/plugins/hermes-bots/screen-open'
import type * as ScreenPortal from '@/plugins/hermes-bots/screen-portal'
import { stubMenuDomApis } from '@/test/jsdom'

import type { SimplifiedAgent } from '../use-simplified-agents'

import { PortalAgentMenu } from './portal-agent-menu'

const plugin = vi.hoisted(() => ({ loaded: true, open: vi.fn(), toggle: vi.fn(), route: null as unknown }))

vi.mock('../simplified-bot-chat-row', () => ({ useBotModeLoaded: () => plugin.loaded }))
vi.mock('@/plugins/hermes-bots/i18n', async original => {
  const module = await original<typeof BotsI18n>()

  return { ...module, useBots: () => module.BOTS_LOCALES.en }
})
vi.mock('@/plugins/hermes-bots/screen-portal', async original => ({
  ...(await original<typeof ScreenPortal>()),
  profileScreenCommand: (route: unknown) => {
    plugin.route = route

    return { label: '', run: plugin.open }
  }
}))
vi.mock('@/plugins/hermes-bots/screen-open', async original => ({
  ...(await original<typeof ScreenOpen>()),
  toggleBotScreenAutoOpen: plugin.toggle
}))

const copy = botsText().screen
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
  plugin.toggle.mockReset()
})

afterEach(() => {
  cleanup()
  $botMeta.set({})
})

it('opens this agent’s Screen, whichever chat is in front', async () => {
  openMenu()

  fireEvent.click(await screen.findByRole('menuitem', { name: copy.menu }))

  expect(plugin.route).toEqual({ connectionId: 'lab', profile: 'ops' })
  expect(plugin.open).toHaveBeenCalledTimes(1)
})

it('turns "open Screen when it is used" on and off for this agent, showing what is stored', async () => {
  openMenu()
  const item = await screen.findByRole('menuitemcheckbox', { name: copy.autoOpenMenu })
  expect(item.getAttribute('aria-checked')).toBe('false')

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
