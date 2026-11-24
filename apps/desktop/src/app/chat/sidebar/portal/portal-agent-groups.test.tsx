import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import { newAgentTarget } from '@/app/agent-settings/new-agent-button'
import { StableNavigateProvider } from '@/app/stable-navigate'
import { RootTooltipProvider } from '@/components/ui/tooltip'
import { en } from '@/i18n/en'
import { $fleetRoster } from '@/store/fleet-roster'
import { $profiles } from '@/store/profile'
import { $connection, $gatewayState } from '@/store/session'
import { stubMenuDomApis } from '@/test/jsdom'

import type * as BotChatRow from '../simplified-bot-chat-row'
import { agentKey, type SimplifiedAgent } from '../use-simplified-agents'

import { useNewAgentOffers } from './new-agent-offers'
import { PortalAgentGroups } from './portal-agent-groups'

// Bot Mode loaded; its Screen row stands in as a marker, so mounting it (which reads a status) shows.
vi.mock('../simplified-bot-chat-row', async original => ({
  ...(await original<typeof BotChatRow>()),
  useBotModeLoaded: () => true
}))
vi.mock('./portal-agent-screen', () => ({
  AgentScreenRow: ({ agent }: { agent: SimplifiedAgent }) => <i data-agent={agent.profile} data-testid="screen-row" />
}))

const agent = (connectionId: string, profile: string, gateway: string): SimplifiedAgent => ({
  connectionId,
  gateway,
  id: JSON.stringify([connectionId, profile]),
  label: profile,
  profile
})

const source = (connectionId: string) => ({ connectionId, reachable: true })
const installed = (connectionId: string, profile: string) => ({ connectionId, profile })
const onNewAgent = vi.fn()

/** Studio is this window's machine; Lab box is another. */
function Groups(props: Omit<Parameters<typeof PortalAgentGroups>[0], 'newAgent'>) {
  return <PortalAgentGroups {...props} newAgent={useNewAgentOffers(props.agents)} />
}

function mount(agents: SimplifiedAgent[], expanded: Record<string, boolean> = {}) {
  return render(
    <MemoryRouter>
      <StableNavigateProvider>
        <RootTooltipProvider>
          <Groups
            agents={{ agents, owner: 'studio', sessionsByAgent: new Map() }}
            allSessions={[]}
            expanded={expanded}
            onNewAgent={onNewAgent}
            onOpenAgent={() => {}}
            onResumeSession={() => {}}
            onToggle={() => {}}
            pending={null}
            query=""
            renderSessions={() => null}
            scrollRef={{ current: null }}
          />
        </RootTooltipProvider>
      </StableNavigateProvider>
    </MemoryRouter>
  )
}

const newAgentOn = (machine: string) =>
  screen.getByRole('button', { name: new RegExp(`^${en.profiles.newProfile}.*: ${machine}$`) })

beforeEach(() => {
  stubMenuDomApis()
  onNewAgent.mockReset()
  $gatewayState.set('open')
  $connection.set({ connectionId: 'studio' } as never)
  $fleetRoster.set({
    agents: [installed('studio', 'default'), installed('lab', 'ops')],
    sources: [source('studio'), source('lab')]
  } as never)
})

afterEach(() => {
  cleanup()
  $fleetRoster.set(null)
  $profiles.set([])
  $connection.set(null)
  $gatewayState.set('idle')
  $apiRequestScope.set({ connectionId: null, profile: null })
})

it('gives every machine its own New agent, each for its own machine', () => {
  mount([agent('studio', 'default', 'Studio'), agent('lab', 'ops', 'Lab box')])

  fireEvent.click(newAgentOn('Lab box'))
  expect(onNewAgent).toHaveBeenLastCalledWith({ connectionId: 'lab', profile: 'ops' })

  fireEvent.click(newAgentOn('Studio'))
  expect(onNewAgent).toHaveBeenLastCalledWith({ connectionId: 'studio', profile: 'default' })
})

it('makes the agent on the row’s machine while another machine’s agent is in front', () => {
  mount([agent('studio', 'default', 'Studio'), agent('lab', 'ops', 'Lab box')])
  // An agent of Lab box in front: requests are scoped to it.
  $apiRequestScope.set({ connectionId: 'lab', profile: 'ops' })

  fireEvent.click(newAgentOn('Studio'))

  expect(newAgentTarget(onNewAgent.mock.lastCall?.[0]).owner).toEqual({ connectionId: 'studio', profile: 'default' })
})

it('borrows only an installed agent of the machine, and waits where it knows none', () => {
  // Lab box is listed for a chat of a profile it no longer has; its roster names another.
  mount([agent('studio', 'default', 'Studio'), agent('lab', 'aaa-gone', 'Lab box')])
  fireEvent.click(newAgentOn('Lab box'))
  expect(onNewAgent).toHaveBeenLastCalledWith({ connectionId: 'lab', profile: 'ops' })

  cleanup()
  onNewAgent.mockReset()
  $fleetRoster.set({ agents: [installed('studio', 'default')], sources: [source('studio'), source('lab')] } as never)
  mount([agent('studio', 'default', 'Studio'), agent('lab', 'aaa-gone', 'Lab box')])
  fireEvent.click(newAgentOn('Lab box'))

  expect(newAgentOn('Lab box').getAttribute('aria-disabled')).toBe('true')
  expect(onNewAgent).not.toHaveBeenCalled()
})

it('passes over an agent the machine lists but does not serve', () => {
  $fleetRoster.set({
    agents: [{ ...installed('lab', 'default'), reachable: false }, installed('lab', 'ops'), installed('studio', 'default')],
    sources: [source('studio'), source('lab')]
  } as never)
  mount([agent('studio', 'default', 'Studio'), agent('lab', 'ops', 'Lab box')])

  fireEvent.click(newAgentOn('Lab box'))

  expect(onNewAgent).toHaveBeenLastCalledWith({ connectionId: 'lab', profile: 'ops' })
})

it('reads this window’s own machine from its profile list where there is no roster', () => {
  $fleetRoster.set(null)
  $profiles.set([{ name: 'default' }] as never)
  mount([agent('studio', 'default', 'Studio')])

  fireEvent.click(newAgentOn('Studio'))

  expect(onNewAgent).toHaveBeenLastCalledWith({ connectionId: 'studio', profile: 'default' })
})

it('reads a live Screen only for the agents of this window’s machine', () => {
  mount([agent('studio', 'default', 'Studio'), agent('lab', 'ops', 'Lab box')])

  expect(screen.getAllByTestId('screen-row').map(row => row.dataset.agent)).toEqual(['default'])
})

it('keeps that to the machine the list was read from while another machine’s agent is in front', () => {
  // Lab box is the active connection; the list's owner, Studio, is still whose agents are listed here.
  $connection.set({ connectionId: 'lab' } as never)
  mount([agent('studio', 'default', 'Studio'), agent('lab', 'ops', 'Lab box')])

  expect(screen.getAllByTestId('screen-row').map(row => row.dataset.agent)).toEqual(['default'])
})

it('reads no Screen for an agent whose card is folded', () => {
  const studio = agent('studio', 'default', 'Studio')
  mount([studio], { [agentKey(studio)]: false })

  expect(screen.queryByTestId('screen-row')).toBeNull()
})
