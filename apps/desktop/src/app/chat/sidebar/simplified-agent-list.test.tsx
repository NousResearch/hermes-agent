import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { AgentSettingsOwner } from '@/app/agent-settings/data'
import { StableNavigateProvider } from '@/app/stable-navigate'
import { RootTooltipProvider } from '@/components/ui/tooltip'
import { en } from '@/i18n/en'
import { $portalAgentsSectionOpen } from '@/store/agent-sidebar-preferences'
import { $fleetRoster } from '@/store/fleet-roster'
import { $connection, $gatewayState } from '@/store/session'
import { stubMenuDomApis } from '@/test/jsdom'

import { SimplifiedAgentList } from './simplified-agent-list'
import type { SimplifiedAgent } from './use-simplified-agents'

const dialog = vi.hoisted(() => ({ owner: null as AgentSettingsOwner | null }))

vi.mock('@/app/agent-settings/new-agent-dialog', () => ({
  default: ({ owner }: { owner: AgentSettingsOwner }) => {
    dialog.owner = owner

    return <div role="dialog" />
  }
}))

const agent = (connectionId: string, profile: string, gateway: string): SimplifiedAgent => ({
  connectionId,
  gateway,
  id: JSON.stringify([connectionId, profile]),
  label: profile,
  profile
})

const STUDIO = agent('studio', 'default', 'Studio')
const LAB = agent('lab', 'ops', 'Lab box')

function mount(agents: SimplifiedAgent[], query = '') {
  return render(
    <MemoryRouter>
      <StableNavigateProvider>
        <RootTooltipProvider>
          <SimplifiedAgentList
            agents={{ agents, owner: 'studio', sessionsByAgent: new Map() }}
            allSessions={[]}
            expanded={{}}
            onResumeSession={() => {}}
            onToggle={() => {}}
            query={query}
            renderSessions={() => null}
          />
        </RootTooltipProvider>
      </StableNavigateProvider>
    </MemoryRouter>
  )
}

/** The section header's own "+", as distinct from a machine row's ("New agent: <machine>"). */
const fallback = () => screen.queryByRole('button', { name: en.profiles.newProfile })

beforeEach(() => {
  stubMenuDomApis()
  dialog.owner = null
  $gatewayState.set('open')
  $connection.set({ connectionId: 'studio' } as never)
  $fleetRoster.set({
    agents: [
      { connectionId: 'studio', profile: 'default' },
      { connectionId: 'lab', profile: 'ops' }
    ],
    sources: [
      { connectionId: 'studio', reachable: true },
      { connectionId: 'lab', reachable: true }
    ]
  } as never)
})

afterEach(() => {
  cleanup()
  $portalAgentsSectionOpen.set(true)
  $fleetRoster.set(null)
  $connection.set(null)
  $gatewayState.set('idle')
})

it('offers New agent in the Agents header only while no machine row is there to offer it', () => {
  mount([STUDIO])
  expect(fallback()).toBeNull()

  act(() => $portalAgentsSectionOpen.set(false))
  expect(fallback()).not.toBeNull()

  cleanup()
  $portalAgentsSectionOpen.set(true)
  mount([STUDIO], 'no agent is called this')
  expect(fallback()).not.toBeNull()

  cleanup()
  mount([STUDIO], 'defa')
  expect(fallback()).toBeNull()
})

it('with one machine that cannot take an agent, waits in place and says why', async () => {
  $gatewayState.set('closed')
  $portalAgentsSectionOpen.set(false)
  mount([STUDIO])

  const button = screen.getByRole('button', { name: `${en.profiles.newProfile} · ${en.shell.gatewayMenu.offline}` })
  fireEvent.click(button)
  // The dialog loads lazily: give an opened one the time the others need to appear.
  await act(() => new Promise(resolve => setTimeout(resolve, 50)))

  expect(button.getAttribute('aria-disabled')).toBe('true')
  expect(dialog.owner).toBeNull()
  expect(screen.queryByRole('dialog')).toBeNull()
})

it('with one machine, starts the agent on it directly', async () => {
  $portalAgentsSectionOpen.set(false)
  mount([STUDIO])

  fireEvent.click(fallback()!)

  await screen.findByRole('dialog')
  expect(dialog.owner).toEqual({ connectionId: 'studio', profile: 'default' })
})

it('with several machines, asks which, and starts the agent on the one picked', async () => {
  $portalAgentsSectionOpen.set(false)
  mount([STUDIO, LAB])

  fireEvent.pointerDown(fallback()!, { button: 0, ctrlKey: false })
  fireEvent.click(await screen.findByRole('menuitem', { name: 'Lab box' }))

  await screen.findByRole('dialog')
  expect(dialog.owner).toEqual({ connectionId: 'lab', profile: 'ops' })
})

it('lists a machine that cannot take an agent, disabled, with the reason its row gives', async () => {
  $fleetRoster.set({
    agents: [
      { connectionId: 'studio', profile: 'default' },
      { connectionId: 'lab', profile: 'ops' }
    ],
    sources: [
      { connectionId: 'studio', reachable: true },
      { connectionId: 'lab', reachable: false }
    ]
  } as never)
  $portalAgentsSectionOpen.set(false)
  mount([STUDIO, LAB])

  fireEvent.pointerDown(fallback()!, { button: 0, ctrlKey: false })
  const item = await screen.findByRole('menuitem', { name: /Lab box/ })

  expect(item.getAttribute('aria-disabled')).toBe('true')
  expect(item.textContent).toBe(`Lab box${en.shell.gatewayMenu.offline}`)
})
