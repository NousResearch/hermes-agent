import { useStore } from '@nanostores/react'

import type { AgentSettingsOwner } from '@/app/agent-settings/data'
import { agentMachine } from '@/app/agent-settings/landing'
import { useI18n } from '@/i18n'
import { $activeConnectionId } from '@/store/connections'
import { $fleetRoster } from '@/store/fleet-roster'
import { $profiles } from '@/store/profile'
import { $gatewayState } from '@/store/session'

import type { useSimplifiedAgents } from '../use-simplified-agents'

export type MachineState = 'open' | 'connecting' | 'idle' | 'closed' | 'error'

/** A machine's state as the app knows it: the gateway's own for the window's machine, the roster's
 *  reach for another. */
export function machineState(
  connectionId: null | string,
  selected: boolean,
  gatewayState: MachineState,
  roster: null | { sources: readonly { connectionId: null | string; reachable?: boolean }[] }
): MachineState {
  const source = roster?.sources.find(source => source.connectionId === connectionId)

  return selected ? gatewayState : source ? (source.reachable ? 'open' : 'closed') : 'idle'
}

export function useMachineStatusText(): Record<MachineState, string> {
  const { t } = useI18n()

  return {
    open: t.shell.gatewayMenu.connected,
    connecting: t.shell.gatewayMenu.connecting,
    idle: t.common.loading,
    closed: t.shell.gatewayMenu.offline,
    error: t.shell.gatewayMenu.offline
  }
}

interface Machine {
  connectionId: null | string
  label: string
  /** The window's own machine. */
  selected: boolean
  state: MachineState
  /** New agent's tip: its name, and what it waits for while this machine cannot take one. */
  tip: string
}

/** A listed machine and where New agent stands for it: the owner to open the dialog for, or what
 *  it waits for. */
export type NewAgentOffer = Machine & ({ owner: AgentSettingsOwner; waiting?: never } | { owner?: never; waiting: string })

/** New agent for every machine the sidebar lists, in the list's order. A machine's row and the
 *  Agents header both read this, so they cannot say different things about the same machine. */
export function useNewAgentOffers({ agents, owner }: ReturnType<typeof useSimplifiedAgents>): NewAgentOffer[] {
  const { t } = useI18n()
  const status = useMachineStatusText()
  const roster = useStore($fleetRoster)
  const profiles = useStore($profiles)
  const activeConnection = useStore($activeConnectionId)
  const gatewayState = useStore($gatewayState)
  const machines = new Map(agents.map(agent => [agent.connectionId, agent.gateway]))

  return [...machines].map(([connectionId, label]) => {
    const selected = connectionId === (activeConnection ?? owner)
    const state = machineState(connectionId, selected, gatewayState, roster)
    const machine = { connectionId, label, selected, state }
    const waitingFor = (waiting: string) => ({ ...machine, tip: `${t.profiles.newProfile} · ${waiting}`, waiting })

    if (state !== 'open') {
      return waitingFor(status[state])
    }

    // An installed agent the machine serves, from the roster or (this window's machine) its own
    // profile list: a row listed only for its chats may be of a profile the machine no longer has,
    // and the dialog reads and creates through this one.
    const rostered = roster?.agents.find(agent => agent.connectionId === connectionId && agent.reachable !== false)
    const profile = rostered ? rostered.targetProfile || rostered.profile : connectionId === owner ? profiles[0]?.name : undefined

    return profile
      ? { ...machine, owner: { connectionId: agentMachine(connectionId), profile }, tip: t.profiles.newProfile }
      : waitingFor(t.common.loading)
  })
}
