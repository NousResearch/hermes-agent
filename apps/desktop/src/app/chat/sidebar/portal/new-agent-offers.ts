import { useStore } from '@nanostores/react'

import type { AgentSettingsOwner } from '@/app/agent-settings/data'
import { agentMachine } from '@/app/agent-settings/landing'
import { useI18n } from '@/i18n'
import { $activeConnectionId } from '@/store/connections'
import { $fleetRoster } from '@/store/fleet-roster'
import { $profiles } from '@/store/profile'
import { $gatewayState } from '@/store/session'

import { machineState, useMachineStatusText } from '../simplified-gateway-header'
import type { useSimplifiedAgents } from '../use-simplified-agents'

/** Where New agent stands for one machine: the owner to open the dialog for, or what it waits for. */
export interface NewAgentOffer {
  connectionId: null | string
  label: string
  owner?: AgentSettingsOwner
  waiting?: string
}

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
    const state = machineState(connectionId, connectionId === (activeConnection ?? owner), gatewayState, roster)

    if (state !== 'open') {
      return { connectionId, label, waiting: status[state] }
    }

    // An installed agent the machine serves, from the roster or (this window's machine) its own
    // profile list: a row listed only for its chats may be of a profile the machine no longer has,
    // and the dialog reads and creates through this one.
    const rostered = roster?.agents.find(agent => agent.connectionId === connectionId && agent.reachable !== false)
    const profile = rostered ? rostered.targetProfile || rostered.profile : connectionId === owner ? profiles[0]?.name : undefined

    return profile
      ? { connectionId, label, owner: { connectionId: agentMachine(connectionId), profile } }
      : { connectionId, label, waiting: t.common.loading }
  })
}
