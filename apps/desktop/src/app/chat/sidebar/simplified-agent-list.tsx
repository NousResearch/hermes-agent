import { useStore } from '@nanostores/react'
import { type ReactNode, useLayoutEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router'

import { SETTINGS_ROUTE } from '@/app/routes'
import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { $portalAgentsSectionOpen, $sidebarAgentScroll } from '@/store/agent-sidebar-preferences'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $activeConnectionId, selectConnection } from '@/store/connections'
import { $fleetRoster } from '@/store/fleet-roster'
import { recordIaAction } from '@/store/ia-metrics'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile, ensureGatewayProfile } from '@/store/profile'
import { $gatewayState } from '@/store/session'
import { setSettingsScope } from '@/store/settings-scope'

import { PortalAgentGroups } from './portal/portal-agent-groups'
import { PortalSectionHeader } from './portal/portal-section-header'
import { useFleetRoster } from './use-fleet-roster'
import type { SimplifiedAgent, useSimplifiedAgents } from './use-simplified-agents'

interface SimplifiedAgentListProps {
  query: string
  revealSession?: { key: string } | null
  agents: ReturnType<typeof useSimplifiedAgents>
  allSessions: SessionInfo[]
  expanded: Record<string, boolean>
  onToggle: (id: string, open: boolean) => void
  onResumeSession: (id: string, row?: SessionInfo) => void
  renderSessions: (sessions: SessionInfo[], agent: SimplifiedAgent) => ReactNode
}

function useOpenAgent(owner: string | null) {
  const { t } = useI18n()
  const navigate = useNavigate()
  const registry = useStore($connectionsRegistry)
  const activeConnection = useStore($activeConnectionId)
  const [pending, setPending] = useState<string | null>(null)

  async function openAgent(agent: SimplifiedAgent) {
    const state =
      agent.connectionId === (activeConnection ?? owner)
        ? $gatewayState.get()
        : $fleetRoster.get()?.sources.find(source => source.connectionId === agent.connectionId)?.reachable

    if (state === 'closed' || state === 'error' || state === false) {
      recordIaAction('ia.gateway.offline_agent')
    }

    const profile = agent.targetProfile || agent.profile
    setPending(agent.id)

    try {
      if (agent.connectionId && registry?.connections.some(connection => connection.id === agent.connectionId)) {
        await selectConnection(agent.connectionId, { profile })

        if ($activeConnectionId.get() !== agent.connectionId || $activeGatewayProfile.get() !== profile) {
          return
        }
      } else {
        if (agent.connectionId !== owner) {
          throw new Error(t.profiles.fleet.gatewayUnreachable(agent.gateway))
        }

        await ensureGatewayProfile(profile)
      }

      setSettingsScope(profile)
      const params = new URLSearchParams({ agentProfile: profile })

      if (agent.connectionId) {
        params.set('connection', agent.connectionId)
      }

      navigate(`${SETTINGS_ROUTE}?${params}`)
    } catch (error) {
      notifyError(error, t.errors.genericFailure)
    } finally {
      setPending(null)
    }
  }

  return { openAgent, pending }
}

function useAgentScroll({ agents, allSessions, revealSession }: SimplifiedAgentListProps) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const initialScroll = useRef($sidebarAgentScroll.get())
  const restored = useRef(false)

  useLayoutEffect(() => {
    const scroller = scrollRef.current

    if (!scroller || restored.current) {
      return
    }

    scroller.scrollTop = initialScroll.current
    // Roster/session loading can temporarily make the list shorter than its saved position.
    restored.current = scroller.scrollHeight - scroller.clientHeight >= initialScroll.current
  }, [agents.agents.length, allSessions.length])

  useLayoutEffect(() => {
    const scroller = scrollRef.current

    if (!revealSession || !scroller) {
      return
    }

    const row = [...scroller.querySelectorAll<HTMLElement>('[data-session-key]')].find(
      element => element.dataset.sessionKey === revealSession.key
    )

    if (!row) {
      return
    }

    restored.current = true
    const viewport = scroller.getBoundingClientRect()
    const bounds = row.getBoundingClientRect()

    const inset = Math.min(
      parseFloat(getComputedStyle(document.documentElement).fontSize) * 2.5 || 40,
      scroller.clientHeight / 4
    )

    const top = viewport.top + inset
    const bottom = viewport.top + scroller.clientHeight - inset
    const delta = bounds.top < top ? bounds.top - top : bounds.bottom > bottom ? bounds.bottom - bottom : 0

    if (delta) {
      scroller.scrollTop += delta
    }
  }, [revealSession])

  useLayoutEffect(() => {
    const scroller = scrollRef.current

    if (!scroller) {
      return
    }

    const save = () => {
      if (restored.current) {
        $sidebarAgentScroll.set(scroller.scrollTop)
      }
    }

    const takeControl = () => {
      restored.current = true
    }

    scroller.addEventListener('scroll', save, { passive: true })
    scroller.addEventListener('wheel', takeControl, { passive: true })
    scroller.addEventListener('touchstart', takeControl, { passive: true })
    scroller.addEventListener('keydown', takeControl)

    return () => {
      save()
      scroller.removeEventListener('scroll', save)
      scroller.removeEventListener('wheel', takeControl)
      scroller.removeEventListener('touchstart', takeControl)
      scroller.removeEventListener('keydown', takeControl)
    }
  }, [])

  return scrollRef
}

/** Mounted only by the redesigned sidebar; hiding never changes the source roster or session stores. */
export function SimplifiedAgentList(props: SimplifiedAgentListProps) {
  const { t } = useI18n()
  const { openAgent, pending } = useOpenAgent(props.agents.owner)
  const scroll = useAgentScroll(props)
  const open = useStore($portalAgentsSectionOpen)
  useFleetRoster(true)

  return (
    <section
      aria-label={t.simplifiedAgents.agents}
      className="portal-section"
      data-open={open}
      data-slot="agent-section"
    >
      <PortalSectionHeader
        count={props.agents.agents.length}
        label={t.simplifiedAgents.agents}
        onToggle={() => $portalAgentsSectionOpen.set(!open)}
        open={open}
      />
      {/* Hidden, not unmounted: the scroller owns restored position and reveal listeners. */}
      <div className="portal-section-body" hidden={!open}>
        <PortalAgentGroups
          {...props}
          onOpenAgent={agent => void openAgent(agent)}
          pending={pending}
          scrollRef={scroll}
        />
      </div>
    </section>
  )
}
