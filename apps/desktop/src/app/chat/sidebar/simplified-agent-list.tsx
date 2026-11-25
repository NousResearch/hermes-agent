import { useStore } from '@nanostores/react'
import { lazy, type ReactNode, Suspense, useLayoutEffect, useRef, useState } from 'react'

import type { AgentSettingsOwner } from '@/app/agent-settings/data'
import { SETTINGS_ROUTE } from '@/app/routes'
import { useStableNavigate } from '@/app/stable-navigate'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Tip } from '@/components/ui/tooltip'
import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { Plus } from '@/lib/icons'
import { $hiddenSidebarAgents, $portalAgentsSectionOpen, $sidebarAgentScroll } from '@/store/agent-sidebar-preferences'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $activeConnectionId, selectConnection } from '@/store/connections'
import { $fleetRoster } from '@/store/fleet-roster'
import { recordIaAction } from '@/store/ia-metrics'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile, ensureGatewayProfile } from '@/store/profile'
import { $gatewayState } from '@/store/session'
import { setSettingsScope } from '@/store/settings-scope'

import { type NewAgentOffer, useNewAgentOffers } from './portal/new-agent-offers'
import { agentMatches, PortalAgentGroups } from './portal/portal-agent-groups'
import { PortalSectionHeader } from './portal/portal-section-header'
import { useFleetRoster } from './use-fleet-roster'
import { agentKey, type SimplifiedAgent, type useSimplifiedAgents } from './use-simplified-agents'

// Loaded when first opened: the dialog belongs to the Agents page's chunk.
const NewAgentDialog = lazy(() => import('@/app/agent-settings/new-agent-dialog'))

interface SimplifiedAgentListProps {
  query: string
  revealSession?: { key: string } | null
  agents: ReturnType<typeof useSimplifiedAgents>
  allSessions: SessionInfo[]
  expanded: Record<string, boolean>
  /** The header's own action, ahead of New agent: New group chat while an empty Group chats stands
   *  down (HADESK-132). */
  headerAction?: ReactNode
  onToggle: (id: string, open: boolean) => void
  onResumeSession: (id: string, row?: SessionInfo) => void
  renderSessions: (sessions: SessionInfo[], agent: SimplifiedAgent) => ReactNode
}

function useOpenAgent(owner: string | null) {
  const { t } = useI18n()
  const navigate = useStableNavigate()
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

interface NewAgentFallbackProps {
  offers: NewAgentOffer[]
  onPick: (owner: AgentSettingsOwner) => void
}

/** New agent in the Agents header: straight to the only machine, or a pick among several. */
function NewAgentFallback({ offers, onPick }: NewAgentFallbackProps) {
  const { t } = useI18n()
  const [only] = offers

  const button = (label: string, onClick?: () => void, waiting?: string) => (
    <Button
      aria-disabled={waiting ? true : undefined}
      aria-label={label}
      className="portal-section-action"
      onClick={onClick}
      size="icon-xs"
      variant="ghost"
    >
      <Plus aria-hidden />
    </Button>
  )

  if (offers.length === 1) {
    return (
      <Tip label={only.tip}>{button(only.tip, only.owner && (() => onPick(only.owner)), only.waiting)}</Tip>
    )
  }

  return (
    <DropdownMenu>
      <Tip label={t.profiles.newProfile}>
        <DropdownMenuTrigger asChild>{button(t.profiles.newProfile)}</DropdownMenuTrigger>
      </Tip>
      <DropdownMenuContent align="end">
        {offers.map(machine => (
          <DropdownMenuItem
            disabled={!machine.owner}
            key={machine.connectionId ?? ''}
            onSelect={() => machine.owner && onPick(machine.owner)}
          >
            {machine.label}
            {machine.waiting && <span className="ml-auto pl-4 text-muted-foreground">{machine.waiting}</span>}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

/** Mounted only by the redesigned sidebar; hiding never changes the source roster or session stores. */
export function SimplifiedAgentList(props: SimplifiedAgentListProps) {
  const { t } = useI18n()
  const { openAgent, pending } = useOpenAgent(props.agents.owner)
  const scroll = useAgentScroll(props)
  const open = useStore($portalAgentsSectionOpen)
  const hidden = useStore($hiddenSidebarAgents)
  // The New agent dialog's owner: an agent of the machine whose "+" opened it.
  const [creating, setCreating] = useState<AgentSettingsOwner | null>(null)
  useFleetRoster(true)

  const offers = useNewAgentOffers(props.agents)
  const query = props.query.trim()
  // Each machine's row offers New agent; the header stands in while no row is on screen.
  const rowsShown = open && (!query || props.agents.agents.some(agent => agentMatches(props.agents, agent, query)))

  // The badge counts what the section lists: hidden agents live in their own row below it.
  const listed = props.agents.agents.filter(agent => !hidden.includes(agentKey(agent))).length

  return (
    <section
      aria-label={t.simplifiedAgents.agents}
      className="portal-section"
      data-open={open}
      // Every non-pinned chat list of the redesigned sidebar lives here, so the section is the
      // classic sections' own unpin target (sessions-section.tsx): a chat carried out of Pinned
      // and let go on it is unpinned by the pinned list's own drag (HADESK-58, HADESK-167).
      data-session-unpin-drop=""
      data-slot="agent-section"
    >
      <PortalSectionHeader
        action={
          <>
            {props.headerAction}
            {!rowsShown && offers.length > 0 && <NewAgentFallback offers={offers} onPick={setCreating} />}
          </>
        }
        count={listed}
        label={t.simplifiedAgents.agents}
        onToggle={() => $portalAgentsSectionOpen.set(!open)}
        open={open}
      />
      {/* Hidden, not unmounted: the scroller owns restored position and reveal listeners. */}
      <div className="portal-section-body" hidden={!open}>
        <PortalAgentGroups
          {...props}
          newAgent={offers}
          onNewAgent={setCreating}
          onOpenAgent={agent => void openAgent(agent)}
          pending={pending}
          scrollRef={scroll}
        />
      </div>
      {creating && (
        <Suspense fallback={null}>
          <NewAgentDialog onClose={() => setCreating(null)} owner={creating} />
        </Suspense>
      )}
    </section>
  )
}
