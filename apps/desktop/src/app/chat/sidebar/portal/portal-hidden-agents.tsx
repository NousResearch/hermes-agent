import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { RowButton } from '@/components/ui/row-button'
import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { ChevronDown, ChevronRight, EyeOff } from '@/lib/icons'
import { coarseElapsed } from '@/lib/time'
import { $hiddenSidebarAgentsExpanded, setSidebarAgentHidden } from '@/store/agent-sidebar-preferences'

import { useHiddenAgentCopy } from '../hidden-agent-copy'
import { SimplifiedBotChatRow } from '../simplified-bot-chat-row'
import { type AgentSidebarAttention, attentionRank, useAgentSidebarAttention } from '../use-agent-sidebar-attention'
import { agentOwner, type SimplifiedAgent } from '../use-simplified-agents'

interface PortalHiddenAgentsProps {
  /** The hidden agents; mount this only when there are some. */
  agents: SimplifiedAgent[]
  /** Every agent and chat the sidebar knows, so a signal is claimed by its one owner. */
  allAgents: SimplifiedAgent[]
  allSessions: SessionInfo[]
  onResumeSession: (id: string, row?: SessionInfo) => void
}

function AttentionDot({ attention }: { attention: AgentSidebarAttention }) {
  const c = useHiddenAgentCopy()

  return (
    <span aria-label={c[attention.kind]} className="portal-hidden-dot" data-attention={attention.kind} role="img" />
  )
}

export function PortalHiddenAgents({ agents, allAgents, allSessions, onResumeSession }: PortalHiddenAgentsProps) {
  const c = useHiddenAgentCopy()
  const { t } = useI18n()
  const expanded = useStore($hiddenSidebarAgentsExpanded)
  const attention = useAgentSidebarAttention(allAgents, allSessions)
  const [now, setNow] = useState(Date.now)

  const sorted = [...agents].sort((a, b) => attentionRank(attention.get(a.id)) - attentionRank(attention.get(b.id)))

  const summary = sorted.length ? attention.get(sorted[0].id) : undefined
  const ticking = expanded && Boolean(summary)

  useEffect(() => {
    if (!ticking) {
      return
    }

    setNow(Date.now())
    const timer = window.setInterval(() => setNow(Date.now()), 60_000)

    return () => window.clearInterval(timer)
  }, [ticking])

  function description(status: AgentSidebarAttention) {
    if (!status.since) {
      return c[status.kind]
    }

    const { unit, value } = coarseElapsed(Math.max(0, now - status.since))
    const units = { day: t.sidebar.row.ageDay, hour: t.sidebar.row.ageHour, minute: t.sidebar.row.ageMin }
    const age = unit === 'second' ? t.sidebar.row.ageNow : `${value}${units[unit]}`

    return `${c[status.kind]} · ${age}`
  }

  return (
    <section aria-label={c.hiddenAgents} className="portal-hidden-agents">
      <RowButton
        aria-expanded={expanded}
        className="portal-hidden-heading"
        onClick={() => $hiddenSidebarAgentsExpanded.set(!expanded)}
      >
        <EyeOff aria-hidden />
        <span className="min-w-0 flex-1 truncate">{c.hiddenAgents}</span>
        {summary && <AttentionDot attention={summary} />}
        <span className="portal-hidden-count">{agents.length}</span>
        {expanded ? <ChevronDown aria-hidden /> : <ChevronRight aria-hidden />}
      </RowButton>
      {expanded && (
        <div className="portal-hidden-rows">
          {sorted.map(agent => {
            const status = attention.get(agent.id)

            return (
              <div className="portal-hidden-row" data-attention={status?.kind} key={agent.id}>
                <SimplifiedBotChatRow
                  agent={agent}
                  appearance="hidden"
                  onOpen={status?.session ? () => onResumeSession(status.session!.id, status.session) : undefined}
                  summary={status && description(status)}
                />
                {status && <AttentionDot attention={status} />}
                <Button
                  aria-label={`${c.show}: ${agent.label} · ${agent.gateway}`}
                  className="portal-hidden-show"
                  onClick={() => setSidebarAgentHidden(agentOwner(agent), false)}
                  size="sm"
                  variant="outline"
                >
                  {c.show}
                </Button>
              </div>
            )
          })}
        </div>
      )}
    </section>
  )
}
