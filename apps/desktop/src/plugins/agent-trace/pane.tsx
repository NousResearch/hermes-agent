/**
 * The right-docked AgentTrace pane — a compact live readout that stays useful
 * beside the conversation: the focused session's running tools and active
 * children, then any other session with activity. Clicking a row parks the
 * selection and opens the full page with its inspection drawer.
 */

import { Button, cn, Codicon, host, StatusDot, Tip, useQuery, useValue } from '@hermes/plugin-sdk'
import { useEffect } from 'react'

import { statusLabel, useAgentTrace } from './i18n'
import {
  $selectedStep,
  $trace,
  activeSessions,
  fetchDelegationStatus,
  hasLiveActivity,
  mergeDelegationStatus,
  sessionById,
  traceKeys
} from './trace-store'
import type { SelectedStep, SessionTrace } from './types'
import { isTerminalSubagent } from './types'
import { fmtDurationS, shortId, SubagentStatusBadge, toolTone } from './ui'

function openStep(sel: SelectedStep): void {
  $selectedStep.set(sel)
  host.navigate('/agent-trace')
}

function PaneSession({ session, focused }: { session: SessionTrace; focused: boolean }) {
  const k = useAgentTrace()
  const runningTools = session.tools.filter(t => t.status === 'running' || t.status === 'generating')
  const activeChildren = session.subagents.filter(n => !isTerminalSubagent(n.status))

  const recentErrors = session.subagents
    .filter(n => n.status === 'failed' || n.status === 'error' || n.status === 'timeout')
    .slice(-2)

  if (runningTools.length === 0 && activeChildren.length === 0 && recentErrors.length === 0 && !session.turnActive) {
    return null
  }

  return (
    <div className="min-w-0">
      <button
        className="flex w-full min-w-0 items-center gap-1.5 rounded px-1 py-1 text-left transition-colors hover:bg-(--ui-bg-quinary)"
        onClick={() => openStep({ kind: 'session', sessionId: session.sessionId })}
        type="button"
      >
        <StatusDot className={cn(session.turnActive && 'animate-pulse')} tone={session.turnActive ? 'good' : 'muted'} />
        <span className="min-w-0 flex-1 truncate text-[0.6875rem] font-medium text-foreground">
          {session.title || k.unknownSession}
        </span>
        {!focused && session.profile && (
          <span className="shrink-0 text-[0.625rem] text-(--ui-text-quaternary)">{session.profile}</span>
        )}
        <span className="shrink-0 font-mono text-[0.625rem] text-(--ui-text-quaternary)">
          {shortId(session.sessionId)}
        </span>
      </button>
      <div className="atrace-children flex flex-col gap-px">
        {activeChildren.map(node => (
          <button
            className="flex w-full min-w-0 items-center gap-1.5 rounded px-1 py-0.5 text-left transition-colors hover:bg-(--ui-bg-quinary)"
            key={node.id}
            onClick={() => openStep({ kind: 'subagent', sessionId: session.sessionId, subagentId: node.id })}
            type="button"
          >
            <Codicon className="shrink-0 text-primary" name="hubot" size="0.7rem" />
            <span className="min-w-0 flex-1 truncate text-[0.6875rem] text-(--ui-text-secondary)">
              {node.goal || shortId(node.id)}
            </span>
            <SubagentStatusBadge label={statusLabel(k, node.status)} status={node.status} />
          </button>
        ))}
        {runningTools.map(tool => (
          <button
            className="flex w-full min-w-0 items-center gap-1.5 rounded px-1 py-0.5 text-left transition-colors hover:bg-(--ui-bg-quinary)"
            key={tool.id}
            onClick={() => openStep({ kind: 'tool', sessionId: session.sessionId, toolId: tool.id })}
            type="button"
          >
            <StatusDot className="shrink-0 animate-pulse" tone={toolTone(tool.status)} />
            <span className="shrink-0 font-mono text-[0.6875rem] text-foreground">{tool.name}</span>
            <span className="min-w-0 flex-1 truncate text-[0.625rem] text-(--ui-text-quaternary)">
              {tool.preview ?? ''}
            </span>
            <span className="shrink-0 tabular-nums text-[0.625rem] text-(--ui-text-quaternary)">
              {tool.startedAt != null ? fmtDurationS(Date.now() / 1000 - tool.startedAt) : ''}
            </span>
          </button>
        ))}
        {recentErrors.map(node => (
          <Tip key={node.id} label={node.error ?? node.goal}>
            <button
              className="flex w-full min-w-0 items-center gap-1.5 rounded px-1 py-0.5 text-left transition-colors hover:bg-(--ui-bg-quinary)"
              onClick={() => openStep({ kind: 'subagent', sessionId: session.sessionId, subagentId: node.id })}
              type="button"
            >
              <Codicon className="shrink-0 text-destructive" name="error" size="0.7rem" />
              <span className="min-w-0 flex-1 truncate text-[0.6875rem] text-destructive">
                {node.goal || shortId(node.id)}
              </span>
            </button>
          </Tip>
        ))}
      </div>
    </div>
  )
}

export function TracePane() {
  const k = useAgentTrace()
  const state = useValue($trace)
  const focusedId = useValue(host.state.focusedSessionId)
  const profile = useValue(host.state.profile)

  // Slow heartbeat so the pane is right even when no frame arrived yet.
  const delegation = useQuery({
    queryKey: traceKeys.delegation(profile),
    queryFn: () => fetchDelegationStatus(profile),
    refetchInterval: 15_000,
    retry: false
  })

  useEffect(() => {
    if (delegation.data) {
      $trace.set(mergeDelegationStatus($trace.get(), delegation.data, Date.now() / 1000))
    }
  }, [delegation.data])

  const focused = sessionById(state, focusedId)
  const focusedLive = focused && hasLiveActivity(focused) ? focused : null
  const others = activeSessions(state).filter(s => s.sessionId !== focusedId && hasLiveActivity(s))
  const hasContent = Boolean(focusedLive) || others.length > 0

  return (
    <div className="flex h-full min-w-0 flex-col text-sm">
      <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto p-2">
        {!hasContent && <p className="px-1 py-2 text-xs text-(--ui-text-quaternary)">{k.paneEmpty}</p>}
        {focusedLive && (
          <>
            <div className="px-1 text-[0.625rem] font-medium uppercase tracking-wide text-(--ui-text-quaternary)">
              {k.paneFocused}
            </div>
            <PaneSession focused session={focusedLive} />
          </>
        )}
        {others.length > 0 && (
          <>
            <div className="px-1 text-[0.625rem] font-medium uppercase tracking-wide text-(--ui-text-quaternary)">
              {k.paneOther}
            </div>
            {others.map(session => (
              <PaneSession focused={false} key={session.sessionId} session={session} />
            ))}
          </>
        )}
      </div>
      <div className="border-t border-(--ui-stroke-tertiary) p-2">
        <Button className="w-full" onClick={() => host.navigate('/agent-trace')} size="sm" variant="outline">
          <Codicon name="pulse" size="0.8rem" />
          {k.openFullView}
        </Button>
      </div>
    </div>
  )
}
