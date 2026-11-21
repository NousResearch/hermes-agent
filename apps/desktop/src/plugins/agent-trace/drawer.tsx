/**
 * Inspection drawer — the right-hand slide-in that opens when a tree step is
 * clicked. Shows the exact arguments, per-call latency, context/token volume
 * and the raw output or error trace. Rendered INSIDE the page's relative
 * container (no portal): it overlays the tree, ESC or the close button
 * dismisses it. Subagent view adds the live child transcript (`subagent.tail`).
 */

import {
  Badge,
  Button,
  Codicon,
  compactNumber,
  host,
  ScrollArea,
  SegmentedControl,
  StatusDot,
  Tip,
  useQuery,
  useValue
} from '@hermes/plugin-sdk'
import { type ReactNode, useEffect, useState } from 'react'

import { statusLabel, useAgentTrace } from './i18n'
import { fetchSubagentTail, findSubagent, findTool, sessionById, traceKeys } from './trace-store'
import { $selectedStep, $trace } from './trace-store'
import type { SelectedStep, SessionTrace, SubagentNode, ToolCall } from './types'
import { isTerminalSubagent } from './types'
import { argsToText, CodeBlock, ContextGauge, fmtDurationS, fmtTs, MetaItem, shortId, SubagentStatusBadge } from './ui'

type DrawerTab = 'args' | 'output' | 'raw' | 'transcript'

function DrawerShell({
  title,
  icon,
  badge,
  onClose,
  children
}: {
  title: string
  icon: string
  badge?: ReactNode
  onClose: () => void
  children: ReactNode
}) {
  const k = useAgentTrace()
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose()
      }
    }

    document.addEventListener('keydown', onKey)

    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <aside
      aria-label={title}
      className="atrace-drawer absolute inset-y-0 right-0 z-20 flex w-full max-w-md flex-col border-l border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) shadow-xl"
      role="dialog"
    >
      <header className="flex items-center gap-2 border-b border-(--ui-stroke-tertiary) px-3 py-2.5">
        <Codicon className="shrink-0 text-(--ui-text-tertiary)" name={icon} size="0.9rem" />
        <h2 className="min-w-0 flex-1 truncate font-mono text-sm font-medium text-foreground">{title}</h2>
        {badge}
        <Button aria-label={k.drawerClose} onClick={onClose} size="icon-xs" variant="ghost">
          <Codicon name="close" size="0.8rem" />
        </Button>
      </header>
      <ScrollArea className="min-h-0 flex-1">
        <div className="flex flex-col gap-3 p-3">{children}</div>
      </ScrollArea>
    </aside>
  )
}

function MetaGrid({ children }: { children: ReactNode }) {
  return (
    <div className="grid grid-cols-2 gap-x-3 gap-y-2 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary) p-2.5">
      {children}
    </div>
  )
}

/** Output tab body — error trace, result text, or the history caveat. */
function ToolOutput({ tool }: { tool: ToolCall }) {
  const k = useAgentTrace()
  const output = tool.error ?? tool.resultText ?? tool.summary ?? null

  return (
    <>
      {tool.status === 'error' && (
        <div className="flex items-center gap-1.5 text-xs font-medium text-destructive">
          <Codicon name="error" size="0.8rem" />
          {k.errorLabel}
        </div>
      )}
      {output ? (
        <CodeBlock text={output} tone={tool.status === 'error' ? 'error' : undefined} />
      ) : (
        <p className="text-xs text-(--ui-text-quaternary)">
          {tool.origin === 'history' ? k.historyOutputNote : k.noOutput}
        </p>
      )}
      {tool.origin === 'history' && output && (
        <p className="text-[0.6875rem] text-(--ui-text-quaternary)">{k.historyOutputNote}</p>
      )}
    </>
  )
}

function ToolInspection({
  tool,
  session,
  sel
}: {
  tool: ToolCall
  session: SessionTrace
  sel: Extract<SelectedStep, { kind: 'tool' }>
}) {
  const k = useAgentTrace()
  const [tab, setTab] = useState<DrawerTab>('args')
  useEffect(() => setTab('args'), [sel.toolId])
  const argsText = argsToText(tool.args, tool.argsText)
  const usage = session.usage

  return (
    <DrawerShell
      badge={
        <Badge
          size="xs"
          variant={tool.status === 'error' ? 'destructive' : tool.status === 'running' ? 'default' : 'muted'}
        >
          {statusLabel(k, tool.status)}
        </Badge>
      }
      icon="tools"
      onClose={() => $selectedStep.set(null)}
      title={tool.name}
    >
      <MetaGrid>
        <MetaItem label={k.latency}>{fmtDurationS(tool.durationS)}</MetaItem>
        <MetaItem label={k.resultSize}>{tool.resultChars != null ? k.chars(tool.resultChars) : '—'}</MetaItem>
        <MetaItem label={k.started}>{fmtTs(tool.startedAt)}</MetaItem>
        <MetaItem label={k.ended}>{fmtTs(tool.endedAt)}</MetaItem>
        <MetaItem label={k.contextUsed}>
          <ContextGauge max={usage?.context_max} percent={usage?.context_percent} used={usage?.context_used} />
        </MetaItem>
        <MetaItem label={k.model}>{session.model ?? '—'}</MetaItem>
      </MetaGrid>
      <div className="flex items-center gap-2 text-[0.6875rem] text-(--ui-text-quaternary)">
        <Badge size="xs" variant="outline">
          {tool.origin === 'live' ? k.originLive : k.originHistory}
        </Badge>
        {tool.subagentId && <span className="font-mono">{shortId(tool.subagentId)}</span>}
        {tool.truncated && <span>{k.truncatedNote}</span>}
      </div>
      <SegmentedControl
        onChange={setTab}
        options={[
          { id: 'args', label: k.drawerArgs },
          { id: 'output', label: k.drawerOutput },
          { id: 'raw', label: k.drawerRaw }
        ]}
        sizing="content"
        value={tab}
      />
      {tab === 'args' &&
        (argsText ? <CodeBlock text={argsText} /> : <p className="text-xs text-(--ui-text-quaternary)">{k.noArgs}</p>)}
      {tab === 'output' && <ToolOutput tool={tool} />}
      {tab === 'raw' &&
        (typeof tool.raw === 'string' ? (
          <CodeBlock text={tool.raw} />
        ) : (
          <p className="text-xs text-(--ui-text-quaternary)">{k.noOutput}</p>
        ))}
    </DrawerShell>
  )
}

function SubagentInspection({
  node,
  session,
  sel
}: {
  node: SubagentNode
  session: SessionTrace
  sel: Extract<SelectedStep, { kind: 'subagent' }>
}) {
  const k = useAgentTrace()
  const [tab, setTab] = useState<DrawerTab>('output')
  useEffect(() => setTab('output'), [sel.subagentId])
  const profile = useValue(host.state.profile)
  const running = !isTerminalSubagent(node.status)

  const tail = useQuery({
    queryKey: traceKeys.tail(session.profile ?? profile, session.sessionId, node.id),
    queryFn: () => fetchSubagentTail(session.profile ?? profile, session.sessionId, node.id),
    refetchInterval: running ? 3_000 : false,
    retry: false
  })

  return (
    <DrawerShell
      badge={<SubagentStatusBadge label={statusLabel(k, node.status)} status={node.status} />}
      icon="hubot"
      onClose={() => $selectedStep.set(null)}
      title={node.goal || shortId(node.id)}
    >
      <MetaGrid>
        <MetaItem label={k.duration}>{fmtDurationS(node.durationS)}</MetaItem>
        <MetaItem label={k.model}>{node.model ?? '—'}</MetaItem>
        <MetaItem label={k.started}>{fmtTs(node.startedAt)}</MetaItem>
        <MetaItem label={k.ended}>{fmtTs(node.endedAt)}</MetaItem>
        <MetaItem label={k.tokensIn}>{compactNumber(node.inputTokens)}</MetaItem>
        <MetaItem label={k.tokensOut}>{compactNumber(node.outputTokens)}</MetaItem>
        <MetaItem label={k.tokensReasoning}>{compactNumber(node.reasoningTokens)}</MetaItem>
        <MetaItem label={k.apiCalls}>{node.apiCalls}</MetaItem>
        <MetaItem label={k.depth}>{node.depth}</MetaItem>
        <MetaItem label={k.toolCalls}>
          {node.toolCount}
          {node.lastTool ? ` · ${node.lastTool}` : ''}
        </MetaItem>
      </MetaGrid>
      {node.taskCount > 1 && (
        <p className="text-[0.6875rem] text-(--ui-text-tertiary)">{k.subagentTask(node.taskIndex, node.taskCount)}</p>
      )}
      {node.childSessionId && (
        <p className="font-mono text-[0.625rem] text-(--ui-text-quaternary)">child: {node.childSessionId}</p>
      )}
      {node.error && (
        <div className="flex items-start gap-1.5 rounded-md border border-destructive/30 bg-destructive/10 p-2 text-xs text-destructive">
          <Codicon className="mt-px shrink-0" name="error" size="0.8rem" />
          <span className="min-w-0 break-words">{node.error}</span>
        </div>
      )}
      <SegmentedControl
        onChange={setTab}
        options={[
          { id: 'output', label: k.drawerOutput },
          { id: 'transcript', label: k.drawerTranscript },
          { id: 'raw', label: k.drawerRaw }
        ]}
        sizing="content"
        value={tab}
      />
      {tab === 'output' && <SubagentOutput node={node} />}
      {tab === 'transcript' &&
        (tail.data?.available && tail.data.text ? (
          <CodeBlock text={tail.data.text} />
        ) : (
          <p className="text-xs text-(--ui-text-quaternary)">{tail.isPending ? '…' : k.transcriptUnavailable}</p>
        ))}
      {tab === 'raw' && <CodeBlock text={JSON.stringify(node, null, 2)} />}
    </DrawerShell>
  )
}

/** Output tab body for a child — summary, recent tool output, reasoning tail. */
function SubagentOutput({ node }: { node: SubagentNode }) {
  const k = useAgentTrace()

  return (
    <>
      {node.summary && <CodeBlock text={node.summary} />}
      {node.outputTail.length > 0 && (
        <div className="flex flex-col gap-1">
          <div className="text-[0.625rem] font-medium uppercase tracking-wide text-(--ui-text-quaternary)">
            {k.outputTail}
          </div>
          {node.outputTail.map((entry, index) => (
            <div className="flex items-start gap-1.5 text-[0.6875rem]" key={`${entry.tool}-${index}`}>
              <StatusDot className="mt-1 shrink-0" tone={entry.is_error ? 'bad' : 'muted'} />
              <span className="shrink-0 font-mono text-(--ui-text-tertiary)">{entry.tool}</span>
              <span className="min-w-0 break-words text-(--ui-text-quaternary)">{entry.preview}</span>
            </div>
          ))}
        </div>
      )}
      {node.thinkingTail && (
        <div className="flex flex-col gap-1">
          <div className="text-[0.625rem] font-medium uppercase tracking-wide text-(--ui-text-quaternary)">
            {k.thinking}
          </div>
          <CodeBlock text={node.thinkingTail} />
        </div>
      )}
      {!node.summary && node.outputTail.length === 0 && !node.thinkingTail && (
        <p className="text-xs text-(--ui-text-quaternary)">{k.noOutput}</p>
      )}
    </>
  )
}

function SessionInspection({ session }: { session: SessionTrace }) {
  const k = useAgentTrace()
  const usage = session.usage
  const tokens = (usage?.input ?? 0) + (usage?.output ?? 0)

  return (
    <DrawerShell
      badge={session.turnActive ? <Badge size="xs">{k.sessionTurnActive}</Badge> : undefined}
      icon="server-process"
      onClose={() => $selectedStep.set(null)}
      title={session.title || k.unknownSession}
    >
      <MetaGrid>
        <MetaItem label={k.model}>{session.model ?? '—'}</MetaItem>
        <MetaItem label="Provider">{session.provider ?? '—'}</MetaItem>
        <MetaItem label={k.statTokens}>{compactNumber(tokens)}</MetaItem>
        <MetaItem label={k.apiCalls}>{usage?.calls ?? 0}</MetaItem>
        <MetaItem label={k.statLatency}>{fmtDurationS(usage?.avg_latency_s)}</MetaItem>
        <MetaItem label={k.statContext}>
          <ContextGauge max={usage?.context_max} percent={usage?.context_percent} used={usage?.context_used} />
        </MetaItem>
        <MetaItem label={k.started}>{fmtTs(session.firstSeenAt)}</MetaItem>
        <MetaItem label={k.ended}>{fmtTs(session.lastActivityAt)}</MetaItem>
      </MetaGrid>
      <p className="font-mono text-[0.625rem] text-(--ui-text-quaternary)">
        {session.sessionId}
        {session.profile ? ` · ${session.profile}` : ''}
      </p>
      {session.lastError && (
        <div className="flex items-start gap-1.5 rounded-md border border-destructive/30 bg-destructive/10 p-2 text-xs text-destructive">
          <Codicon className="mt-px shrink-0" name="warning" size="0.8rem" />
          <span className="min-w-0 break-words">{session.lastError.message}</span>
        </div>
      )}
      <div className="flex items-center gap-3 text-xs text-(--ui-text-tertiary)">
        <Tip label={k.statToolsTip}>
          <span className="inline-flex items-center gap-1">
            <Codicon name="tools" size="0.8rem" />
            {k.sessionTools(session.tools.length)}
          </span>
        </Tip>
        <Tip label={k.statAgentsTip}>
          <span className="inline-flex items-center gap-1">
            <Codicon name="hubot" size="0.8rem" />
            {k.sessionAgents(session.subagents.length)}
          </span>
        </Tip>
      </div>
    </DrawerShell>
  )
}

/** The drawer itself — null when nothing is selected. Mount inside a
 *  `relative` container; it overlays the right edge. */
export function InspectionDrawer() {
  const sel = useValue($selectedStep)
  const state = useValue($trace)

  if (!sel) {
    return null
  }

  const session = sessionById(state, sel.sessionId)

  if (!session) {
    return null
  }

  if (sel.kind === 'tool') {
    const tool = findTool(state, sel)

    return tool ? <ToolInspection sel={sel} session={session} tool={tool} /> : null
  }

  if (sel.kind === 'subagent') {
    const node = findSubagent(state, sel)

    return node ? <SubagentInspection node={node} sel={sel} session={session} /> : null
  }

  return <SessionInspection session={session} />
}
