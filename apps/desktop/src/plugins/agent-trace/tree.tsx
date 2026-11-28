/**
 * The execution tree — sessions → delegated subagents (recursive) → tool
 * calls. Rows are buttons: clicking one opens the inspection drawer via the
 * caller's `onSelect`. Filter matches force expansion so a search never hides
 * its own hits. Local expansion state only — the trace data itself lives in
 * the store atoms.
 */

import { cn, Codicon, compactNumber, relativeTime, StatusDot, Tip } from '@hermes/plugin-sdk'
import { useState } from 'react'

import { statusLabel, useAgentTrace } from './i18n'
import type { SelectedStep, SessionTrace, SubagentNode, ToolCall } from './types'
import { isTerminalSubagent } from './types'
import { ContextGauge, fmtDurationS, shortId, SubagentStatusBadge, toolTone } from './ui'

function toolMatches(tool: ToolCall, filter: string): boolean {
  return !filter || tool.name.toLowerCase().includes(filter) || (tool.preview ?? '').toLowerCase().includes(filter)
}

function nodeMatches(node: SubagentNode, filter: string): boolean {
  return (
    !filter ||
    node.goal.toLowerCase().includes(filter) ||
    (node.model ?? '').toLowerCase().includes(filter) ||
    (node.lastTool ?? '').toLowerCase().includes(filter) ||
    node.tools.some(t => toolMatches(t, filter))
  )
}

function sessionMatches(session: SessionTrace, filter: string): boolean {
  return (
    !filter ||
    (session.title ?? '').toLowerCase().includes(filter) ||
    session.sessionId.toLowerCase().includes(filter) ||
    (session.model ?? '').toLowerCase().includes(filter) ||
    session.tools.some(t => toolMatches(t, filter)) ||
    session.subagents.some(n => nodeMatches(n, filter))
  )
}

function Caret({ expanded, onToggle, label }: { expanded: boolean; onToggle: () => void; label: string }) {
  return (
    <button
      aria-expanded={expanded}
      aria-label={label}
      className="grid size-4 shrink-0 place-items-center rounded text-(--ui-text-quaternary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground"
      onClick={event => {
        event.stopPropagation()
        onToggle()
      }}
      type="button"
    >
      <Codicon name={expanded ? 'chevron-down' : 'chevron-right'} size="0.7rem" />
    </button>
  )
}

function ToolRow({
  tool,
  sessionId,
  subagentId,
  selected,
  onSelect,
  depth
}: {
  tool: ToolCall
  sessionId: string
  subagentId?: string | null
  selected: SelectedStep | null
  onSelect: (sel: SelectedStep) => void
  depth: number
}) {
  const k = useAgentTrace()
  const active = selected?.kind === 'tool' && selected.toolId === tool.id && selected.sessionId === sessionId
  const running = tool.status === 'running' || tool.status === 'generating'

  return (
    <button
      className={cn(
        'flex w-full min-w-0 items-center gap-2 rounded px-1.5 py-1 text-left transition-colors hover:bg-(--ui-bg-quinary)',
        active && 'bg-(--ui-bg-quinary) ring-1 ring-(--ui-stroke-secondary)'
      )}
      onClick={() => onSelect({ kind: 'tool', sessionId, toolId: tool.id, subagentId: subagentId ?? null })}
      style={{ paddingLeft: `${depth * 0.875 + 0.375}rem` }}
      type="button"
    >
      <StatusDot className={cn(running && 'animate-pulse')} tone={toolTone(tool.status)} />
      <span className="shrink-0 font-mono text-[0.7rem] text-foreground">{tool.name}</span>
      {tool.labels?.[0] && (
        <Tip label={tool.labels[0].text}>
          <span className="shrink-0 text-[0.65rem] text-(--ui-text-quaternary)">{tool.labels[0].emoji}</span>
        </Tip>
      )}
      <span className="min-w-0 flex-1 truncate text-[0.6875rem] text-(--ui-text-quaternary)">{tool.preview ?? ''}</span>
      {tool.status === 'error' && <Codicon className="shrink-0 text-destructive" name="error" size="0.7rem" />}
      {tool.progress && tool.progress.total != null && (
        <span className="shrink-0 tabular-nums text-[0.65rem] text-(--ui-text-tertiary)">
          {k.toolProgress(tool.progress.value, tool.progress.total)}
        </span>
      )}
      <span className="shrink-0 tabular-nums text-[0.65rem] text-(--ui-text-quaternary)">
        {running ? statusLabel(k, tool.status) : fmtDurationS(tool.durationS)}
      </span>
    </button>
  )
}

function SubagentBlock({
  node,
  session,
  filter,
  selected,
  onSelect,
  depth
}: {
  node: SubagentNode
  session: SessionTrace
  filter: string
  selected: SelectedStep | null
  onSelect: (sel: SelectedStep) => void
  depth: number
}) {
  const k = useAgentTrace()
  const [expanded, setExpanded] = useState(true)
  const children = session.subagents.filter(n => n.parentId === node.id)
  const open = expanded || Boolean(filter)

  const active =
    selected?.kind === 'subagent' && selected.subagentId === node.id && selected.sessionId === session.sessionId

  const running = !isTerminalSubagent(node.status)
  const tokens = node.inputTokens + node.outputTokens
  const visibleTools = node.tools.filter(t => toolMatches(t, filter))

  return (
    <div className="min-w-0">
      <div
        className={cn(
          'flex w-full min-w-0 items-center gap-1.5 rounded px-1.5 py-1 transition-colors hover:bg-(--ui-bg-quinary)',
          active && 'bg-(--ui-bg-quinary) ring-1 ring-(--ui-stroke-secondary)'
        )}
        style={{ paddingLeft: `${depth * 0.875 + 0.375}rem` }}
      >
        <Caret expanded={open} label={node.goal} onToggle={() => setExpanded(v => !v)} />
        <button
          className="flex min-w-0 flex-1 items-center gap-1.5 text-left"
          onClick={() => onSelect({ kind: 'subagent', sessionId: session.sessionId, subagentId: node.id })}
          type="button"
        >
          <Codicon
            className={cn('shrink-0', running ? 'text-primary' : 'text-(--ui-text-quaternary)')}
            name="hubot"
            size="0.8rem"
          />
          <span className="min-w-0 flex-1 truncate text-xs text-foreground">{node.goal || shortId(node.id)}</span>
          <SubagentStatusBadge label={statusLabel(k, node.status)} status={node.status} />
          {node.taskCount > 1 && (
            <span className="shrink-0 text-[0.625rem] tabular-nums text-(--ui-text-quaternary)">
              {k.subagentTask(node.taskIndex, node.taskCount)}
            </span>
          )}
          {node.model && (
            <span className="hidden shrink-0 font-mono text-[0.625rem] text-(--ui-text-quaternary) xl:inline">
              {node.model}
            </span>
          )}
          {tokens > 0 && (
            <Tip
              label={`${k.tokensIn}: ${compactNumber(node.inputTokens)} · ${k.tokensOut}: ${compactNumber(node.outputTokens)}`}
            >
              <span className="shrink-0 tabular-nums text-[0.625rem] text-(--ui-text-tertiary)">
                {compactNumber(tokens)} tok
              </span>
            </Tip>
          )}
          <span className="shrink-0 tabular-nums text-[0.625rem] text-(--ui-text-quaternary)">
            {running ? k.sessionTools(node.toolCount) : fmtDurationS(node.durationS)}
          </span>
        </button>
      </div>
      {open && (children.length > 0 || visibleTools.length > 0) && (
        <div className="atrace-children">
          {children.map(child => (
            <SubagentBlock
              depth={depth + 1}
              filter={filter}
              key={child.id}
              node={child}
              onSelect={onSelect}
              selected={selected}
              session={session}
            />
          ))}
          {visibleTools.map(tool => (
            <ToolRow
              depth={depth + 1}
              key={tool.id}
              onSelect={onSelect}
              selected={selected}
              sessionId={session.sessionId}
              subagentId={node.id}
              tool={tool}
            />
          ))}
        </div>
      )}
    </div>
  )
}

/** The header's right-hand meta cluster — its own unit to keep SessionBlock
 *  under the code-health complexity cap. */
function SessionHeaderMeta({ session, showProfile }: { session: SessionTrace; showProfile: boolean }) {
  const k = useAgentTrace()
  const usage = session.usage
  const tokens = (usage?.input ?? 0) + (usage?.output ?? 0)

  return (
    <>
      {showProfile && session.profile && (
        <span className="shrink-0 rounded bg-(--ui-bg-quinary) px-1 py-px text-[0.625rem] text-(--ui-text-tertiary)">
          {session.profile}
        </span>
      )}
      {session.model && (
        <span className="hidden shrink-0 font-mono text-[0.625rem] text-(--ui-text-quaternary) lg:inline">
          {session.model}
        </span>
      )}
      <span className="shrink-0 text-[0.625rem] tabular-nums text-(--ui-text-tertiary)">
        {k.sessionTools(session.tools.length)}
      </span>
      {session.subagents.length > 0 && (
        <span className="shrink-0 text-[0.625rem] tabular-nums text-(--ui-text-tertiary)">
          {k.sessionAgents(session.subagents.length)}
        </span>
      )}
      {tokens > 0 && (
        <span className="hidden shrink-0 tabular-nums text-[0.625rem] text-(--ui-text-tertiary) md:inline">
          {compactNumber(tokens)} tok
        </span>
      )}
      <ContextGauge
        className="hidden shrink-0 md:inline-flex"
        max={usage?.context_max}
        percent={usage?.context_percent}
        used={usage?.context_used}
      />
      <span className="shrink-0 text-[0.625rem] text-(--ui-text-quaternary)">
        {relativeTime(session.lastActivityAt * 1000)}
      </span>
    </>
  )
}

function SessionBlock({
  session,
  filter,
  selected,
  onSelect,
  showProfile
}: {
  session: SessionTrace
  filter: string
  selected: SelectedStep | null
  onSelect: (sel: SelectedStep) => void
  showProfile: boolean
}) {
  const k = useAgentTrace()
  const [expanded, setExpanded] = useState(true)
  const open = expanded || Boolean(filter)
  const active = selected?.kind === 'session' && selected.sessionId === session.sessionId
  const roots = session.subagents.filter(n => !n.parentId || !session.subagents.some(p => p.id === n.parentId))
  const visibleRoots = roots.filter(n => nodeMatches(n, filter))
  const visibleTools = session.tools.filter(t => toolMatches(t, filter))

  const sortedTools = [...visibleTools].sort(
    (a, b) => (a.startedAt ?? a.endedAt ?? 0) - (b.startedAt ?? b.endedAt ?? 0)
  )

  return (
    <section className="min-w-0 rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-elevated)">
      <div
        className={cn(
          'flex w-full min-w-0 items-center gap-2 rounded-t-lg px-2 py-1.5 transition-colors hover:bg-(--ui-bg-quinary)',
          active && 'bg-(--ui-bg-quinary)'
        )}
      >
        <Caret expanded={open} label={session.title ?? session.sessionId} onToggle={() => setExpanded(v => !v)} />
        <button
          className="flex min-w-0 flex-1 items-center gap-2 text-left"
          onClick={() => onSelect({ kind: 'session', sessionId: session.sessionId })}
          type="button"
        >
          <StatusDot
            className={cn(session.turnActive && 'animate-pulse')}
            tone={session.turnActive ? 'good' : 'muted'}
          />
          <span className="min-w-0 flex-1 truncate text-xs font-medium text-foreground">
            {session.title || k.unknownSession}{' '}
            <span className="font-mono text-[0.625rem] font-normal text-(--ui-text-quaternary)">
              {shortId(session.sessionId)}
            </span>
          </span>
          <SessionHeaderMeta session={session} showProfile={showProfile} />
        </button>
      </div>
      {session.lastError && (
        <Tip label={session.lastError.message}>
          <div className="flex items-center gap-1.5 border-t border-(--ui-stroke-tertiary) px-3 py-1 text-[0.6875rem] text-destructive">
            <Codicon name="warning" size="0.7rem" />
            <span className="truncate">{session.lastError.message}</span>
          </div>
        </Tip>
      )}
      {open && (
        <div className="flex flex-col gap-px border-t border-(--ui-stroke-tertiary) p-1.5">
          {session.generating && (
            <div className="flex items-center gap-2 px-1.5 py-1 text-[0.6875rem] text-(--ui-text-tertiary)">
              <Codicon name="loading" size="0.7rem" spinning />
              {k.generating(session.generating.name)}
            </div>
          )}
          {visibleRoots.map(node => (
            <SubagentBlock
              depth={0}
              filter={filter}
              key={node.id}
              node={node}
              onSelect={onSelect}
              selected={selected}
              session={session}
            />
          ))}
          {sortedTools.map(tool => (
            <ToolRow
              depth={0}
              key={tool.id}
              onSelect={onSelect}
              selected={selected}
              sessionId={session.sessionId}
              tool={tool}
            />
          ))}
          {visibleRoots.length === 0 && sortedTools.length === 0 && !session.generating && (
            <div className="px-1.5 py-1 text-[0.6875rem] text-(--ui-text-quaternary)">
              {filter ? k.noMatch : k.emptyDescription}
            </div>
          )}
        </div>
      )}
    </section>
  )
}

export function TraceTree({
  sessions,
  filter,
  selected,
  onSelect,
  showProfile = false
}: {
  sessions: SessionTrace[]
  filter: string
  selected: SelectedStep | null
  onSelect: (sel: SelectedStep) => void
  showProfile?: boolean
}) {
  const lowered = filter.trim().toLowerCase()
  const visible = sessions.filter(s => sessionMatches(s, lowered))

  return (
    <div className="flex flex-col gap-2">
      {visible.map(session => (
        <SessionBlock
          filter={lowered}
          key={session.sessionId}
          onSelect={onSelect}
          selected={selected}
          session={session}
          showProfile={showProfile}
        />
      ))}
    </div>
  )
}
