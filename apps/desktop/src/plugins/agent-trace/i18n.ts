/**
 * Plugin-scoped i18n for AgentTrace — bundles shipped under the plugin id via
 * ctx.i18n.register, never touching core catalogs. `useAgentTrace()` binds
 * the message SHAPE so components keep typed access (kanban's pattern).
 * English-only for now; other locales fall back to `en` by SDK resolution.
 */

import { type PluginLocaleBundles, type PluginTranslate, usePluginI18n } from '@hermes/plugin-sdk'
import { useMemo } from 'react'

type AgentTraceMessages = {
  nav: string
  openCommand: string
  countTip: (agents: number, tools: number) => string
  title: string
  subtitle: string
  scopeFocused: string
  scopeAll: string
  filterPlaceholder: string
  refresh: string
  pause: string
  resume: string
  live: string
  paused: string
  emptyTitle: string
  emptyDescription: string
  noMatch: string
  statAgents: string
  statTools: string
  statContext: string
  statTokens: string
  statLatency: string
  statAgentsTip: string
  statToolsTip: string
  statContextTip: string
  statTokensTip: string
  statLatencyTip: string
  sessionTools: (n: number) => string
  sessionAgents: (n: number) => string
  toolCalls: string
  sessionTurnActive: string
  sessionLastError: string
  generating: (name: string) => string
  openFullView: string
  paneTitle: string
  paneEmpty: string
  paneFocused: string
  paneOther: string
  drawerClose: string
  drawerArgs: string
  drawerOutput: string
  drawerRaw: string
  drawerTranscript: string
  latency: string
  started: string
  ended: string
  duration: string
  resultSize: string
  contextUsed: string
  tokensIn: string
  tokensOut: string
  tokensReasoning: string
  apiCalls: string
  model: string
  depth: string
  goal: string
  summary: string
  outputTail: string
  thinking: string
  childTranscript: string
  transcriptUnavailable: string
  historyOutputNote: string
  truncatedNote: string
  noArgs: string
  noOutput: string
  errorLabel: string
  originLive: string
  originHistory: string
  chars: (n: number) => string
  status: Record<
    | 'queued'
    | 'running'
    | 'completed'
    | 'failed'
    | 'error'
    | 'timeout'
    | 'interrupted'
    | 'done'
    | 'generating'
    | 'spawn_requested',
    string
  >
  unknownSession: string
  subagentTask: (index: number, count: number) => string
  openSession: string
  toolProgress: (value: number, total: number) => string
}

export const en: AgentTraceMessages = {
  nav: 'AgentTrace',
  openCommand: 'AgentTrace: Open execution trace',
  countTip: (agents, tools) =>
    `${agents} subagent${agents === 1 ? '' : 's'} active · ${tools} tool call${tools === 1 ? '' : 's'} running`,
  title: 'AgentTrace',
  subtitle: 'Live execution tree — delegated subagents and tool calls, grouped by session.',
  scopeFocused: 'Focused session',
  scopeAll: 'All sessions',
  filterPlaceholder: 'Filter tools, agents, sessions…',
  refresh: 'Refresh snapshots',
  pause: 'Pause live updates',
  resume: 'Resume live updates',
  live: 'Live',
  paused: 'Paused',
  emptyTitle: 'No execution traced yet',
  emptyDescription: 'Send a message or delegate a task — tool calls and subagents appear here as they run.',
  noMatch: 'Nothing matches the current filter.',
  statAgents: 'Active agents',
  statTools: 'Running tools',
  statContext: 'Context',
  statTokens: 'Tokens',
  statLatency: 'Avg latency',
  statAgentsTip: 'Delegated children not in a terminal state, across traced sessions.',
  statToolsTip: 'Tool calls started but not completed yet.',
  statContextTip: 'Context window consumption of the selected session (streamed usage).',
  statTokensTip: 'Cumulative input + output tokens of the selected session.',
  statLatencyTip: 'Rolling average API latency of the selected session.',
  sessionTools: n => `${n} tool${n === 1 ? '' : 's'}`,
  sessionAgents: n => `${n} agent${n === 1 ? '' : 's'}`,
  toolCalls: 'Tool calls',
  sessionTurnActive: 'Turn in progress',
  sessionLastError: 'Last error',
  generating: name => `Generating ${name} arguments…`,
  openFullView: 'Open full view',
  paneTitle: 'AgentTrace',
  paneEmpty: 'No activity yet.',
  paneFocused: 'Focused session',
  paneOther: 'Other sessions',
  drawerClose: 'Close inspection',
  drawerArgs: 'Arguments',
  drawerOutput: 'Output',
  drawerRaw: 'Raw',
  drawerTranscript: 'Transcript',
  latency: 'Latency',
  started: 'Started',
  ended: 'Ended',
  duration: 'Duration',
  resultSize: 'Result size',
  contextUsed: 'Context used',
  tokensIn: 'Input tokens',
  tokensOut: 'Output tokens',
  tokensReasoning: 'Reasoning tokens',
  apiCalls: 'API calls',
  model: 'Model',
  depth: 'Depth',
  goal: 'Goal',
  summary: 'Summary',
  outputTail: 'Recent tool output',
  thinking: 'Reasoning tail',
  childTranscript: 'Child transcript',
  transcriptUnavailable: 'No live transcript for this child yet.',
  historyOutputNote:
    'Persisted row — the gateway keeps full output only for edit/clarify tools; arguments and preview are shown as stored.',
  truncatedNote: 'Content truncated for display.',
  noArgs: 'No arguments recorded.',
  noOutput: 'No output recorded.',
  errorLabel: 'Error',
  originLive: 'Live',
  originHistory: 'History',
  chars: n => `${n.toLocaleString()} chars`,
  status: {
    queued: 'Queued',
    running: 'Running',
    completed: 'Completed',
    failed: 'Failed',
    error: 'Error',
    timeout: 'Timed out',
    interrupted: 'Interrupted',
    done: 'Done',
    generating: 'Generating',
    spawn_requested: 'Spawn requested'
  },
  unknownSession: 'Untitled session',
  subagentTask: (index, count) => `Task ${index + 1}/${count}`,
  openSession: 'Open this session',
  toolProgress: (value, total) => `${value}/${total}`
}

export const AGENT_TRACE_LOCALES: PluginLocaleBundles = { en }

// Bind the message SHAPE to a plugin translator (kanban's pattern): string
// leaves resolve now, function leaves forward their args through t(path, …).
type Bound<T> = {
  [K in keyof T]: T[K] extends (...args: infer A) => string
    ? (...args: A) => string
    : T[K] extends object
      ? Bound<T[K]>
      : string
}

function bind<T extends object>(t: PluginTranslate, template: T, prefix = ''): Bound<T> {
  const out = {} as Record<string, unknown>

  for (const [key, value] of Object.entries(template)) {
    const path = prefix ? `${prefix}.${key}` : key
    out[key] =
      typeof value === 'function'
        ? (...args: unknown[]) => t(path, ...args)
        : value && typeof value === 'object'
          ? bind(t, value as object, path)
          : t(path)
  }

  return out as Bound<T>
}

export type AgentTraceText = Bound<AgentTraceMessages>

/** The AgentTrace strings for the active locale — one hook every component reads. */
export function useAgentTrace(): AgentTraceText {
  const t = usePluginI18n('agent-trace')

  return useMemo(() => bind(t, en), [t])
}

/** Status label with a raw-id fallback for unknown backend statuses. */
export function statusLabel(k: AgentTraceText, status: string): string {
  return k.status[status as keyof AgentTraceText['status']] ?? status
}
