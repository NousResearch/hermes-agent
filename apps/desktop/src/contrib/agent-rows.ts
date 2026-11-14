import { useMemo } from 'react'

import type { SubagentProgress, SubagentStatus, SubagentStreamEntry, SubagentStreamKind } from '@/store/subagents'

import { plainText } from './plain-text'
import type { $pluginRecords } from './plugins-store'
import { useContributions } from './react/use-contributions'
import type { Contribution } from './types'

/**
 * Agent rows — agents a plugin runs outside `delegate_task` (a workflow's
 * workers on the Kanban board, a coding harness and its subagents), counted by
 * the status bar's Agents item and listed on the Agents page beside Hermes's
 * own subagents.
 *
 *   data kind (`data`):  agents.rows  (AgentRowsContribution)
 *
 * The host owns the indicator and the page; a plugin only reports rows in the
 * page's own row shape and registers the same id again when they change (as
 * Kanban card badges do). The host validates and caps the rows and namespaces
 * them by plugin: no plugin can replace, nest under or hide another plugin's
 * rows or Hermes's own.
 */
export const AGENT_ROWS_AREA = 'agents.rows'

/** What the host accepts per plugin. Over-long text is shortened and long
 *  lists trimmed, never refused; rows past `perPlugin` are dropped. */
export const AGENT_ROWS_LIMITS = {
  fileChars: 300,
  files: 50,
  goalChars: 200,
  idChars: 160,
  modelChars: 120,
  perPlugin: 200,
  stream: 24,
  streamChars: 220,
  summaryChars: 600,
  toolChars: 96
} as const

/** One agent as a plugin reports it: the Agents page's own row shape. Times
 *  are epoch milliseconds. Every field is a standing compatibility promise. */
export interface PluginAgentRow {
  /** Stable within the plugin: the key a later registration updates. */
  id: string
  /** Another row of the same plugin to nest under (a subagent under its agent). */
  parentId?: null | string
  /** The row's title: what the agent works on. */
  goal: string
  status: SubagentStatus
  model?: string
  startedAt?: number
  updatedAt?: number
  durationSeconds?: number
  costUsd?: number
  inputTokens?: number
  outputTokens?: number
  toolCount?: number
  /** The tool it is running now (shown while `running`). */
  currentTool?: string
  filesRead?: readonly string[]
  filesWritten?: readonly string[]
  /** Latest entries first-to-last: thinking, tool, progress or summary lines. */
  stream?: readonly SubagentStreamEntry[]
  summary?: string
}

/** Payload of an `agents.rows` data contribution. */
export interface AgentRowsContribution {
  rows: readonly PluginAgentRow[]
}

const STATUSES = new Set<string>(['completed', 'failed', 'interrupted', 'queued', 'running'])
const STREAM_KINDS = new Set<string>(['progress', 'summary', 'thinking', 'tool'])

const amount = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined

const text = (value: unknown, maxChars: number): string | undefined => plainText(value, maxChars) ?? undefined

const paths = (value: unknown): string[] =>
  Array.isArray(value)
    ? value
        .map(item => plainText(item, AGENT_ROWS_LIMITS.fileChars))
        .filter((item): item is string => item !== null)
        .slice(0, AGENT_ROWS_LIMITS.files)
    : []

function streamOf(value: unknown, now: number): SubagentStreamEntry[] {
  if (!Array.isArray(value)) {
    return []
  }

  const entries: SubagentStreamEntry[] = []

  for (const raw of value.slice(-AGENT_ROWS_LIMITS.stream)) {
    if (!raw || typeof raw !== 'object') {
      continue
    }

    const { at, isError, kind, text: line } = raw as Record<string, unknown>
    const body = plainText(line, AGENT_ROWS_LIMITS.streamChars)

    if (!body || typeof kind !== 'string' || !STREAM_KINDS.has(kind)) {
      continue
    }

    entries.push({
      at: amount(at) ?? now,
      isError: isError === true || undefined,
      kind: kind as SubagentStreamKind,
      text: body
    })
  }

  return entries
}

/** One validated row in the page's shape, its ids namespaced by `source`;
 *  `parentId` is still the plugin's own (resolved once all its rows are known). */
function toRow(raw: unknown, source: string, now: number): null | SubagentProgress {
  if (!raw || typeof raw !== 'object') {
    return null
  }

  const value = raw as Record<string, unknown>
  const id = plainText(value.id, AGENT_ROWS_LIMITS.idChars)
  const goal = plainText(value.goal, AGENT_ROWS_LIMITS.goalChars)

  if (!id || !goal || typeof value.status !== 'string' || !STATUSES.has(value.status)) {
    return null
  }

  const startedAt = amount(value.startedAt) ?? now

  return {
    costUsd: amount(value.costUsd),
    currentTool: value.status === 'running' ? text(value.currentTool, AGENT_ROWS_LIMITS.toolChars) : undefined,
    durationSeconds: amount(value.durationSeconds),
    filesRead: paths(value.filesRead),
    filesWritten: paths(value.filesWritten),
    goal,
    id,
    inputTokens: amount(value.inputTokens),
    model: text(value.model, AGENT_ROWS_LIMITS.modelChars),
    outputTokens: amount(value.outputTokens),
    parentId: plainText(value.parentId, AGENT_ROWS_LIMITS.idChars),
    source,
    startedAt,
    status: value.status as SubagentStatus,
    stream: streamOf(value.stream, now),
    summary: text(value.summary, AGENT_ROWS_LIMITS.summaryChars),
    taskCount: 1,
    taskIndex: 0,
    toolCount: amount(value.toolCount),
    updatedAt: amount(value.updatedAt) ?? startedAt
  }
}

// Code-unit order, not localeCompare: the same in every locale.
const compare = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0)

const byProvenance = (a: Contribution, b: Contribution) =>
  compare(a.source ?? 'core', b.source ?? 'core') || compare(a.id, b.id)

const warned = new Set<string>()

/** Every plugin's rows, validated, capped per plugin and namespaced as
 *  `<source>#<id>` so they can never collide with Hermes's subagent ids or
 *  another plugin's. A row nests only under a row of its own plugin; a
 *  contribution whose `rows` is missing, not a list or throws contributes
 *  nothing. Pure over the area's contributions. */
export function resolvePluginAgentRows(
  contributions: readonly Contribution[],
  now: number = Date.now()
): SubagentProgress[] {
  const bySource = new Map<string, SubagentProgress[]>()

  for (const contribution of [...contributions].sort(byProvenance)) {
    const source = contribution.source ?? 'core'
    const rows = bySource.get(source) ?? []
    let list: readonly unknown[] = []

    try {
      const value = (contribution.data as Partial<AgentRowsContribution> | undefined)?.rows
      list = Array.isArray(value) ? value : []
    } catch (error) {
      if (!warned.has(contribution.id)) {
        warned.add(contribution.id)
        console.warn(`[${AGENT_ROWS_AREA}] ${contribution.id} rows could not be read; skipped`, error)
      }
    }

    for (const raw of list) {
      if (rows.length >= AGENT_ROWS_LIMITS.perPlugin) {
        break
      }

      const row = toRow(raw, source, now)

      if (row && !rows.some(existing => existing.id === row.id)) {
        rows.push(row)
      }
    }

    bySource.set(source, rows)
  }

  return [...bySource.entries()].flatMap(([source, rows]) => {
    const own = new Set(rows.map(row => row.id))

    return rows.map(row => ({
      ...row,
      id: `${source}#${row.id}`,
      parentId: row.parentId && row.parentId !== row.id && own.has(row.parentId) ? `${source}#${row.parentId}` : null
    }))
  })
}

/** The plugin rows of the Agents page and the status-bar count, live. */
export function usePluginAgentRows(): SubagentProgress[] {
  const contributions = useContributions(AGENT_ROWS_AREA)

  return useMemo(() => resolvePluginAgentRows(contributions), [contributions])
}

/** The name a plugin's rows are grouped under on the Agents page. */
export function agentRowsPluginName(source: string, records: ReturnType<typeof $pluginRecords.get>): string {
  const id = source.startsWith('plugin:') ? source.slice('plugin:'.length) : source

  return records[id]?.name || id
}
