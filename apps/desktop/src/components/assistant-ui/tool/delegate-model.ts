import { firstStringField, normalize } from '@/lib/text'
import type { SubagentProgress, SubagentStatus } from '@/store/subagents'

import { numberValue, parseMaybeObject } from './fallback-model'

/**
 * A delegation runs somewhere the transcript can't see: the tool call carries
 * the goals it dispatched, the subagent store carries what those children are
 * actually doing, and the tool result carries how they finished. One row is
 * all three of those views of the same child.
 */
export interface DelegateRow {
  /** Latest relayed activity, oldest → newest. The card tickers the tail. */
  activity: string[]
  durationSeconds?: number
  goal: string
  id: string
  /** The call's delegation id, when its result carried one — the exact join key
   *  against the subagent store, which stamps the same id on every native child. */
  delegationId?: string
  model?: string
  /** The child's own session id, when it reported one — opens its window. */
  sessionId?: string
  status: DelegateRowStatus
}

/**
 * `dispatched` is the state the other two sources can't describe: a background
 * delegation whose children outlived the turn, seen from a transcript that has
 * been reloaded since. It is running, but nothing here is watching it, so it
 * must not spin.
 */
export type DelegateRowStatus = SubagentStatus | 'dispatched'

const field = (record: Record<string, unknown>, key: string): string => firstStringField(record, [key])

/** The goals a `delegate_task` call dispatched, in task order. */
export function delegateGoals(args: unknown): string[] {
  const record = parseMaybeObject(args)
  const tasks = Array.isArray(record.tasks) ? record.tasks : []

  if (tasks.length > 0) {
    return tasks.map((task, index) => field(parseMaybeObject(task), 'goal') || `Task ${index + 1}`)
  }

  const goal = field(record, 'goal')

  return goal ? [goal] : []
}

function resultRows(result: unknown): Record<string, unknown>[] {
  const record = parseMaybeObject(result)
  const results = Array.isArray(record.results) ? record.results : []

  return results.map(parseMaybeObject)
}

// The delegate tool settles result rows with statuses like 'ok', 'error',
// 'timeout', 'failed'/'failure' (tools/delegate_tool.py). Anything that is
// not a success must render as failed — mapping unknown statuses to
// 'completed' hid timed-out children behind a green check (#73728, #85492).
function settledRowStatus(status: string): DelegateRowStatus {
  return status === '' || status === 'ok' || status === 'completed' ? 'completed' : 'failed'
}

function dispatchedGoals(result: unknown): string[] {
  const record = parseMaybeObject(result)

  if (field(record, 'status') !== 'dispatched') {
    return []
  }

  return Array.isArray(record.goals) ? record.goals.filter((goal): goal is string => typeof goal === 'string') : []
}

/**
 * The rows a call describes on its own — before any live subagent state is
 * layered on. This is what a rehydrated transcript has to work with: the goals
 * it dispatched, and whatever the result said about how they went.
 *
 * A call with no result yet is still being placed, so its rows read as
 * running; the moment a background dispatch answers, they drop to parked.
 */
export function delegateRowsFromCall(args: unknown, result: unknown, toolCallId = ''): DelegateRow[] {
  const goals = delegateGoals(args)
  const finished = resultRows(result)
  const dispatched = dispatchedGoals(result)
  const titles = goals.length > 0 ? goals : dispatched.length > 0 ? dispatched : finished.map(() => 'Delegated task')
  const idle: DelegateRowStatus = result === undefined ? 'running' : 'dispatched'
  // Both the settled combined result and the background dispatch handle name
  // the delegation; native children carry the same id, so it is the one join
  // key that survives a repeated goal string (#136532).
  const delegationId = field(parseMaybeObject(result), 'delegation_id') || undefined

  return titles.map((goal, index) => {
    const entry = finished[index]
    const summary = entry ? field(entry, 'summary') : ''

    return {
      activity: summary ? [summary] : [],
      delegationId,
      durationSeconds: entry ? (numberValue(entry.duration_seconds) ?? undefined) : undefined,
      goal,
      id: `${toolCallId}:${index}`,
      model: entry ? field(entry, 'model') || undefined : undefined,
      status: entry ? settledRowStatus(field(entry, 'status')) : idle
    }
  })
}

function fromSubagent(live: SubagentProgress, fallbackId: string, fallbackGoal: string): DelegateRow {
  return {
    activity: live.stream.map(entry => entry.text).filter(Boolean),
    durationSeconds: live.durationSeconds,
    goal: live.goal || fallbackGoal,
    id: live.id || fallbackId,
    model: live.model,
    sessionId: live.sessionId,
    status: live.status
  }
}

/**
 * Layer the session's live subagents over the rows a call describes.
 *
 * Four joins, narrowest first. The delegate fallback (used when the gateway
 * relays no native `subagent.*` events) keys its rows off the tool call id, so
 * those match exactly. Native children stamp the call's delegation id, so a
 * result that carries it joins exactly — even when two delegations share a
 * goal string, which is how an older card used to re-adopt a newer call's
 * workers (#136532). Without it (old transcripts, no live transcripts), the
 * goal string verbatim from the same arguments this call was built from is the
 * next best key. Failing both, task order is how the delegate tool numbers its
 * children — but only trust it when the two sides agree on how many there are,
 * or a second delegation in the same turn will claim the first one's workers.
 * A row that knows its delegation id trusts it exclusively: a live child not
 * carrying it is another call's worker, and a fuzzy join would only misfile it.
 *
 * Live state wins wherever it exists: a settled result tells you a child
 * finished, but only the store knows what it is doing right now.
 */
// A child of this call carries the call's bare delegation id; splitting one
// call into async units keeps that id as the unit ids' prefix (…-1, …-2).
const sameDelegation = (childDelegationId: string | undefined, callDelegationId: string): boolean =>
  childDelegationId === callDelegationId || (childDelegationId?.startsWith(`${callDelegationId}-`) ?? false)

export function mergeDelegateRows(
  rows: readonly DelegateRow[],
  live: readonly SubagentProgress[],
  toolCallId = ''
): DelegateRow[] {
  if (live.length === 0) {
    return [...rows]
  }

  const unclaimed = [...live]

  const claim = (predicate: (candidate: SubagentProgress) => boolean): SubagentProgress | undefined => {
    const index = unclaimed.findIndex(predicate)

    return index >= 0 ? unclaimed.splice(index, 1)[0] : undefined
  }

  const prefix = toolCallId ? `delegate-tool:${toolCallId}:` : ''
  const byId = rows.map((_row, index) => (prefix ? claim(c => c.id === `${prefix}${index}`) : undefined))

  const byDelegation = rows.map((row, index) =>
    row.delegationId && byId[index] === undefined
      ? claim(c => sameDelegation(c.delegationId, row.delegationId!) && c.taskIndex === index)
      : undefined
  )

  const fuzzy = rows.map(row => !row.delegationId)

  const byGoal = rows.map(
    (row, index) =>
      byId[index] ??
      byDelegation[index] ??
      (fuzzy[index] ? claim(c => normalize(c.goal) === normalize(row.goal)) : undefined)
  )

  const sameShape = rows.length === live.length

  return rows.map((row, index) => {
    const matched = byGoal[index] ?? (sameShape && fuzzy[index] ? claim(c => c.taskIndex === index) : undefined)

    return matched ? fromSubagent(matched, row.id, row.goal) : row
  })
}

export const isDelegateRowLive = (status: DelegateRowStatus): boolean => status === 'running' || status === 'queued'
