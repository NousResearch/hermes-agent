import { normalize } from '@/lib/text'
import { isFailedSubagent, type SubagentOutcome, type SubagentProgress, type SubagentStatus } from '@/store/subagents'

import { firstStringField, numberValue, parseMaybeObject } from './fallback-model'

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
  model?: string
  /**
   * The child's *logical* result, which is a different claim from `status`.
   * Absent on envelopes that predate the field — and absence is not success.
   */
  outcome?: SubagentOutcome
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

/**
 * How a row is allowed to read, once lifecycle and logical result are kept
 * apart.
 *
 * `status: 'completed'` says the child's loop ended and nothing more; the
 * delegate envelope carries no `success` outcome by design, because the
 * strongest thing the backend can assert about returned work is `unverified` —
 * a request for the parent to check the evidence, not a verdict on it. So a
 * settled row is failed, or it is outstanding. It is never done.
 */
export type DelegateRowTone = 'failed' | 'live' | 'parked' | 'partial' | 'unverified'

export const isDelegateRowLive = (status: DelegateRowStatus): boolean => status === 'running' || status === 'queued'

/**
 * The one place that decides whether a delegated child reads as a success.
 *
 * Both the card and the Spawn-tree panel answer this question, and they must
 * not drift, so the failure half defers to the store's `isFailedSubagent`.
 */
export function delegateRowTone(row: Pick<DelegateRow, 'outcome' | 'status'>): DelegateRowTone {
  if (isDelegateRowLive(row.status)) {
    return 'live'
  }

  if (row.status === 'dispatched') {
    return 'parked'
  }

  if (isFailedSubagent({ outcome: row.outcome, status: row.status })) {
    return 'failed'
  }

  // `partial` proves the loop ran out of budget or was cut short; every other
  // settled shape — `unverified`, `unknown`, or an envelope too old to say —
  // is output nobody has checked. Neither one earns a green check.
  return row.outcome === 'partial' ? 'partial' : 'unverified'
}

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

/** Lifecycle words a settled result may report, per the delegate envelope. */
const SETTLED_STATUSES = new Set<string>(['completed', 'error', 'failed', 'interrupted', 'timeout'])
const OUTCOMES = new Set<string>(['failed', 'partial', 'unknown', 'unverified'])

/**
 * A settled entry's own lifecycle word. Envelopes written before the vocabulary
 * widened only ever said `completed` or `failed`, so an unrecognized value
 * falls back to `completed` — which, on its own, promises nothing.
 */
const settledStatus = (entry: Record<string, unknown>): DelegateRowStatus => {
  const status = field(entry, 'status')

  return SETTLED_STATUSES.has(status) ? (status as SubagentStatus) : 'completed'
}

const settledOutcome = (entry: Record<string, unknown>): SubagentOutcome | undefined => {
  const outcome = field(entry, 'outcome')

  return OUTCOMES.has(outcome) ? (outcome as SubagentOutcome) : undefined
}

function resultRows(result: unknown): Record<string, unknown>[] {
  const record = parseMaybeObject(result)
  const results = Array.isArray(record.results) ? record.results : []

  return results.map(parseMaybeObject)
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

  return titles.map((goal, index) => {
    const entry = finished[index]
    const summary = entry ? field(entry, 'summary') : ''

    return {
      activity: summary ? [summary] : [],
      durationSeconds: entry ? (numberValue(entry.duration_seconds) ?? undefined) : undefined,
      goal,
      id: `${toolCallId}:${index}`,
      model: entry ? field(entry, 'model') || undefined : undefined,
      outcome: entry ? settledOutcome(entry) : undefined,
      status: entry ? settledStatus(entry) : idle
    }
  })
}

function fromSubagent(live: SubagentProgress, row: DelegateRow): DelegateRow {
  return {
    activity: live.stream.map(entry => entry.text).filter(Boolean),
    durationSeconds: live.durationSeconds,
    goal: live.goal || row.goal,
    id: live.id || row.id,
    model: live.model,
    // Merge, don't clobber. The store's lifecycle events carry no logical
    // outcome of their own, and letting one of those blank out a `partial` or
    // `failed` the result envelope already proved would repaint a known
    // non-success row as merely unverified — the false green, one layer up.
    outcome: live.outcome ?? row.outcome,
    sessionId: live.sessionId,
    status: live.status
  }
}

/**
 * Layer the session's live subagents over the rows a call describes.
 *
 * Three joins, narrowest first. The delegate fallback (used when the gateway
 * relays no native `subagent.*` events) keys its rows off the tool call id, so
 * those match exactly. Native events carry no tool linkage, but they do carry
 * the goal string verbatim from the same arguments this call was built from.
 * Failing both, task order is how the delegate tool numbers its children — but
 * only trust it when the two sides agree on how many there are, or a second
 * delegation in the same turn will claim the first one's workers.
 *
 * Live state wins wherever it exists: a settled result tells you a child
 * finished, but only the store knows what it is doing right now.
 */
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
  const byGoal = rows.map((row, index) => byId[index] ?? claim(c => normalize(c.goal) === normalize(row.goal)))
  const sameShape = rows.length === live.length

  return rows.map((row, index) => {
    const matched = byGoal[index] ?? (sameShape ? claim(c => c.taskIndex === index) : undefined)

    return matched ? fromSubagent(matched, row) : row
  })
}
