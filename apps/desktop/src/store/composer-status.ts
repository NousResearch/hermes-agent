import { atom, computed } from 'nanostores'

import { translateNow } from '@/i18n'
import type { TodoItem, TodoStatus } from '@/lib/todos'

import { $gateway } from './gateway'
import { dispatchNativeNotification } from './native-notifications'
import { $subagentsBySession, type SubagentProgress } from './subagents'
import { $todosBySession } from './todos'

/** Composer status stack feed — merged goals, todos, subagents, background per session. */
export type StatusItemState = 'done' | 'failed' | 'running'
export type StatusItemType = 'background' | 'goal' | 'subagent' | 'todo'

export type GoalCardStatus = 'active' | 'paused' | 'done'

export interface ComposerStatusItem {
  /** background: non-zero exit shown inline when failed. */
  exitCode?: number
  /** subagent: active tool label shown on the right. */
  currentTool?: string
  /** goal: active | paused | done. */
  goalStatus?: GoalCardStatus
  /** goal: latest judge verdict. */
  goalVerdict?: string
  id: string
  /** background process: captured stdout/stderr tail for the inline viewer. */
  output?: string
  /** goal: judge reason / paused reason summary. */
  reason?: string
  /** subagent: its own stored session id — row click opens that session window
   *  (livestreamed by the gateway's child-session mirror). */
  sessionId?: string
  state: StatusItemState
  title: string
  /** goal: current judge turn count. */
  turnsUsed?: number
  /** goal: max judge turn budget. */
  maxTurns?: number
  /** todo: the full four-state status driving the row's checkmark glyph. */
  todoStatus?: TodoStatus
  type: StatusItemType
}

// Writable source for background work, synced from the gateway's process
// registry (`terminal(background=true)` spawns) via `process.list`.
export const $backgroundStatusBySession = atom<Record<string, ComposerStatusItem[]>>({})
export const $goalStatusBySession = atom<Record<string, ComposerStatusItem>>({})

const GOAL_BUDGET_RE = /\((?<max>\d+)-turn budget\)/
const GOAL_TURNS_RE = /(?<used>\d+)\/(?<max>\d+)(?:\s+turns)?/

const firstLine = (text: string) => text.trim().split('\n')[0]?.trim() ?? ''

export function setGoalStatusFromText(sid: string, text: string) {
  const line = firstLine(text)

  if (!sid || !line) {
    return
  }

  const current = $goalStatusBySession.get()
  const prev = current[sid]
  let next: ComposerStatusItem | null = null

  if (line.startsWith('⊙ Goal set')) {
    const title = line.split(':').slice(1).join(':').trim() || prev?.title || 'Goal'
    const maxTurns = Number(line.match(GOAL_BUDGET_RE)?.groups?.max || '') || prev?.maxTurns
    next = {
      id: 'goal',
      goalStatus: 'active',
      maxTurns,
      sessionId: sid,
      state: 'running',
      title,
      type: 'goal'
    }
  } else if (line.startsWith('⊙ Goal (active')) {
    const turns = line.match(GOAL_TURNS_RE)?.groups
    next = {
      ...(prev ?? { id: 'goal', sessionId: sid, title: 'Goal', type: 'goal' as const }),
      goalStatus: 'active',
      maxTurns: Number(turns?.max || '') || prev?.maxTurns,
      state: 'running',
      title: line.split('):').slice(1).join('):').trim() || prev?.title || 'Goal',
      turnsUsed: Number(turns?.used || '') || prev?.turnsUsed
    }
  } else if (line.startsWith('⏸ Goal (paused')) {
    const turns = line.match(GOAL_TURNS_RE)?.groups
    next = {
      ...(prev ?? { id: 'goal', sessionId: sid, title: 'Goal', type: 'goal' as const }),
      goalStatus: 'paused',
      maxTurns: Number(turns?.max || '') || prev?.maxTurns,
      state: 'failed',
      title: line.split('):').slice(1).join('):').trim() || prev?.title || 'Goal',
      turnsUsed: Number(turns?.used || '') || prev?.turnsUsed
    }
  } else if (line.startsWith('✓ Goal done')) {
    const turns = line.match(GOAL_TURNS_RE)?.groups
    next = {
      ...(prev ?? { id: 'goal', sessionId: sid, title: 'Goal', type: 'goal' as const }),
      goalStatus: 'done',
      maxTurns: Number(turns?.max || '') || prev?.maxTurns,
      state: 'done',
      title: line.split('):').slice(1).join('):').trim() || prev?.title || 'Goal',
      turnsUsed: Number(turns?.used || '') || prev?.turnsUsed
    }
  } else if (line.startsWith('⏸ Goal paused')) {
    const turns = line.match(GOAL_TURNS_RE)?.groups
    next = {
      ...(prev ?? { id: 'goal', sessionId: sid, title: 'Goal', type: 'goal' as const }),
      goalStatus: 'paused',
      maxTurns: Number(turns?.max || '') || prev?.maxTurns,
      reason: line.replace(/^⏸ Goal paused\s*[—-]?\s*/, '').trim(),
      state: 'failed',
      turnsUsed: Number(turns?.used || '') || prev?.turnsUsed
    }
  } else if (line.startsWith('✓ Goal achieved')) {
    next = {
      ...(prev ?? { id: 'goal', sessionId: sid, title: 'Goal', type: 'goal' as const }),
      goalStatus: 'done',
      reason: line.replace(/^✓ Goal achieved:\s*/, '').trim(),
      state: 'done'
    }
  } else if (line.startsWith('↻ Continuing toward goal')) {
    const turns = line.match(GOAL_TURNS_RE)?.groups
    next = {
      ...(prev ?? { id: 'goal', sessionId: sid, title: 'Goal', type: 'goal' as const }),
      goalStatus: 'active',
      goalVerdict: 'continue',
      maxTurns: Number(turns?.max || '') || prev?.maxTurns,
      reason: line.split(':').slice(1).join(':').trim() || prev?.reason,
      state: 'running',
      turnsUsed: Number(turns?.used || '') || prev?.turnsUsed
    }
  } else if (/goal\s+(cleared|stopped|deleted)/i.test(line) || /^No (active )?goal/i.test(line)) {
    const out = { ...current }
    delete out[sid]
    $goalStatusBySession.set(out)

    return
  } else if (/goal\s+resumed/i.test(line)) {
    next = {
      ...(prev ?? { id: 'goal', sessionId: sid, title: 'Goal', type: 'goal' as const }),
      goalStatus: 'active',
      state: 'running'
    }
  } else if (/goal\s+paused/i.test(line)) {
    next = {
      ...(prev ?? { id: 'goal', sessionId: sid, title: 'Goal', type: 'goal' as const }),
      goalStatus: 'paused',
      state: 'failed'
    }
  }

  if (next) {
    $goalStatusBySession.set({ ...current, [sid]: next })
  }
}

// Rows the user X-ed away. The registry keeps finished processes around for a
// while, so without this every refresh would resurrect a dismissed row.
const dismissedBySession = new Map<string, Set<string>>()

const subToItem = (s: SubagentProgress): ComposerStatusItem => ({
  currentTool: s.currentTool,
  id: s.id,
  sessionId: s.sessionId,
  state: 'running',
  title: s.goal,
  type: 'subagent'
})

const todoToItem = (t: TodoItem): ComposerStatusItem => ({
  id: `todo:${t.id}`,
  state: t.status === 'in_progress' ? 'running' : 'done',
  title: t.content,
  todoStatus: t.status,
  type: 'todo'
})

// The single thing the stack reads: a typed, merged item list per session.
export const $statusItemsBySession = computed(
  [$goalStatusBySession, $subagentsBySession, $backgroundStatusBySession, $todosBySession],
  (goals, subs, background, todos) => {
    const out: Record<string, ComposerStatusItem[]> = {}

    const push = (sid: string, items: ComposerStatusItem[]) => {
      if (items.length > 0) {
        out[sid] = out[sid] ? [...out[sid], ...items] : items
      }
    }

    for (const [sid, goal] of Object.entries(goals)) {
      push(sid, [goal])
    }

    for (const [sid, list] of Object.entries(todos)) {
      push(sid, list.map(todoToItem))
    }

    for (const [sid, list] of Object.entries(subs)) {
      push(sid, list.filter(s => s.status === 'running' || s.status === 'queued').map(subToItem))
    }

    for (const [sid, list] of Object.entries(background)) {
      push(sid, list)
    }

    return out
  }
)

// Fixed render order for the groups in the stack (top → bottom, above queue).
const TYPE_ORDER: readonly StatusItemType[] = ['goal', 'todo', 'subagent', 'background']

export interface StatusGroup {
  items: ComposerStatusItem[]
  type: StatusItemType
}

export function groupStatusItems(items: readonly ComposerStatusItem[]): StatusGroup[] {
  const byType = new Map<StatusItemType, ComposerStatusItem[]>()

  for (const item of items) {
    const list = byType.get(item.type)

    if (list) {
      list.push(item)
    } else {
      byType.set(item.type, [item])
    }
  }

  return TYPE_ORDER.filter(type => byType.has(type)).map(type => ({ items: byType.get(type)!, type }))
}

const writeBackground = (sid: string, items: ComposerStatusItem[]) => {
  const current = $backgroundStatusBySession.get()
  const next = { ...current }

  if (items.length > 0) {
    next[sid] = items
  } else {
    delete next[sid]
  }

  $backgroundStatusBySession.set(next)
}

// `tui_gateway` process.list entry (tools/process_registry.list_sessions + output_tail).
interface GatewayProcessEntry {
  command?: string
  exit_code?: number
  output_tail?: string
  session_id?: string
  status?: string
}

const toBackgroundItem = (proc: GatewayProcessEntry): ComposerStatusItem => {
  const exited = proc.status === 'exited'
  const exitCode = typeof proc.exit_code === 'number' ? proc.exit_code : undefined

  return {
    exitCode,
    id: proc.session_id ?? '',
    output: proc.output_tail || undefined,
    state: exited ? (exitCode ? 'failed' : 'done') : 'running',
    title: (proc.command ?? '').split('\n')[0]!.trim() || 'background process',
    type: 'background'
  }
}

const sameItem = (a: ComposerStatusItem, b: ComposerStatusItem) =>
  a.state === b.state && a.title === b.title && a.output === b.output && a.exitCode === b.exitCode

/**
 * Layout-stable sync of the registry snapshot into the store: existing rows
 * keep their position (status flips happen in place, never reorder), new
 * processes append, dismissed ids stay gone, and unchanged rows keep their
 * object identity so memoised rows skip re-rendering.
 */
export function reconcileBackgroundProcesses(sid: string, procs: GatewayProcessEntry[]) {
  const dismissed = dismissedBySession.get(sid)

  const fresh = new Map(
    procs
      .filter(proc => proc.session_id && !dismissed?.has(proc.session_id))
      .map(proc => [proc.session_id!, toBackgroundItem(proc)])
  )

  const prev = $backgroundStatusBySession.get()[sid] ?? []

  // running → exited since the last snapshot = a background process just finished.
  const prevState = new Map(prev.map(item => [item.id, item.state]))

  for (const [id, item] of fresh) {
    if (item.state !== 'running' && prevState.get(id) === 'running') {
      dispatchNativeNotification({
        body: item.title,
        kind: 'backgroundDone',
        sessionId: sid,
        title: translateNow(
          item.state === 'failed'
            ? 'notifications.native.backgroundFailedTitle'
            : 'notifications.native.backgroundDoneTitle'
        )
      })
    }
  }

  const kept = prev.flatMap(old => {
    const next = fresh.get(old.id)
    fresh.delete(old.id)

    return next ? [sameItem(old, next) ? old : next] : []
  })

  const next = [...kept, ...fresh.values()]

  // Dismissals only need remembering while the registry still reports the id.
  if (dismissed) {
    const reported = new Set(procs.map(proc => proc.session_id))

    for (const id of dismissed) {
      if (!reported.has(id)) {
        dismissed.delete(id)
      }
    }
  }

  if (next.length === prev.length && next.every((item, i) => item === prev[i])) {
    return
  }

  writeBackground(sid, next)
}

/** Pull the session's live process snapshot from the gateway. */
export async function refreshBackgroundProcesses(sid: string): Promise<void> {
  const gateway = $gateway.get()

  if (!sid || !gateway) {
    return
  }

  try {
    const result = await gateway.request<{ processes?: GatewayProcessEntry[] }>('process.list', { session_id: sid })

    reconcileBackgroundProcesses(sid, result?.processes ?? [])
  } catch {
    // Transient socket loss — the next trigger (event or poll) retries.
  }
}

/** X on a finished row: drop it now and keep it dropped across refreshes. */
export function dismissBackgroundProcess(sid: string, id: string) {
  const dismissed = dismissedBySession.get(sid) ?? new Set<string>()
  dismissed.add(id)
  dismissedBySession.set(sid, dismissed)

  const list = $backgroundStatusBySession.get()[sid] ?? []

  writeBackground(
    sid,
    list.filter(item => item.id !== id)
  )
}

/** X on a running row: kill the process for real, then drop the row. */
export function stopBackgroundProcess(sid: string, id: string) {
  void $gateway
    .get()
    ?.request('process.kill', { process_id: id, session_id: sid })
    .catch(() => undefined)
  dismissBackgroundProcess(sid, id)
}

/**
 * Rewind cleanup: a restore/edit discards the turns that spawned these
 * processes, so they belong to an abandoned timeline. Kill the live ones and
 * drop every row. Ids are marked dismissed so an in-flight `process.list` poll
 * (kill is async) can't resurrect them; reconcile garbage-collects those once
 * the registry stops reporting them.
 */
export function resetSessionBackground(sid: string) {
  if (!sid) {
    return
  }

  const gateway = $gateway.get()
  const list = $backgroundStatusBySession.get()[sid] ?? []
  const dismissed = dismissedBySession.get(sid) ?? new Set<string>()

  for (const item of list) {
    dismissed.add(item.id)

    if (item.state === 'running') {
      void gateway?.request('process.kill', { process_id: item.id, session_id: sid }).catch(() => undefined)
    }
  }

  dismissedBySession.set(sid, dismissed)
  writeBackground(sid, [])
}
