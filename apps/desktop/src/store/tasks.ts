import { atom } from 'nanostores'

// Mirrors action_runtime.task_registry.TaskStatus: everything except
// 'running' is terminal (TERMINAL_STATUSES on the Python side).
export type AgentTaskStatus = 'blocked' | 'failed' | 'needs_input' | 'partial' | 'running' | 'succeeded'

export interface AgentTask {
  id: string
  sessionId: string
  intent: string
  goal: string
  label?: string
  model?: string
  status: AgentTaskStatus
  /** ms epoch (converted from the registry's epoch-seconds floats). */
  startedAt: number
  finishedAt?: number
  toolCount: number
  lastTool?: string
  error?: string
}

export type AgentTaskPayload = Record<string, unknown>

// How long a terminal row stays visible before the board drops it.
export const TASK_LINGER_MS = 60_000

const STATUSES: ReadonlySet<string> = new Set(['blocked', 'failed', 'needs_input', 'partial', 'running', 'succeeded'])

export const $tasksBySession = atom<Record<string, AgentTask[]>>({})

export const isTerminalTaskStatus = (status: AgentTaskStatus) => status !== 'running'

const isStr = (v: unknown): v is string => typeof v === 'string'
const str = (v: unknown) => (isStr(v) ? v : '')
const num = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? v : undefined)

const asStatus = (v: unknown): AgentTaskStatus => (isStr(v) && STATUSES.has(v) ? (v as AgentTaskStatus) : 'running')

// Registry timestamps are epoch SECONDS (Python time.time()); the renderer
// works in ms everywhere else.
const toMs = (v: unknown): number | undefined => {
  const seconds = num(v)

  return seconds === undefined ? undefined : Math.round(seconds * 1000)
}

function asRecord(value: unknown): AgentTaskPayload {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as AgentTaskPayload) : {}
}

function toTask(sid: string, payload: AgentTaskPayload, prev: AgentTask | undefined): AgentTask {
  const status = asStatus(payload.status)

  return {
    id: prev?.id ?? str(payload.task_id),
    sessionId: sid,
    intent: str(payload.intent) || prev?.intent || '',
    goal: str(payload.goal) || prev?.goal || '',
    label: str(payload.label) || prev?.label,
    model: str(payload.model) || prev?.model,
    status,
    startedAt: toMs(payload.started_at) ?? prev?.startedAt ?? Date.now(),
    finishedAt: toMs(payload.finished_at) ?? (isTerminalTaskStatus(status) ? Date.now() : prev?.finishedAt),
    toolCount: num(payload.tool_count) ?? prev?.toolCount ?? 0,
    lastTool: str(payload.last_tool) || prev?.lastTool,
    error: str(payload.error) || prev?.error
  }
}

const expired = (task: AgentTask, nowMs: number) =>
  isTerminalTaskStatus(task.status) && nowMs - (task.finishedAt ?? task.startedAt) > TASK_LINGER_MS

export function pruneExpiredAgentTasks(nowMs: number = Date.now()): boolean {
  const map = $tasksBySession.get()
  const nextMap: Record<string, AgentTask[]> = {}
  let changed = false

  for (const [sid, list] of Object.entries(map)) {
    const kept = list.filter(task => !expired(task, nowMs))

    if (kept.length !== list.length) {
      changed = true
    }

    if (kept.length > 0) {
      nextMap[sid] = kept
    } else if (list.length > 0) {
      changed = true
    }
  }

  if (changed) {
    $tasksBySession.set(nextMap)
  }

  return changed
}

function setSessionList(map: Record<string, AgentTask[]>, sid: string, list: AgentTask[]) {
  if (list.length > 0) {
    return { ...map, [sid]: list }
  }

  const { [sid]: _drop, ...rest } = map

  return rest
}

export function upsertAgentTask(sid: string, payload: AgentTaskPayload) {
  const id = str(payload.task_id)

  if (!sid || !id) {
    return
  }

  const map = $tasksBySession.get()
  const now = Date.now()
  const list = (map[sid] ?? []).filter(task => !expired(task, now))
  const idx = list.findIndex(task => task.id === id)
  const prev = idx >= 0 ? list[idx] : undefined

  // The completion snapshot is the terminal answer — a late/duplicate update
  // for an already-terminal record must not resurrect it (honest-status).
  if (prev && isTerminalTaskStatus(prev.status)) {
    $tasksBySession.set(setSessionList(map, sid, list))

    return
  }

  const next = toTask(sid, payload, prev)
  const nextList = idx >= 0 ? list.map(task => (task.id === id ? next : task)) : [...list, next]
  $tasksBySession.set(setSessionList(map, sid, nextList))
}

/**
 * Route a `task.started` / `task.completed` push event into the store.
 *
 * The payload IS the registry snapshot, so its `session_id` is authoritative;
 * `event.session_id` (stamped by the gateway from that same snapshot) is the
 * fallback. An event carrying neither is unattributable background work and
 * is dropped — never attached to whichever chat happens to be focused (same
 * rule as `subagent.*` in gatewayEventRequiresSessionId).
 *
 * Returns true when the event was a task event (consumed or dropped).
 */
export function handleAgentTaskEvent(event: { payload?: unknown; session_id?: string; type?: string }): boolean {
  if (event.type !== 'task.started' && event.type !== 'task.completed') {
    return false
  }

  const payload = asRecord(event.payload)
  const sid = str(payload.session_id) || event.session_id || ''

  if (sid) {
    upsertAgentTask(sid, payload)
  }

  return true
}

/**
 * Reconcile the store with a `task.list` fetch ({tasks: [snapshots]}).
 *
 * The endpoint returns ACTIVE snapshots only, so a stored "running" row the
 * registry no longer lists is stale (it finished while we were disconnected
 * and we missed the completion event) — drop it rather than invent a terminal
 * status for it. Recent terminal rows are kept so a seed right after a
 * completion event doesn't blink the row away early.
 */
export function seedAgentTasks(tasks: unknown) {
  const now = Date.now()
  const fetched = new Map<string, AgentTaskPayload[]>()

  for (const item of Array.isArray(tasks) ? tasks : []) {
    const payload = asRecord(item)
    const sid = str(payload.session_id)

    if (sid && str(payload.task_id)) {
      fetched.set(sid, [...(fetched.get(sid) ?? []), payload])
    }
  }

  const prevMap = $tasksBySession.get()
  const nextMap: Record<string, AgentTask[]> = {}

  for (const sid of new Set([...Object.keys(prevMap), ...fetched.keys()])) {
    const payloads = fetched.get(sid) ?? []
    const listedIds = new Set(payloads.map(payload => str(payload.task_id)))

    const kept = (prevMap[sid] ?? []).filter(task =>
      isTerminalTaskStatus(task.status) ? !expired(task, now) : listedIds.has(task.id)
    )

    let list = kept

    for (const payload of payloads) {
      const id = str(payload.task_id)
      const prev = list.find(task => task.id === id)

      if (prev && isTerminalTaskStatus(prev.status)) {
        continue
      }

      const next = toTask(sid, payload, prev)
      list = prev ? list.map(task => (task.id === id ? next : task)) : [...list, next]
    }

    if (list.length > 0) {
      nextMap[sid] = list
    }
  }

  $tasksBySession.set(nextMap)
}

/** Board rows: every running task plus terminal ones still inside the linger window. */
export function visibleAgentTasks(
  map: Record<string, AgentTask[]>,
  nowMs: number,
  sessionId?: null | string
): AgentTask[] {
  const lists = sessionId ? [map[sessionId] ?? []] : Object.values(map)

  return lists
    .flat()
    .filter(task => !expired(task, nowMs))
    .sort((a, b) => a.startedAt - b.startedAt || a.id.localeCompare(b.id))
}

export const activeAgentTaskCount = (items: readonly AgentTask[]) =>
  items.filter(task => task.status === 'running').length
