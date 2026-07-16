import { atom } from 'nanostores'

import { capitalize } from '@/lib/text'

import { sessionLineageRootId } from './session'

export type SubagentStatus = 'completed' | 'failed' | 'interrupted' | 'queued' | 'running'
export type SubagentStreamKind = 'progress' | 'summary' | 'thinking' | 'tool'

export interface SubagentStreamEntry {
  at: number
  isError?: boolean
  kind: SubagentStreamKind
  text: string
}

export interface SubagentProgress {
  id: string
  profile?: string
  parentId: null | string
  /** Durable parent-session id used by history/sidebar surfaces. The store map
   *  itself stays keyed by the live runtime id for stream routing. */
  ownerSessionId?: string
  /** Still running after its parent turn completed; detached work survives a
   *  later foreground interrupt and reports back in a new parent turn. */
  detached?: boolean
  /** Terminal detached result awaiting the parent completion turn's start. */
  handoff?: boolean
  goal: string
  /** The child's own stored session id — lets UIs open its session window. */
  sessionId?: string
  model?: string
  status: SubagentStatus
  taskCount: number
  taskIndex: number
  startedAt: number
  updatedAt: number
  durationSeconds?: number
  costUsd?: number
  inputTokens?: number
  outputTokens?: number
  toolCount?: number
  filesRead: string[]
  filesWritten: string[]
  stream: SubagentStreamEntry[]
  summary?: string
  /** Active tool while running — cleared on terminal status. */
  currentTool?: string
}

export interface SubagentNode extends SubagentProgress {
  children: SubagentNode[]
}

export type SubagentPayload = Record<string, unknown>

export const isValidSubagentSnapshotRow = (row: unknown): row is SubagentPayload => {
  if (!row || typeof row !== 'object') {
    return false
  }

  const payload = row as SubagentPayload

  return (
    typeof payload.subagent_id === 'string' &&
    payload.subagent_id.trim().length > 0 &&
    (payload.owner_session_id === undefined || typeof payload.owner_session_id === 'string')
  )
}

const TERMINAL: ReadonlySet<SubagentStatus> = new Set(['completed', 'failed', 'interrupted'])
const MAX_STREAM = 24
const PREVIEW_MAX = 220
const TOOL_PREVIEW_MAX = 96

export const $subagentsBySession = atom<Record<string, SubagentProgress[]>>({})

const isStr = (v: unknown): v is string => typeof v === 'string'
const str = (v: unknown) => (isStr(v) ? v : '')
const num = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? v : undefined)
const strList = (v: unknown) => (Array.isArray(v) ? v.filter(isStr) : [])
const UNKNOWN_PROFILE_SCOPE = '\u0001unknown'

const profileKey = (profile: null | string | undefined) =>
  profile === undefined ? UNKNOWN_PROFILE_SCOPE : profile?.trim() || 'default'

export const subagentSessionScopeKey = (profile: null | string | undefined, sid: string) =>
  `${profileKey(profile)}\u0000${sid}`

const profileMatches = (item: SubagentProgress, profile?: string) =>
  profile === undefined || profileKey(item.profile) === profileKey(profile)

const asStatus = (v: unknown, eventType = ''): SubagentStatus => {
  if (v === 'completed' || v === 'failed' || v === 'interrupted' || v === 'queued' || v === 'running') {
    return v
  }

  if (v === 'error' || v === 'timeout') {
    return 'failed'
  }

  // Completion is terminal even when a newer backend adds a status string this
  // renderer does not know yet. Never leave a finished reviewer running forever.
  return eventType === 'subagent.complete' ? 'failed' : 'running'
}

const compact = (text: string, max = PREVIEW_MAX) => {
  const line = text.replace(/\s+/g, ' ').trim()

  if (!line) {
    return ''
  }

  return line.length > max ? `${line.slice(0, max - 1)}…` : line
}

const toolLabel = (name: string) => name.split('_').filter(Boolean).map(capitalize).join(' ') || name

const formatTool = (name: string, preview = '') => {
  const snippet = compact(preview, TOOL_PREVIEW_MAX)

  return snippet ? `${toolLabel(name)}("${snippet}")` : toolLabel(name)
}

interface TailEntry {
  isError?: boolean
  preview?: string
  tool?: string
}

const asTail = (v: unknown): TailEntry[] =>
  Array.isArray(v)
    ? v
        .filter((item): item is Record<string, unknown> => !!item && typeof item === 'object')
        .map(item => ({
          isError: item.is_error === true,
          preview: str(item.preview) || undefined,
          tool: str(item.tool) || undefined
        }))
    : []

const idOf = (p: SubagentPayload) =>
  str(p.subagent_id) || `${str(p.parent_id) || 'root'}:${num(p.task_index) ?? 0}:${str(p.goal)}`

const appendStream = (stream: SubagentStreamEntry[], entry: SubagentStreamEntry) => {
  const last = stream.at(-1)

  if (last?.kind === entry.kind && last.text === entry.text && last.isError === entry.isError) {
    return stream
  }

  return [...stream, entry].slice(-MAX_STREAM)
}

function streamFromPayload(
  payload: SubagentPayload,
  status: SubagentStatus,
  eventType: string,
  at: number
): SubagentStreamEntry[] {
  const out: SubagentStreamEntry[] = []
  const tool = str(payload.tool_name)
  const preview = str(payload.tool_preview) || str(payload.text)
  const text = compact(str(payload.text) || preview)

  for (const tail of asTail(payload.output_tail)) {
    const line = tail.tool ? formatTool(tail.tool, tail.preview ?? '') : compact(tail.preview ?? '')

    if (line) {
      out.push({ at, isError: tail.isError, kind: tail.tool ? 'tool' : 'progress', text: line })
    }
  }

  if (tool) {
    out.push({ at, isError: !!payload.error, kind: 'tool', text: formatTool(tool, preview) })
  }

  if (eventType === 'subagent.progress' && text) {
    out.push({ at, isError: !!payload.error, kind: 'progress', text })
  }

  if (eventType === 'subagent.thinking' && text) {
    out.push({ at, kind: 'thinking', text })
  }

  const summary = compact(str(payload.summary) || str(payload.text))

  if (TERMINAL.has(status) && summary) {
    out.push({ at, isError: status === 'failed', kind: 'summary', text: summary })
  }

  return out
}

function toProgress(
  payload: SubagentPayload,
  prev: SubagentProgress | undefined,
  eventType = '',
  ownerSessionId?: string,
  profile?: string
): SubagentProgress {
  const at = Date.now()
  const status = asStatus(payload.status, eventType)
  const tool = str(payload.tool_name)
  const stream = streamFromPayload(payload, status, eventType, at).reduce(appendStream, prev?.stream ?? [])
  const filesRead = strList(payload.files_read)
  const filesWritten = strList(payload.files_written)

  return {
    id: prev?.id ?? idOf(payload),
    profile: prev?.profile ?? profileKey(profile),
    parentId: str(payload.parent_id) || prev?.parentId || null,
    ownerSessionId: prev?.ownerSessionId || ownerSessionId,
    detached: payload.detached === true || prev?.detached,
    handoff: TERMINAL.has(status) && (payload.detached === true || prev?.detached) ? true : prev?.handoff,
    goal: str(payload.goal) || prev?.goal || 'Subagent',
    sessionId: str(payload.child_session_id) || prev?.sessionId,
    model: str(payload.model) || prev?.model,
    status,
    taskCount: num(payload.task_count) ?? prev?.taskCount ?? 1,
    taskIndex: num(payload.task_index) ?? prev?.taskIndex ?? 0,
    startedAt: prev?.startedAt ?? ((num(payload.started_at) ?? 0) * 1_000 || at),
    updatedAt: at,
    durationSeconds: num(payload.duration_seconds) ?? prev?.durationSeconds,
    costUsd: num(payload.cost_usd) ?? prev?.costUsd,
    inputTokens: num(payload.input_tokens) ?? prev?.inputTokens,
    outputTokens: num(payload.output_tokens) ?? prev?.outputTokens,
    toolCount: num(payload.tool_count) ?? prev?.toolCount,
    filesRead: filesRead.length ? filesRead : (prev?.filesRead ?? []),
    filesWritten: filesWritten.length ? filesWritten : (prev?.filesWritten ?? []),
    stream,
    summary: str(payload.summary) || prev?.summary,
    currentTool: TERMINAL.has(status) ? undefined : tool || prev?.currentTool
  }
}

export function clearSessionSubagents(sid: string, profile?: string) {
  const map = $subagentsBySession.get()
  const list = map[sid]

  if (!list) {
    return
  }

  if (profile !== undefined) {
    const kept = list.filter(item => !profileMatches(item, profile))

    if (kept.length === list.length) {
      return
    }

    if (kept.length > 0) {
      $subagentsBySession.set({ ...map, [sid]: kept })

      return
    }
  }

  const { [sid]: _drop, ...rest } = map
  $subagentsBySession.set(rest)
}

export function clearAllSubagents() {
  $subagentsBySession.set({})
}

/** Remove settled children at the start of a new parent turn without dropping
 * independent reviewers that are still running in the background. */
export function pruneSettledSessionSubagents(sid: string, consumeHandoffs = false, profile?: string): boolean {
  const map = $subagentsBySession.get()
  const list = map[sid]

  if (!list?.length) {
    return false
  }

  const kept = list.filter(
    item => !profileMatches(item, profile) || !TERMINAL.has(item.status) || (item.handoff === true && !consumeHandoffs)
  )

  const retainedTarget = kept.filter(item => profileMatches(item, profile))

  if (kept.length === 0) {
    const { [sid]: _drop, ...rest } = map
    $subagentsBySession.set(rest)

    return false
  }

  if (kept.length !== list.length) {
    $subagentsBySession.set({ ...map, [sid]: kept })
  }

  return retainedTarget.some(item => !TERMINAL.has(item.status) || item.handoff === true)
}

/** Consume only the detached terminal rows owned by one admitted async
 * delivery. Other completed batches in the same parent session retain their
 * handoff leases until their own delivery starts. */
export function consumeSessionSubagentHandoffs(sid: string, subagentIds: readonly string[], profile?: string): boolean {
  const ids = new Set(subagentIds.filter(Boolean))

  if (ids.size === 0) {
    return false
  }

  const map = $subagentsBySession.get()
  const list = map[sid]

  if (!list?.length) {
    return false
  }

  const kept = list.filter(
    item => !profileMatches(item, profile) || !ids.has(item.id) || !TERMINAL.has(item.status) || item.handoff !== true
  )

  if (kept.length === list.length) {
    return false
  }

  if (kept.length > 0) {
    $subagentsBySession.set({ ...map, [sid]: kept })
  } else {
    const { [sid]: _drop, ...rest } = map
    $subagentsBySession.set(rest)
  }

  return true
}

/** Parent completion while children are still active proves they were
 * dispatched asynchronously; synchronous delegation cannot finish its parent
 * turn before joining every child. */
export function markSessionSubagentsDetached(sid: string, profile?: string): boolean {
  const map = $subagentsBySession.get()
  const list = map[sid]

  if (!list?.some(item => profileMatches(item, profile) && !TERMINAL.has(item.status) && !item.detached)) {
    return false
  }

  $subagentsBySession.set({
    ...map,
    [sid]: list.map(item =>
      profileMatches(item, profile) && !TERMINAL.has(item.status) ? { ...item, detached: true } : item
    )
  })

  return true
}

/** Stop owns the foreground turn, not detached reviewers. Keep only those
 * children so their late progress/completion remains visible and routable. */
export function preserveDetachedSessionSubagents(sid: string, profile?: string): boolean {
  const map = $subagentsBySession.get()
  const list = map[sid]

  if (!list?.length) {
    return false
  }

  const detached = list.filter(
    item =>
      !profileMatches(item, profile) ||
      (item.detached === true && (!TERMINAL.has(item.status) || item.handoff === true))
  )

  const retainedTarget = detached.filter(item => profileMatches(item, profile))

  if (detached.length === 0) {
    const { [sid]: _drop, ...rest } = map
    $subagentsBySession.set(rest)

    return false
  }

  if (detached.length !== list.length) {
    $subagentsBySession.set({ ...map, [sid]: detached })
  }

  return retainedTarget.length > 0
}

export function hasDetachedSessionSubagents(sid: string, profile?: string): boolean {
  const list = $subagentsBySession.get()[sid]

  return Boolean(
    list?.some(
      item =>
        profileMatches(item, profile) && item.detached === true && (!TERMINAL.has(item.status) || item.handoff === true)
    )
  )
}

export function pruneDelegateFallbackSubagents(sid: string, profile?: string) {
  const map = $subagentsBySession.get()
  const list = map[sid]

  if (!list?.length) {
    return
  }

  const next = list.filter(item => !profileMatches(item, profile) || !item.id.startsWith('delegate-tool:'))

  if (next.length === list.length) {
    return
  }

  $subagentsBySession.set({ ...map, [sid]: next })
}

export function upsertSubagent(
  sid: string,
  payload: SubagentPayload,
  createIfMissing = true,
  eventType?: string,
  ownerSessionId?: string,
  profile?: string
) {
  let map = $subagentsBySession.get()
  let list = map[sid] ?? []
  const id = idOf(payload)
  const normalizedProfile = profileKey(profile)
  let idx = list.findIndex(item => item.id === id && profileKey(item.profile) === normalizedProfile)
  let prev = idx >= 0 ? list[idx] : undefined

  if (!prev) {
    for (const [otherSid, otherList] of Object.entries(map)) {
      if (otherSid === sid) {
        continue
      }

      const otherIndex = otherList.findIndex(item => item.id === id && profileKey(item.profile) === normalizedProfile)

      if (otherIndex >= 0) {
        prev = otherList[otherIndex]
        const remaining = otherList.filter((_, itemIndex) => itemIndex !== otherIndex)
        const { [otherSid]: _drop, ...rest } = map
        map = remaining.length > 0 ? { ...map, [otherSid]: remaining } : rest
        list = map[sid] ?? []
        idx = -1

        break
      }
    }
  }

  if (!prev && !createIfMissing) {
    return undefined
  }

  if (prev && TERMINAL.has(prev.status)) {
    return prev
  }

  const next = toProgress(payload, prev, eventType, ownerSessionId, normalizedProfile)

  const nextList =
    idx >= 0
      ? list.map((item, itemIndex) => (itemIndex === idx ? next : item))
      : [...list.filter(item => item !== prev), next]

  $subagentsBySession.set({ ...map, [sid]: nextList })

  return next
}

/** Reconcile the event-only cache after a socket gap. Active backend rows are
 * authoritative for non-terminal work; terminal handoff leases remain until
 * their explicit async-delivery turn starts. */
export function reconcileProfileSubagents(
  profile: string,
  activeRows: readonly SubagentPayload[],
  requestedAt = Number.POSITIVE_INFINITY
) {
  const normalizedProfile = profileKey(profile)
  const validRows = activeRows.filter(isValidSubagentSnapshotRow)
  const activeIds = new Set(validRows.map(idOf))
  const current = $subagentsBySession.get()
  const next: Record<string, SubagentProgress[]> = {}

  for (const [sid, list] of Object.entries(current)) {
    const kept = list.filter(
      item =>
        profileKey(item.profile) !== normalizedProfile ||
        TERMINAL.has(item.status) ||
        activeIds.has(item.id) ||
        item.updatedAt >= requestedAt
    )

    if (kept.length > 0) {
      next[sid] = kept
    }
  }

  $subagentsBySession.set(next)

  for (const row of validRows) {
    const rawOwnerSessionId = str(row.owner_session_id)
    const ownerSessionId = rawOwnerSessionId ? sessionLineageRootId(rawOwnerSessionId, normalizedProfile) : ''
    const sid = ownerSessionId || `reconciled:${idOf(row)}`
    const status = asStatus(row.status)
    const terminal = TERMINAL.has(status)
    upsertSubagent(
      sid,
      { ...row, status },
      true,
      terminal ? 'subagent.complete' : 'subagent.start',
      ownerSessionId || undefined,
      normalizedProfile
    )
  }
}

export function buildSubagentTree(items: readonly SubagentProgress[]): SubagentNode[] {
  const nodes = new Map<string, SubagentNode>()
  const nodeKey = (profile: string | undefined, id: string) => `${profileKey(profile)}\u0000${id}`

  for (const item of items) {
    nodes.set(nodeKey(item.profile, item.id), { ...item, children: [] })
  }

  const roots: SubagentNode[] = []

  for (const node of nodes.values()) {
    const parent = node.parentId ? nodes.get(nodeKey(node.profile, node.parentId)) : null

    if (parent) {
      parent.children.push(node)
    } else {
      roots.push(node)
    }
  }

  const sort = (a: SubagentNode, b: SubagentNode) =>
    a.startedAt - b.startedAt || a.taskIndex - b.taskIndex || a.goal.localeCompare(b.goal)

  const walk = (node: SubagentNode) => node.children.sort(sort).forEach(walk)
  roots.sort(sort).forEach(walk)

  return roots
}

export const activeSubagentCount = (items: readonly SubagentProgress[]) =>
  items.filter(item => item.status === 'queued' || item.status === 'running').length

export const failedSubagentCount = (items: readonly SubagentProgress[]) =>
  items.filter(item => item.status === 'failed' || item.status === 'interrupted').length

/** Flatten every session's subagents — the scope the Spawn-tree panel and the
 *  status-bar indicator must agree on. */
export const allSubagents = (bySession: Record<string, SubagentProgress[]>) => Object.values(bySession).flat()
