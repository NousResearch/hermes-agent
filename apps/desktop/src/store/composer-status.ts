import { atom, computed } from 'nanostores'

import { $gateway } from './gateway'
import { $subagentsBySession, type SubagentProgress } from './subagents'

/**
 * Unified, typed status feed for the composer status stack.
 *
 * Everything the stack shows is one flat `ComposerStatusItem[]` per session,
 * each carrying a `type` so the stack can group it magically. Subagents are
 * mirrored from the overlay's store (still the source of truth for the Agents
 * view); background work is owned here. The merged view is a derived/computed
 * atom, so the stack has a single thing to read and never juggles sources.
 */
export type StatusItemState = 'done' | 'failed' | 'running'
export type StatusItemType = 'background' | 'subagent'

export interface ComposerStatusItem {
  /** background: non-zero exit shown inline when failed. */
  exitCode?: number
  /** subagent: active tool label shown on the right. */
  currentTool?: string
  id: string
  /** background process: captured stdout/stderr tail for the inline viewer. */
  output?: string
  state: StatusItemState
  title: string
  type: StatusItemType
}

// Writable source for background work, synced from the gateway's process
// registry (`terminal(background=true)` spawns) via `process.list`.
export const $backgroundStatusBySession = atom<Record<string, ComposerStatusItem[]>>({})

// Rows the user X-ed away. The registry keeps finished processes around for a
// while, so without this every refresh would resurrect a dismissed row.
const dismissedBySession = new Map<string, Set<string>>()

const subToItem = (s: SubagentProgress): ComposerStatusItem => ({
  currentTool: s.currentTool,
  id: s.id,
  state: 'running',
  title: s.goal,
  type: 'subagent'
})

// The single thing the stack reads: a typed, merged item list per session.
export const $statusItemsBySession = computed([$subagentsBySession, $backgroundStatusBySession], (subs, background) => {
  const out: Record<string, ComposerStatusItem[]> = {}

  const push = (sid: string, items: ComposerStatusItem[]) => {
    if (items.length > 0) {
      out[sid] = out[sid] ? [...out[sid], ...items] : items
    }
  }

  for (const [sid, list] of Object.entries(subs)) {
    push(sid, list.filter(s => s.status === 'running' || s.status === 'queued').map(subToItem))
  }

  for (const [sid, list] of Object.entries(background)) {
    push(sid, list)
  }

  return out
})

// Fixed render order for the groups in the stack (top → bottom, above queue).
const TYPE_ORDER: readonly StatusItemType[] = ['subagent', 'background']

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
