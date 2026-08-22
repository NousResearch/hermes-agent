/**
 * Roadmaps plugin — data layer (pure, SDK-free).
 *
 * Everything derived from a roadmap snapshot plus stable English error copy.
 * The backend owns the business rules; this module only reads and labels them,
 * so the plugin stays a pure renderer. Pure functions here are unit-tested
 * without any `@hermes/plugin-sdk` import.
 */

import { NODE_ORDER } from './config'
import type {
  PlanPayload,
  RoadmapNode,
  RoadmapRelation,
  RoadmapScope,
  RoadmapVersion,
  SnapshotResponse
} from './types'

/**
 * Stable English copy per structured error code. The backend message is NEVER
 * displayed — only this guidance, keyed by the code. The REST bridge rejects
 * with `Error("409: {"detail":{"code":5067,"message":"…"}}")`; `rpcError`
 * extracts just the code.
 */
export const ERROR_GUIDANCE: Record<number, { title: string; hint: string }> = {
  5061: {
    title: 'Service unavailable',
    hint: 'The roadmaps backend is temporarily unavailable. Try again in a moment.'
  },
  5062: {
    title: 'Not found',
    hint: 'The project no longer exists in this profile. Check the project identifier and try again.'
  },
  5063: {
    title: 'Invalid parameters',
    hint: 'The scope (profile, project, roadmap) or one of the fields is invalid. Check the selection, then try again.'
  },
  5064: {
    title: 'Stale version',
    hint: 'The roadmap changed since this snapshot was loaded. Reload the snapshot, then retry the action.'
  },
  5065: {
    title: 'Not found',
    hint: 'The roadmap, node, or todo no longer exists in this scope. Reload the list.'
  },
  5066: {
    title: 'Invalid transition',
    hint: 'The current state does not allow this action (e.g. completing a blocked node). Fix the state, then try again.'
  },
  5067: {
    title: 'Conflict',
    hint: 'A roadmap or plan version with this identifier already exists. Reload the list and choose a different name.'
  }
}

export const UNKNOWN_ERROR_HINT =
  'Something unexpected went wrong. Retry, and reload the snapshot if the problem persists.'

/**
 * Extract the structured code from a REST/request rejection. Handles both
 * shapes: an error already carrying a `.code` property, and the Electron REST
 * bridge's `Error("409: {"detail":{"code":5067,…}}")`. `code: null` marks a
 * locally-authored validation failure — null/undefined codes are NOT coerced
 * to 0 (that would mislabel local validation as "unknown guidance").
 */
export function rpcError(err: unknown): { code: number | null } {
  if (err && typeof err === 'object' && 'code' in err) {
    const raw = (err as { code?: unknown }).code

    if (typeof raw === 'number' && Number.isFinite(raw)) {return { code: raw }}

    if (typeof raw === 'string' && raw.trim() !== '' && !Number.isNaN(Number(raw))) {
      return { code: Number(raw) }
    }
  }

  const message = err instanceof Error ? err.message : String(err)
  const brace = message.indexOf('{')

  if (brace !== -1) {
    try {
      const parsed = JSON.parse(message.slice(brace)) as { detail?: unknown }
      const detail = parsed?.detail

      if (detail && typeof detail === 'object') {
        const code = (detail as { code?: unknown }).code

        if (typeof code === 'number' && Number.isFinite(code)) {return { code }}

        if (typeof code === 'string' && code.trim() !== '' && !Number.isNaN(Number(code))) {
          return { code: Number(code) }
        }
      }
    } catch {
      // Not JSON — fall through to "no structured code".
    }
  }

  return { code: null }
}

/** Stable copy for query-level failures (list / snapshot): code + hint only. */
export function errorCopy(err: unknown): { code: number | null; hint: string } {
  const code = rpcError(err).code
  const entry = code != null ? ERROR_GUIDANCE[code] : undefined

  return { code, hint: entry?.hint ?? UNKNOWN_ERROR_HINT }
}

/** A mutation error in React state: `{ code, hint }` where `hint` is only ever
 *  a locally-authored validation text (never a backend string). */
export interface MutationError {
  code: number | null
  hint?: string
}

/** Stable copy for mutation failures. `code == null` → local validation. */
export function mutationErrorCopy(error: MutationError | null): { title: string; hint: string; code: number | null } | null {
  if (!error) {return null}

  if (error.code == null) {
    return { title: 'Action failed', hint: error.hint || UNKNOWN_ERROR_HINT, code: null }
  }

  const entry = ERROR_GUIDANCE[error.code]

  return { title: entry?.title ?? 'Action failed', hint: entry?.hint ?? UNKNOWN_ERROR_HINT, code: error.code }
}

/**
 * Local identifier validation, mirroring the backend contract: non-empty
 * string, at most 128 characters, no control characters. A loop instead of a
 * regex so the shared ESLint config (no-control-regex) stays happy.
 */
export function isValidIdentifier(value: unknown): value is string {
  if (typeof value !== 'string' || value.trim() === '' || value.length > 128) {return false}

  for (const ch of value) {
    const code = ch.codePointAt(0)

    if (code === undefined || code < 32 || code === 127) {return false}
  }

  return true
}

/** Local roadmap-title validation, mirroring the backend contract: non-empty
 *  after trimming, at most 200 characters, no control characters. */
export function validateRoadmapTitle(value: unknown): value is string {
  if (typeof value !== 'string') {return false}
  const t = value.trim()

  if (t === '' || t.length > 200) {return false}

  for (const ch of t) {
    const code = ch.codePointAt(0)

    if (code === undefined || code < 32 || code === 127) {return false}
  }

  return true
}

/** Progress must be an integer in [0, 100] — mirrors the backend contract. */
export function validateProgress(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0 && value <= 100
}

/**
 * Defense in depth: a REST response must carry the scope it was asked for
 * before its data is consumed (protects against a mis-routed response).
 */
export function assertResponseScope(
  response: { scope?: RoadmapScope } | null | undefined,
  expected: { profile?: string; projectId?: string; roadmapId?: string }
): boolean {
  const got: Partial<RoadmapScope> = response?.scope ?? {}
  const okProfile = expected.profile == null || got.profile_id === expected.profile
  const okProject = expected.projectId == null || got.project_id === expected.projectId
  const okRoadmap = expected.roadmapId == null || got.roadmap_id === expected.roadmapId

  return okProfile && okProject && okRoadmap
}

/** Display label: prefer the human title, fall back to the id. */
export const nodeLabel = (n: RoadmapNode | null | undefined): string => n?.title || n?.node_id || '?'

/** "1 node" / "3 nodes" — English pluralization for counts. */
export const plural = (n: number, s: string): string => `${n} ${s}${n === 1 ? '' : 's'}`

/** Human-readable date (en-US, matching the native English UI), raw fallback. */
export function formatDate(value: unknown): string {
  if (value === null || value === undefined || value === '') {return ''}
  const d = new Date(value as string | number)

  return Number.isNaN(d.getTime())
    ? String(value)
    : d.toLocaleString('en-US', { dateStyle: 'medium', timeStyle: 'short' })
}

// ── snapshot derivation ──────────────────────────────────────────────────────

/** The active plan version of a snapshot (null when none is active). */
export function activeVersion(snapshot: SnapshotResponse | null | undefined): RoadmapVersion | null {
  const roadmap = snapshot?.roadmap
  const v = roadmap?.active_version

  return roadmap?.versions?.find((x) => x.version === v) ?? null
}

/** Thread: actionable nodes, blocked first, then in_progress, then ready. */
export function threadNodes(version: RoadmapVersion | null | undefined): RoadmapNode[] {
  const nodes = version?.nodes ?? []

  return nodes
    .filter((n) => n.state === 'ready' || n.state === 'in_progress' || n.state === 'blocked')
    .sort(
      (a, b) =>
        (NODE_ORDER[a.state] ?? 9) - (NODE_ORDER[b.state] ?? 9) ||
        String(a.node_id).localeCompare(String(b.node_id))
    )
}

const CANONICAL_RELATIONS = new Set(['depends_on', 'blocks'])

/** Map: canonical relations of the active version (active by default). */
export interface MappedRelation extends RoadmapRelation {
  from: RoadmapNode | null
  to: RoadmapNode | null
}

export function mapRelations(
  version: RoadmapVersion | null | undefined,
  { includeInactive = false }: { includeInactive?: boolean } = {}
): MappedRelation[] {
  const nodes = version?.nodes ?? []
  const byId = new Map(nodes.map((n) => [n.node_id, n]))

  return (version?.relations ?? [])
    .filter((r) => includeInactive || r.state === 'active')
    .filter((r) => CANONICAL_RELATIONS.has(r.kind))
    .map((r) => ({ ...r, from: byId.get(r.from_node_id) ?? null, to: byId.get(r.to_node_id) ?? null }))
    .filter((r) => r.from && r.to)
    .sort((a, b) => String(a.relation_id).localeCompare(String(b.relation_id)))
}

/** Plan: every roadmap version, newest first. */
export function planVersions(snapshot: SnapshotResponse | null | undefined): RoadmapVersion[] {
  const versions = snapshot?.roadmap?.versions ?? []

  return [...versions].sort((a, b) => (Number(b.version) || 0) - (Number(a.version) || 0))
}

/** Milestones: milestone/objective nodes of the active version. */
export function milestoneNodes(version: RoadmapVersion | null | undefined): RoadmapNode[] {
  return (version?.nodes ?? [])
    .filter((n) => n.kind === 'milestone' || n.kind === 'objective')
    .sort((a, b) => String(a.node_id).localeCompare(String(b.node_id)))
}

/** A node's depends_on relations are satisfied when every target is done. */
export function depsSatisfied(node: RoadmapNode, version: RoadmapVersion | null | undefined): boolean {
  const byId = new Map((version?.nodes ?? []).map((n) => [n.node_id, n]))

  const deps = (version?.relations ?? [])
    .filter((r) => r.state === 'active' && r.kind === 'depends_on' && r.from_node_id === node.node_id)
    .map((r) => byId.get(r.to_node_id))

  // A missing target node counts as satisfied (nothing verifiable blocks it).
  return deps.every((d) => !d || d.state === 'completed' || d.state === 'cancelled')
}

/** Direct depends_on relations of a node, each tagged satisfied or not. */
export interface DepInfo {
  target: RoadmapNode | null
  targetId: string
  satisfied: boolean
}

export function nodeDepsInfo(
  node: RoadmapNode,
  version: RoadmapVersion | null | undefined
): { deps: DepInfo[]; total: number; satisfied: number } {
  const byId = new Map((version?.nodes ?? []).map((n) => [n.node_id, n]))

  const deps = (version?.relations ?? [])
    .filter((r) => r.state === 'active' && r.kind === 'depends_on' && r.from_node_id === node.node_id)
    .map((r) => {
      const target = byId.get(r.to_node_id) ?? null

      return {
        target,
        targetId: r.to_node_id,
        satisfied: !target || target.state === 'completed' || target.state === 'cancelled'
      }
    })

  const total = deps.length
  const satisfied = deps.filter((d) => d.satisfied).length

  return { deps, total, satisfied }
}

/** Nodes that wait on this node (incoming depends_on), active version. */
export function nodeDependants(node: RoadmapNode, version: RoadmapVersion | null | undefined): RoadmapNode[] {
  const byId = new Map((version?.nodes ?? []).map((n) => [n.node_id, n]))

  return (version?.relations ?? [])
    .filter((r) => r.state === 'active' && r.kind === 'depends_on' && r.to_node_id === node.node_id)
    .map((r) => byId.get(r.from_node_id))
    .filter((n): n is RoadmapNode => Boolean(n))
}

/** Incoming blocks relations: who is holding this node down. */
export interface Blocker {
  from: RoadmapNode
  reason: string | null
  relationId: string
}

export function nodeBlockers(node: RoadmapNode, version: RoadmapVersion | null | undefined): Blocker[] {
  const byId = new Map((version?.nodes ?? []).map((n) => [n.node_id, n]))

  return (version?.relations ?? [])
    .filter((r) => r.state === 'active' && r.kind === 'blocks' && r.to_node_id === node.node_id)
    .map((r) => ({ from: byId.get(r.from_node_id) ?? null, reason: r.reason, relationId: r.relation_id }))
    .filter((b): b is Blocker => Boolean(b.from))
}

/** Outgoing blocks relations: nodes this node is holding down. */
export function nodeBlocks(node: RoadmapNode, version: RoadmapVersion | null | undefined): RoadmapNode[] {
  const byId = new Map((version?.nodes ?? []).map((n) => [n.node_id, n]))

  return (version?.relations ?? [])
    .filter((r) => r.state === 'active' && r.kind === 'blocks' && r.from_node_id === node.node_id)
    .map((r) => byId.get(r.to_node_id))
    .filter((n): n is RoadmapNode => Boolean(n))
}

/** Orchestration buckets, computed ONLY from the active version's real data. */
export interface CopilotSections {
  now: RoadmapNode[]
  inflight: RoadmapNode[]
  waiting: RoadmapNode[]
  blocked: RoadmapNode[]
}

export function copilotSections(version: RoadmapVersion | null | undefined): CopilotSections | null {
  const nodes = version?.nodes ?? []

  if (nodes.length === 0) {return null}
  const now = nodes.filter((n) => n.state === 'ready' && depsSatisfied(n, version))
  const inflight = nodes.filter((n) => n.state === 'in_progress')
  const waiting = nodes.filter((n) => n.state === 'ready' && !depsSatisfied(n, version))
  const blocked = nodes.filter((n) => n.state === 'blocked' && n.block_reason)

  return { now, inflight, waiting, blocked }
}

/** The single most critical actionable node, by tier (see comments). */
export interface NextAction {
  node: RoadmapNode
  tier: number
  kind: string
  pending: number
  satisfied: number
  total: number
}

export function nextAction(version: RoadmapVersion | null | undefined): NextAction | null {
  const nodes = version?.nodes ?? []

  if (nodes.length === 0) {return null}
  let best: NextAction | null = null

  for (const n of nodes) {
    if (!['ready', 'in_progress', 'blocked'].includes(n.state)) {continue}
    const { total, satisfied } = nodeDepsInfo(n, version)
    const pending = total - satisfied
    let tier: number
    let kind: string

    if (n.state === 'blocked') {
      if (pending === 0) {
        tier = 0
        kind = 'unblock'
      } else {
        tier = 3
        kind = 'wait-deps'
      }
    } else if (n.state === 'ready') {
      if (pending === 0 && !n.owner_agent) {
        tier = 1
        kind = 'claim'
      } else if (pending === 0 && n.owner_agent) {
        tier = 2
        kind = 'advance'
      } else {
        tier = 4
        kind = 'wait-deps'
      }
    } else if (!n.owner_agent) {
      tier = 5
      kind = 'assign'
    } else {
      tier = 6
      kind = 'advance'
    }

    const cand: NextAction = { node: n, tier, kind, pending, satisfied, total }

    if (
      !best ||
      tier < best.tier ||
      (tier === best.tier && pending < best.pending) ||
      (tier === best.tier && pending === best.pending && String(n.node_id) < String(best.node.node_id))
    ) {
      best = cand
    }
  }

  return best
}

/**
 * Implicit critical path: the longest depends_on chain among actionable nodes
 * (ready / in_progress), guarded against cycles. Returns node_ids from the
 * deepest dependant down to the leaf dependency.
 */
export function criticalChain(version: RoadmapVersion | null | undefined): string[] {
  const nodes = version?.nodes ?? []
  const byId = new Map(nodes.map((n) => [n.node_id, n]))
  const depsOf = new Map<string, string[]>()

  for (const r of version?.relations ?? []) {
    if (r.state !== 'active' || r.kind !== 'depends_on') {continue}
    const arr = depsOf.get(r.from_node_id) ?? []
    arr.push(r.to_node_id)
    depsOf.set(r.from_node_id, arr)
  }

  const memo = new Map<string, number>()

  const depth = (id: string, seen: Set<string>): number => {
    if (memo.has(id)) {return memo.get(id)!}

    if (seen.has(id)) {return 0}
    seen.add(id)
    let d = 0

    for (const depId of depsOf.get(id) ?? []) {d = Math.max(d, 1 + depth(depId, seen))}
    seen.delete(id)
    memo.set(id, d)

    return d
  }

  const actionable = nodes.filter((n) => n.state === 'ready' || n.state === 'in_progress')

  if (actionable.length === 0) {return []}
  let best: RoadmapNode | null = null
  let bestDepth = -1

  for (const n of actionable) {
    const d = depth(n.node_id, new Set())

    if (d > bestDepth) {
      bestDepth = d
      best = n
    }
  }

  if (!best || bestDepth <= 0) {return best ? [best.node_id] : []}
  const chain = [best.node_id]
  let cur = best
  const seen = new Set([best.node_id])

  while (chain.length <= nodes.length) {
    const deps = (depsOf.get(cur.node_id) ?? [])
      .filter((d) => !seen.has(d))
      .map((dId) => ({ dId, depth: depth(dId, new Set()) }))
      .sort((a, b) => b.depth - a.depth)

    if (deps.length === 0) {break}
    const next = byId.get(deps[0].dId)

    if (!next) {break}
    seen.add(next.node_id)
    chain.push(next.node_id)
    cur = next
  }

  return chain
}

/** Group milestones by their parent node (when one exists); flat otherwise. */
export interface MilestoneGroup {
  label: string | null
  nodes: RoadmapNode[]
}

export function groupMilestones(version: RoadmapVersion | null | undefined): MilestoneGroup[] {
  const nodes = milestoneNodes(version)
  const byId = new Map((version?.nodes ?? []).map((n) => [n.node_id, n]))
  const groups = new Map<string, RoadmapNode[]>()
  const flat: RoadmapNode[] = []

  for (const n of nodes) {
    const parent = n.parent_node_id ? byId.get(n.parent_node_id) ?? null : null

    if (parent) {
      const arr = groups.get(parent.node_id) ?? []
      arr.push(n)
      groups.set(parent.node_id, arr)
    } else {
      flat.push(n)
    }
  }

  const entries: MilestoneGroup[] = [...groups.entries()].map(([parentId, groupNodes]) => ({
    label: nodeLabel(byId.get(parentId)),
    nodes: groupNodes
  }))

  if (flat.length > 0) {entries.push({ label: null, nodes: flat })}

  return entries
}

// ── Vision draft parsing (pure, over a parsed JSON payload) ──────────────────

/** Extract the LAST ```json … ``` fence from text and parse it (null when no
 *  complete JSON fence is present yet). */
export function extractPlanJsonBlock(text: string): unknown {
  if (typeof text !== 'string' || text.trim() === '') {return null}
  const fences = [...text.matchAll(/```json\s*([\s\S]*?)```/gi)]

  if (fences.length === 0) {return null}
  const last = fences[fences.length - 1]

  if (!last || last[1].trim() === '') {return null}

  try {
    return JSON.parse(last[1].trim())
  } catch {
    return null
  }
}

/** Compact preview of a parsed plan payload: proposed title + counts + kinds. */
export interface PlanPreview {
  title: string
  counts: { nodes: number; relations: number; todos: number }
  kinds: string[]
  nodes: Array<{ node_id: string; title: string; kind: string; [key: string]: unknown }>
  relations: Array<{ relation_id: string; from_node_id: string; to_node_id: string; kind: string; [key: string]: unknown }>
  todos: Array<{ todo_id: string; title: string; [key: string]: unknown }>
}

export function planPreviewFromJson(obj: unknown): PlanPreview | null {
  if (!obj || typeof obj !== 'object' || Array.isArray(obj)) {return null}
  const nodes = Array.isArray((obj as { nodes?: unknown }).nodes) ? ((obj as { nodes: unknown[] }).nodes as PlanPreview['nodes']) : []

  if (nodes.length === 0) {return null}

  const relations = Array.isArray((obj as { relations?: unknown }).relations)
    ? ((obj as { relations: unknown[] }).relations as PlanPreview['relations'])
    : []

  const todos = Array.isArray((obj as { todos?: unknown }).todos)
    ? ((obj as { todos: unknown[] }).todos as PlanPreview['todos'])
    : []

  const kinds: string[] = []
  const seenKinds = new Set<string>()

  for (const n of nodes) {
    const kind = typeof n?.kind === 'string' ? n.kind : ''

    if (kind !== '' && !seenKinds.has(kind)) {
      seenKinds.add(kind)
      kinds.push(kind)
    }
  }

  const title = typeof (obj as { title?: unknown }).title === 'string' ? ((obj as { title: string }).title).trim() : ''

  return {
    title: title !== '' ? title : '',
    counts: { nodes: nodes.length, relations: relations.length, todos: todos.length },
    kinds,
    nodes,
    relations,
    todos
  }
}

/** Local structural validation of a parsed plan payload BEFORE plans.create. */
export function validatePlanPayload(payload: unknown): PlanPayload {
  const nodes = Array.isArray((payload as { nodes?: unknown } | null)?.nodes)
    ? ((payload as { nodes: unknown[] }).nodes as PlanPayload['nodes'])
    : []

  const relations = Array.isArray((payload as { relations?: unknown } | null)?.relations)
    ? ((payload as { relations: unknown[] }).relations as PlanPayload['relations'])
    : []

  const todos = Array.isArray((payload as { todos?: unknown } | null)?.todos)
    ? ((payload as { todos: unknown[] }).todos as PlanPayload['todos'])
    : []

  for (const [i, item] of nodes.entries()) {
    if (!item || typeof item !== 'object') {throw localValidationError(`nodes[${i}] must be an object.`)}

    if (!isValidIdentifier(item.node_id)) {
      throw localValidationError(`nodes[${i}].node_id must be a non-empty identifier of at most 128 characters.`)
    }

    if (!validateRoadmapTitle(item.title)) {
      throw localValidationError(`nodes[${i}].title must be non-empty, at most 200 characters.`)
    }

    if (typeof item.kind !== 'string' || item.kind.trim() === '') {
      throw localValidationError(`nodes[${i}].kind must be a non-empty string.`)
    }
  }

  for (const [i, item] of relations.entries()) {
    if (!item || typeof item !== 'object') {throw localValidationError(`relations[${i}] must be an object.`)}

    for (const key of ['relation_id', 'from_node_id', 'to_node_id']) {
      if (!isValidIdentifier(item[key])) {
        throw localValidationError(`relations[${i}].${key} must be a non-empty identifier of at most 128 characters.`)
      }
    }

    if (typeof item.kind !== 'string' || item.kind.trim() === '') {
      throw localValidationError(`relations[${i}].kind must be a non-empty string.`)
    }
  }

  for (const [i, item] of todos.entries()) {
    if (!item || typeof item !== 'object') {throw localValidationError(`todos[${i}] must be an object.`)}

    if (!isValidIdentifier(item.todo_id)) {
      throw localValidationError(`todos[${i}].todo_id must be a non-empty identifier of at most 128 characters.`)
    }

    if (!validateRoadmapTitle(item.title)) {
      throw localValidationError(`todos[${i}].title must be non-empty, at most 200 characters.`)
    }
  }

  return { nodes, relations, todos }
}

/** Locally-authored validation failure: `code: null` + stable English hint. */
export function localValidationError(hint: string): Error & { code: null; hint: string } {
  return Object.assign(new Error(hint), { code: null as null, hint })
}
