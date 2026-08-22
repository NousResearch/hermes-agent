/**
 * Roadmaps plugin — data layer over the plugin REST door.
 *
 * Everything goes through `ctx.rest` — the plugin's own `/api/plugins/roadmaps`
 * FastAPI router (`plugins/roadmaps/dashboard/plugin_api.py`), reused as-is via
 * the desktop's namespace-scoped REST door. No new backend, no `host.request`.
 *
 * Fetching, caching, polling and invalidation are React Query's job (the app's
 * standard, via the SDK). This module owns the query keys, the REST calls, and
 * the persisted scope atoms. Every scoped call passes `?profile=&project_id=`
 * explicitly (the plugin_api requires both as query params); the roadmap id is
 * a path segment.
 */

import { atom, type PluginRestOptions, type PluginStorage, useQuery } from '@hermes/plugin-sdk'

import { QUERY } from './config'
import { assertResponseScope } from './data'
import type {
  PlanningRulesResponse,
  PlanPayload,
  PlansResponse,
  RoadmapsResponse,
  SnapshotResponse
} from './types'

type Rest = <T>(path: string, opts?: PluginRestOptions) => Promise<T>

let rest: null | Rest = null

/** Selected project id (free-form; the roadmaps API has no projects list). */
export const $projectId = atom<string>('')

/** Selected roadmap id within the project. Persisted. */
export const $roadmapId = atom<string>('')

const PROJECT_KEY = 'projectId'
const ROADMAP_KEY = 'roadmapId'

// A persisted, subscribable atom (the structural slice we need).
interface Persisted<T> {
  get(): T
  set(value: T): void
  listen(cb: (value: T) => void): () => void
}

/**
 * Bind the plugin's REST door at register time and return a disposer the host
 * runs on unload/disable. Hydrates the persisted scope atoms from storage and
 * keeps storage in sync with them.
 */
export function bindApi(r: Rest, storage: PluginStorage): () => void {
  rest = r
  const unsubs: Array<() => void> = []

  const persist = <T>(a: Persisted<T>, key: string, fallback: T) => {
    a.set(storage.get(key, fallback))
    unsubs.push(a.listen(value => storage.set(key, value)))
  }

  persist($projectId, PROJECT_KEY, '')
  persist($roadmapId, ROADMAP_KEY, '')

  return () => {
    unsubs.forEach(unsub => unsub())
    rest = null
  }
}

function call<T>(path: string, opts?: PluginRestOptions): Promise<T> {
  return rest ? rest<T>(path, opts) : Promise.reject(new Error('roadmaps api not ready'))
}

/** The `?profile=&project_id=` suffix every scoped endpoint requires. */
export function scopeQuery(profile: string, projectId: string): string {
  const search = new URLSearchParams({ profile, project_id: projectId })

  return `?${search.toString()}`
}

const encode = (value: string) => encodeURIComponent(value)

// ── query keys (scope-qualified so switching scope is a clean cache miss) ────

export const roadmapsListKey = (profile: string, projectId: string) =>
  ['roadmaps', 'list', profile, projectId] as const

export const roadmapSnapshotKey = (profile: string, projectId: string, roadmapId: string) =>
  ['roadmaps', 'steer', profile, projectId, roadmapId] as const

export const roadmapPlansKey = (profile: string, projectId: string, roadmapId: string) =>
  ['roadmaps', 'plans', profile, projectId, roadmapId] as const

export const planningRulesKey = ['roadmaps', 'planning-rules'] as const

// ── reads ─────────────────────────────────────────────────────────────────────

export const fetchRoadmaps = (profile: string, projectId: string) =>
  call<RoadmapsResponse>(`/roadmaps${scopeQuery(profile, projectId)}`)

export const fetchSnapshot = (profile: string, projectId: string, roadmapId: string) =>
  call<SnapshotResponse>(`/roadmaps/${encode(roadmapId)}/snapshot${scopeQuery(profile, projectId)}`)

export const fetchPlans = (profile: string, projectId: string, roadmapId: string) =>
  call<PlansResponse>(`/roadmaps/${encode(roadmapId)}/plans${scopeQuery(profile, projectId)}`)

export const fetchPlanningRules = () => call<PlanningRulesResponse>('/planning-rules')

// ── writes ────────────────────────────────────────────────────────────────────

export interface RoadmapCreateBody {
  actor: string
  title: string
  roadmap_id?: string
}

export interface RoadmapUpdateBody {
  actor: string
  expected_version: number
  title: string
}

export interface VersionedBody {
  actor: string
  expected_version: number
}

export const createRoadmap = (profile: string, projectId: string, body: RoadmapCreateBody) =>
  call<{ roadmap_id: string }>(`/roadmaps${scopeQuery(profile, projectId)}`, { method: 'POST', body })

export const updateRoadmap = (profile: string, projectId: string, roadmapId: string, body: RoadmapUpdateBody) =>
  call(`/roadmaps/${encode(roadmapId)}${scopeQuery(profile, projectId)}`, { method: 'PATCH', body })

export const archiveRoadmap = (profile: string, projectId: string, roadmapId: string, body: VersionedBody) =>
  call(`/roadmaps/${encode(roadmapId)}/archive${scopeQuery(profile, projectId)}`, { method: 'POST', body })

export const createPlan = (profile: string, projectId: string, roadmapId: string, body: Record<string, unknown>) =>
  call<{ version: number }>(`/roadmaps/${encode(roadmapId)}/plans${scopeQuery(profile, projectId)}`, {
    method: 'POST',
    body
  })

export const validatePlan = (
  profile: string,
  projectId: string,
  roadmapId: string,
  version: number,
  body: VersionedBody
) =>
  call(`/roadmaps/${encode(roadmapId)}/plans/${version}/validate${scopeQuery(profile, projectId)}`, {
    method: 'POST',
    body
  })

export const activatePlan = (
  profile: string,
  projectId: string,
  roadmapId: string,
  version: number,
  body: VersionedBody
) =>
  call(`/roadmaps/${encode(roadmapId)}/plans/${version}/activate${scopeQuery(profile, projectId)}`, {
    method: 'POST',
    body
  })

const nodeMutation = (op: string) => (
  profile: string,
  projectId: string,
  roadmapId: string,
  nodeId: string,
  body: Record<string, unknown>
) =>
  call(`/roadmaps/${encode(roadmapId)}/nodes/${encode(nodeId)}/${op}${scopeQuery(profile, projectId)}`, {
    method: 'POST',
    body
  })

export const claimNode = nodeMutation('claim')
export const advanceNode = nodeMutation('advance')
export const updateProgress = nodeMutation('progress')
export const completeNode = nodeMutation('complete')
export const blockNode = nodeMutation('block')
export const unblockNode = nodeMutation('unblock')

export const updateTodo = (
  profile: string,
  projectId: string,
  roadmapId: string,
  todoId: string,
  body: Record<string, unknown>
) =>
  call(`/roadmaps/${encode(roadmapId)}/todos/${encode(todoId)}${scopeQuery(profile, projectId)}`, {
    method: 'POST',
    body
  })

// ── query hooks ───────────────────────────────────────────────────────────────

/** Roadmaps list — feeds the roadmap selector. One query, invalidated by key. */
export function useRoadmapsList(profile: string, projectId: string, enabled: boolean) {
  return useQuery({
    queryKey: roadmapsListKey(profile, projectId),
    queryFn: async () => {
      const res = await fetchRoadmaps(profile, projectId)

      if (!assertResponseScope(res, { profile, projectId })) {
        throw Object.assign(new Error('Response out of scope'), { code: 5063 })
      }

      return res
    },
    enabled,
    refetchInterval: QUERY.listRefetchMs
  })
}

/** Snapshot — the ONLY source of truth for the views. */
export function useRoadmapSnapshot(profile: string, projectId: string, roadmapId: string, enabled: boolean) {
  return useQuery({
    queryKey: roadmapSnapshotKey(profile, projectId, roadmapId),
    queryFn: async () => {
      const res = await fetchSnapshot(profile, projectId, roadmapId)

      if (!assertResponseScope(res, { profile, projectId, roadmapId })) {
        throw Object.assign(new Error('Response out of scope'), { code: 5063 })
      }

      return res
    },
    enabled,
    refetchInterval: QUERY.snapshotRefetchMs
  })
}

/** Plan version metadata — newest first, feeds the Plan view timeline. */
export function useRoadmapPlans(profile: string, projectId: string, roadmapId: string, enabled: boolean) {
  return useQuery({
    queryKey: roadmapPlansKey(profile, projectId, roadmapId),
    queryFn: async () => {
      const res = await fetchPlans(profile, projectId, roadmapId)

      if (!assertResponseScope(res, { profile, projectId, roadmapId })) {
        throw Object.assign(new Error('Response out of scope'), { code: 5063 })
      }

      return res
    },
    enabled,
    refetchInterval: QUERY.plansRefetchMs
  })
}

/** The versioned Vision planning rules (global — no scope). */
export function usePlanningRules(enabled: boolean) {
  return useQuery({ queryKey: planningRulesKey, queryFn: fetchPlanningRules, enabled })
}

/** Build the `plans.create` body from a validated plan payload + actor. */
export function planCreateBody(payload: PlanPayload, actor: string, reason?: string): Record<string, unknown> {
  return {
    actor,
    nodes: payload.nodes,
    relations: payload.relations,
    todos: payload.todos,
    source: 'desktop',
    ...(reason ? { reason } : {})
  }
}
