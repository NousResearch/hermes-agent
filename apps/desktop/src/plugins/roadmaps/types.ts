/**
 * Roadmaps plugin — the slice of the `/api/plugins/roadmaps` REST contract the
 * UI renders. The backend (`plugins/roadmaps/dashboard/plugin_api.py`) wraps
 * `RoadmapsService` (reads) and `RoadmapsWriter` (versioned writes); we type
 * only what the views read so a schema addition never breaks the build.
 *
 * Every read/write is scoped by `profile` + `project_id` (+ `roadmap_id`) — the
 * profile is the active gateway profile (read from `host.state.profile` and
 * passed explicitly as a query param), the project is a free-form identifier.
 */

/** Scope identity echoed back by the backend on every response. */
export interface RoadmapScope {
  profile_id: string
  project_id: string
  roadmap_id?: string
}

/** The plugin's resolved scope (profile + project + roadmap), used by the
 *  views and mutations. `null` while the selection is incomplete. */
export interface Scope {
  profile: string
  projectId: string
  roadmapId: string
}

/** One row of `GET /roadmaps` — a roadmap in the selected project. */
export interface RoadmapListItem {
  profile_id: string
  project_id: string
  roadmap_id: string
  title: string | null
  purpose: string | null
  lifecycle_state: string
  active_version: number | null
  project_name?: string | null
}

export interface RoadmapsResponse {
  roadmaps: RoadmapListItem[]
  scope: RoadmapScope
}

/** One node of a plan version. `state` is the machine lifecycle state. */
export interface RoadmapNode {
  node_id: string
  version: number
  title: string
  kind: string
  state: string
  progress: number | null
  owner_agent: string | null
  parent_node_id: string | null
  description: string | null
  block_reason: string | null
  created_at: number | null
}

/** One canonical relation (`depends_on` / `blocks`) of a plan version. */
export interface RoadmapRelation {
  relation_id: string
  version: number
  from_node_id: string
  to_node_id: string
  kind: string
  state: string
  reason: string | null
}

/** One todo attached to a node. */
export interface RoadmapTodo {
  todo_id: string
  version: number
  node_id: string
  title: string
  state: string
}

/** One plan version: its metadata plus the full node/relation/todo payload. */
export interface RoadmapVersion {
  version: number
  state: string
  source: string | null
  reason: string | null
  created_at: number | null
  nodes: RoadmapNode[]
  relations: RoadmapRelation[]
  todos: RoadmapTodo[]
}

/** A roadmap with all its plan versions (the snapshot payload). */
export interface Roadmap {
  profile_id: string
  project_id: string
  roadmap_id: string
  title: string | null
  purpose: string | null
  lifecycle_state: string
  active_version: number | null
  versions: RoadmapVersion[]
}

/** `GET /roadmaps/{id}/snapshot` — the single source of truth for the views. */
export interface SnapshotResponse {
  found: boolean
  scope: RoadmapScope
  roadmap: Roadmap | null
}

/** One row of `GET /roadmaps/{id}/plans` — plan metadata, newest first. */
export interface PlanMeta {
  version: number
  state: string
  source: string | null
  reason: string | null
  created_at: number | null
}

export interface PlansResponse {
  plans: PlanMeta[]
  scope: RoadmapScope
}

/** `GET /planning-rules` — the versioned Vision planning rules. */
export interface PlanningRulesResponse {
  version: string
  rules: { prompt: string }
}

/** Normalized plan payload built from a parsed Vision draft (before create). */
export interface PlanPayload {
  title?: string
  nodes: Array<{ node_id: string; title: string; kind: string; [key: string]: unknown }>
  relations: Array<{ relation_id: string; from_node_id: string; to_node_id: string; kind: string; [key: string]: unknown }>
  todos: Array<{ todo_id: string; title: string; [key: string]: unknown }>
}
