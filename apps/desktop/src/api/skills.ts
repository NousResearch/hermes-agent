import type {
  SkillHubPreview,
  SkillHubScanResult,
  SkillHubSearchResponse,
  SkillHubSourcesResponse,
  SkillInfo,
  StarmapGraph
} from '@/types/hermes'
import type { ActionResponse } from '@/types/hermes'

import { capabilityScoped, hermesApi, type ProfileScope, profileScoped } from './client'

export function getSkills(profile?: ProfileScope): Promise<SkillInfo[]> {
  return window.hermesDesktop.api<SkillInfo[]>({
    ...capabilityScoped(profile),
    path: '/api/skills'
  })
}

/** Raw SKILL.md text (frontmatter included) for ANY skill — bundled, hub, or
 *  learned — backing the Capabilities detail pane's full-skill view. */
export function getSkillContent(
  name: string,
  profile?: ProfileScope
): Promise<{ content: string; name: string; path: string }> {
  return window.hermesDesktop.api<{ content: string; name: string; path: string }>({
    ...capabilityScoped(profile),
    path: `/api/skills/content?name=${encodeURIComponent(name)}`
  })
}

export function setSkillEnabled(
  name: string,
  enabled: boolean,
  profile?: ProfileScope
): Promise<{ ok: boolean; name: string; enabled: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean; name: string; enabled: boolean }>({
    ...capabilityScoped(profile),
    path: '/api/skills/toggle',
    method: 'PUT',
    body: { name, enabled }
  })
}

export function getStarmapGraph(profile?: string): Promise<StarmapGraph> {
  return hermesApi<StarmapGraph>({
    ...profileScoped(profile),
    // Backend REST contract — stays /api/learning even though the UI feature is
    // now "star map". Renaming this would break against an un-upgraded backend.
    path: '/api/learning/graph'
  })
}

export interface LearningNodeDetail {
  content: string
  kind: 'memory' | 'skill'
  label: string
  ok: boolean
}

export function getLearningNode(id: string, profile?: ProfileScope): Promise<LearningNodeDetail> {
  return window.hermesDesktop.api<LearningNodeDetail>({
    ...capabilityScoped(profile),
    path: `/api/learning/node?id=${encodeURIComponent(id)}`
  })
}

export function deleteLearningNode(id: string, profile?: ProfileScope): Promise<{ message: string; ok: boolean }> {
  return window.hermesDesktop.api<{ message: string; ok: boolean }>({
    ...capabilityScoped(profile),
    path: '/api/learning/node',
    method: 'DELETE',
    body: { id }
  })
}

export function editLearningNode(
  id: string,
  content: string,
  profile?: ProfileScope
): Promise<{ message: string; ok: boolean }> {
  return window.hermesDesktop.api<{ message: string; ok: boolean }>({
    ...capabilityScoped(profile),
    path: '/api/learning/node',
    method: 'PUT',
    body: { content, id }
  })
}

/** Fetch a merged learning graph from multiple profiles. Each node/edge/card
 *  is tagged with its source profile, and node ids are prefixed to avoid
 *  collisions. */
export function getStarmapGraphMultiProfile(profiles: string[]): Promise<StarmapGraph> {
  const params = new URLSearchParams({ profiles: profiles.join(',') })

  return hermesApi<StarmapGraph>({
    path: `/api/learning/graph?${params.toString()}`
  })
}

/** Cross-profile memory insertion: copy a node's content from one profile
 *  into another profile's MEMORY.md. */
export function crossInsertLearningNode(
  id: string,
  sourceProfile: string,
  targetProfile: string
): Promise<{ message: string; ok: boolean }> {
  return hermesApi<{ message: string; ok: boolean }>({
    body: { id, source_profile: sourceProfile, target_profile: targetProfile },
    method: 'POST',
    path: '/api/learning/node/cross-insert'
  })
}

// ---------------------------------------------------------------------------
// Provider memory nodes — journey nodes whose facts are DERIVED by an external
// memory provider (e.g. Honcho conclusions) rather than authored in a Hermes
// session. These helpers expose the source corpus behind such a node so a user
// can audit provenance ("where did this fact come from?") and, if useful,
// resurrect the originating conversation as a first-class Hermes session.
// ---------------------------------------------------------------------------

/** One raw message from a provider-side session (journey source corpus). */
export interface ProviderSessionMessage {
  content: string
  peer: string
  /** 'user' | 'assistant' when the provider knows which peer is the human. */
  role?: string
  /** Unix seconds, or null when the provider didn't record a time. */
  timestamp: null | number
}

export interface ProviderSessionResponse {
  messages: ProviderSessionMessage[]
  provider: null | string
  session_id: string
}

/** Source corpus behind a provider-contributed journey node — the raw
 *  provider-side conversation a derived fact (e.g. a Honcho conclusion)
 *  came from. Empty `messages` means unavailable, not an error. */
export function getLearningProviderSession(sessionId: string): Promise<ProviderSessionResponse> {
  return window.hermesDesktop.api<ProviderSessionResponse>({
    ...profileScoped(),
    path: `/api/learning/provider-session?session_id=${encodeURIComponent(sessionId)}`
  })
}

export interface MaterializedProviderSession {
  created: boolean
  message_count: number
  ok: boolean
  provider: null | string
  session_id: string
  title: string
}

/** Recreate a provider-side conversation (journey source corpus) as a real
 *  Hermes session, so it can be read and continued like any other session.
 *  Idempotent: an already-materialized conversation returns `created: false`
 *  with the same session id. */
export function materializeLearningProviderSession(sessionId: string): Promise<MaterializedProviderSession> {
  return window.hermesDesktop.api<MaterializedProviderSession>({
    ...profileScoped(),
    path: '/api/learning/provider-session/materialize',
    method: 'POST',
    body: { session_id: sessionId }
  })
}

/** Safe, provenance-tagged draft text for recalling a journey node into a
 *  session as reference context. The recalled body is scanned, delimiter-
 *  defanged, and wrapped in an untrusted-data block server-side — the caller
 *  stashes `text` as the target session's composer draft (user reviews + sends). */
export interface LearningRecallDraft {
  connected_count: number
  /** Threat-scan pattern ids matched in the recalled body (empty = clean). */
  findings: string[]
  id: string
  kind: 'memory' | 'skill'
  label: string
  ok: boolean
  text: string
  truncated: boolean
}

export function getLearningRecallDraft(id: string, profile?: string): Promise<LearningRecallDraft> {
  return window.hermesDesktop.api<LearningRecallDraft>({
    // Scope to the node's OWN profile when given (cross-profile recall from a
    // merged multi-profile graph); otherwise fall back to the active profile.
    ...profileScoped(profile),
    path: `/api/learning/recall-draft?id=${encodeURIComponent(id)}`
  })
}

// ---------------------------------------------------------------------------
// Skills hub — search / preview / scan / install (parity with `hermes skills`
// and the dashboard's Browse-hub tab). Installs spawn background actions whose
// logs are tailed via getActionStatus().
// ---------------------------------------------------------------------------

const HUB_REQUEST_TIMEOUT_MS = 45_000

export function getSkillHubSources(profile?: null | string): Promise<SkillHubSourcesResponse> {
  return hermesApi<SkillHubSourcesResponse>({
    ...profileScoped(profile),
    path: '/api/skills/hub/sources',
    timeoutMs: HUB_REQUEST_TIMEOUT_MS
  })
}

export function searchSkillsHub(
  query: string,
  source = 'all',
  limit = 20,
  profile?: null | string
): Promise<SkillHubSearchResponse> {
  const params = new URLSearchParams({ q: query, source, limit: String(limit) })

  return hermesApi<SkillHubSearchResponse>({
    ...profileScoped(profile),
    path: `/api/skills/hub/search?${params.toString()}`,
    timeoutMs: HUB_REQUEST_TIMEOUT_MS
  })
}

export function previewSkillHub(identifier: string, profile?: null | string): Promise<SkillHubPreview> {
  return hermesApi<SkillHubPreview>({
    ...profileScoped(profile),
    path: `/api/skills/hub/preview?identifier=${encodeURIComponent(identifier)}`,
    timeoutMs: HUB_REQUEST_TIMEOUT_MS
  })
}

export function scanSkillHub(identifier: string, profile?: null | string): Promise<SkillHubScanResult> {
  return hermesApi<SkillHubScanResult>({
    ...profileScoped(profile),
    path: `/api/skills/hub/scan?identifier=${encodeURIComponent(identifier)}`,
    timeoutMs: HUB_REQUEST_TIMEOUT_MS
  })
}

export function installSkillFromHub(identifier: string, profile?: ProfileScope): Promise<ActionResponse> {
  return window.hermesDesktop.api<ActionResponse>({
    ...capabilityScoped(profile),
    path: '/api/skills/hub/install',
    method: 'POST',
    body: { identifier }
  })
}

export function uninstallSkillFromHub(name: string, profile?: ProfileScope): Promise<ActionResponse> {
  return window.hermesDesktop.api<ActionResponse>({
    ...capabilityScoped(profile),
    path: '/api/skills/hub/uninstall',
    method: 'POST',
    body: { name }
  })
}

export function updateSkillsFromHub(profile?: ProfileScope): Promise<ActionResponse> {
  return window.hermesDesktop.api<ActionResponse>({
    ...capabilityScoped(profile),
    path: '/api/skills/hub/update',
    method: 'POST',
    body: {}
  })
}
