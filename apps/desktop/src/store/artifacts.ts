import { atom, computed } from 'nanostores'

import { artifactContentHash, type ArtifactDetection, type ArtifactKind, artifactSlug } from '@/lib/artifact-detect'
import { persistentAtom } from '@/lib/persisted'

import { $rightRailActiveTabId, PREVIEW_PANE_ID, RIGHT_RAIL_PREVIEW_TAB_ID, selectRightRailTab } from './layout'
import { setPaneOpen } from './panes'
import { $activeSessionId, $selectedStoredSessionId } from './session'

/**
 * ARTIFACT REGISTRY — substantial generated content (HTML pages, large SVGs,
 * long code) produced in the transcript, promoted out of the message flow into
 * versioned, openable artifacts. The renderer owns this state: artifacts are a
 * presentation of message content the backend already persists, so the store
 * is a cache keyed by session with bounded history.
 *
 * Identity: one artifact = one (session, slug) pair, where the slug derives
 * from kind + language + title. When the model regenerates "the dashboard"
 * three times in a session, that is ONE artifact with three versions, exactly
 * like a document the user keeps refining — not three cards.
 */

export interface ArtifactVersion {
  content: string
  createdAt: number
  hash: string
}

export interface ArtifactRecord {
  createdAt: number
  id: string
  kind: ArtifactKind
  language: string
  sessionId: string
  slug: string
  title: string
  updatedAt: number
  /** Oldest → newest. The last entry is the current version. */
  versions: ArtifactVersion[]
}

type ArtifactRegistry = Record<string, ArtifactRecord[]>

const STORAGE_KEY = 'hermes.desktop.artifacts.v1'
const MAX_ARTIFACTS_PER_SESSION = 24
const MAX_VERSIONS_PER_ARTIFACT = 20
const MAX_SESSIONS = 40
// localStorage is ~5MB; artifacts carry full content, so cap the persisted
// bytes per artifact aggressively. Oversized artifacts survive in memory for
// the app's lifetime but persist only their newest version(s) that fit.
const MAX_PERSISTED_CHARS_PER_ARTIFACT = 120_000

export type ArtifactTabId = `artifact:${string}`

export function artifactTabId(artifactId: string): ArtifactTabId {
  return `artifact:${artifactId}`
}

export function artifactIdFromTabId(tabId: string): string | null {
  return tabId.startsWith('artifact:') ? tabId.slice('artifact:'.length) : null
}

function isArtifactVersion(value: unknown): value is ArtifactVersion {
  if (!value || typeof value !== 'object') {
    return false
  }

  const r = value as Record<string, unknown>

  return typeof r.content === 'string' && typeof r.createdAt === 'number' && typeof r.hash === 'string'
}

function isArtifactRecord(value: unknown): value is ArtifactRecord {
  if (!value || typeof value !== 'object') {
    return false
  }

  const r = value as Record<string, unknown>

  return (
    typeof r.createdAt === 'number' &&
    typeof r.id === 'string' &&
    (r.kind === 'code' || r.kind === 'html' || r.kind === 'svg') &&
    typeof r.language === 'string' &&
    typeof r.sessionId === 'string' &&
    typeof r.slug === 'string' &&
    typeof r.title === 'string' &&
    typeof r.updatedAt === 'number' &&
    Array.isArray(r.versions) &&
    r.versions.length > 0 &&
    r.versions.every(isArtifactVersion)
  )
}

function sanitizeRegistry(value: unknown): ArtifactRegistry {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {}
  }

  const out: ArtifactRegistry = {}

  for (const [sessionId, records] of Object.entries(value as Record<string, unknown>)) {
    if (!Array.isArray(records)) {
      continue
    }

    const valid = records.filter(isArtifactRecord)

    if (valid.length > 0) {
      out[sessionId] = valid
    }
  }

  return out
}

function persistedVersions(record: ArtifactRecord): ArtifactVersion[] {
  const kept: ArtifactVersion[] = []
  let budget = MAX_PERSISTED_CHARS_PER_ARTIFACT

  // Newest first; always keep at least the current version even if oversized.
  for (let i = record.versions.length - 1; i >= 0; i -= 1) {
    const version = record.versions[i]!

    if (kept.length > 0 && version.content.length > budget) {
      break
    }

    budget -= version.content.length
    kept.unshift(version)
  }

  return kept
}

function pruneRegistry(registry: ArtifactRegistry): ArtifactRegistry {
  const entries = Object.entries(registry)
    .map(([sessionId, records]) => {
      const trimmed = [...records]
        .sort((a, b) => b.updatedAt - a.updatedAt)
        .slice(0, MAX_ARTIFACTS_PER_SESSION)
        .sort((a, b) => a.createdAt - b.createdAt)

      return [sessionId, trimmed] as const
    })
    .filter(([, records]) => records.length > 0)
    .sort(([, a], [, b]) => {
      const latest = (records: readonly ArtifactRecord[]) => Math.max(...records.map(record => record.updatedAt))

      return latest(b) - latest(a)
    })
    .slice(0, MAX_SESSIONS)

  return Object.fromEntries(entries)
}

export const $artifactRegistry = persistentAtom<ArtifactRegistry>(
  STORAGE_KEY,
  {},
  {
    decode: raw => sanitizeRegistry(JSON.parse(raw) as unknown),
    encode: registry =>
      JSON.stringify(
        Object.fromEntries(
          Object.entries(pruneRegistry(registry)).map(([sessionId, records]) => [
            sessionId,
            records.map(record => ({ ...record, versions: persistedVersions(record) }))
          ])
        )
      )
  }
)

/** Artifact tabs open in the right rail (ids into the registry). */
export const $artifactTabs = atom<ArtifactTabId[]>([])

/** Per-tab selected version index; absent = newest. Ephemeral by design: a
 *  reopened artifact always lands on its current version. */
export const $artifactVersionSelection = atom<Record<string, number>>({})

function currentArtifactSessionId(): string {
  return $selectedStoredSessionId.get() || $activeSessionId.get() || ''
}

export function getArtifact(artifactId: string): ArtifactRecord | null {
  for (const records of Object.values($artifactRegistry.get())) {
    const found = records.find(record => record.id === artifactId)

    if (found) {
      return found
    }
  }

  return null
}

export const $openArtifacts = computed([$artifactRegistry, $artifactTabs], (registry, tabs) => {
  const byId = new Map<string, ArtifactRecord>()

  for (const records of Object.values(registry)) {
    for (const record of records) {
      byId.set(record.id, record)
    }
  }

  return tabs
    .map(tabId => {
      const id = artifactIdFromTabId(tabId)
      const record = id ? byId.get(id) : undefined

      return record ? { record, tabId } : null
    })
    .filter((entry): entry is { record: ArtifactRecord; tabId: ArtifactTabId } => entry !== null)
})

export function artifactsForSession(sessionId: string | null | undefined): ArtifactRecord[] {
  const id = sessionId?.trim()

  if (!id) {
    return []
  }

  return $artifactRegistry.get()[id] ?? []
}

interface UpsertResult {
  artifactId: string
  record: ArtifactRecord
  /** True when this call appended a NEW version (vs. deduped/no-op). */
  versionAdded: boolean
}

/**
 * Register (or version) an artifact for a session. Same slug + same content
 * hash is a no-op (streaming remounts and transcript re-renders call this
 * repeatedly); same slug + new content appends a version.
 */
export function upsertArtifact(
  sessionId: string | null | undefined,
  detection: ArtifactDetection,
  content: string
): UpsertResult | null {
  const id = sessionId?.trim()
  const trimmed = content.trim()

  if (!id || !trimmed) {
    return null
  }

  const slug = artifactSlug(detection)
  const hash = artifactContentHash(trimmed)
  const registry = $artifactRegistry.get()
  const records = registry[id] ?? []
  const existing = records.find(record => record.slug === slug)
  const now = Date.now()

  if (existing) {
    const known = existing.versions.some(version => version.hash === hash)

    if (known) {
      return { artifactId: existing.id, record: existing, versionAdded: false }
    }

    const versions = [...existing.versions, { content: trimmed, createdAt: now, hash }].slice(
      -MAX_VERSIONS_PER_ARTIFACT
    )

    const next: ArtifactRecord = {
      ...existing,
      // A regenerated artifact may carry a sharper title (html <title> arrives
      // late in the stream); prefer the newest non-generic one.
      title: detection.title || existing.title,
      updatedAt: now,
      versions
    }

    $artifactRegistry.set(
      pruneRegistry({
        ...registry,
        [id]: records.map(record => (record.id === existing.id ? next : record))
      })
    )

    return { artifactId: existing.id, record: next, versionAdded: true }
  }

  const record: ArtifactRecord = {
    createdAt: now,
    id: `${id}:${slug}`,
    kind: detection.kind,
    language: detection.language,
    sessionId: id,
    slug,
    title: detection.title,
    updatedAt: now,
    versions: [{ content: trimmed, createdAt: now, hash }]
  }

  $artifactRegistry.set(pruneRegistry({ ...registry, [id]: [...records, record] }))

  return { artifactId: record.id, record, versionAdded: true }
}

export function upsertCurrentSessionArtifact(detection: ArtifactDetection, content: string): UpsertResult | null {
  return upsertArtifact(currentArtifactSessionId(), detection, content)
}

/** Open an artifact tab in the right rail and select it. User-initiated only
 *  (card click) — never called from streaming, per the no-hijack rule. */
export function openArtifactTab(artifactId: string) {
  const tabId = artifactTabId(artifactId)
  const current = $artifactTabs.get()

  if (!current.includes(tabId)) {
    $artifactTabs.set([...current, tabId])
  }

  // Land on the newest version whenever (re)opened.
  const selection = $artifactVersionSelection.get()

  if (artifactId in selection) {
    const { [artifactId]: _dropped, ...rest } = selection
    $artifactVersionSelection.set(rest)
  }

  setPaneOpen(PREVIEW_PANE_ID, true)
  selectRightRailTab(tabId)
}

export function closeArtifactTab(tabId: ArtifactTabId): boolean {
  const current = $artifactTabs.get()
  const index = current.indexOf(tabId)

  if (index === -1) {
    return false
  }

  const next = current.filter(id => id !== tabId)

  $artifactTabs.set(next)

  const artifactId = artifactIdFromTabId(tabId)

  if (artifactId) {
    const { [artifactId]: _dropped, ...rest } = $artifactVersionSelection.get()
    $artifactVersionSelection.set(rest)
  }

  if ($rightRailActiveTabId.get() === tabId) {
    selectRightRailTab(next[Math.min(index, next.length - 1)] ?? RIGHT_RAIL_PREVIEW_TAB_ID)
  }

  return true
}

export function selectArtifactVersion(artifactId: string, versionIndex: number) {
  const record = getArtifact(artifactId)

  if (!record) {
    return
  }

  const clamped = Math.max(0, Math.min(record.versions.length - 1, versionIndex))
  const selection = $artifactVersionSelection.get()

  if (clamped === record.versions.length - 1) {
    if (artifactId in selection) {
      const { [artifactId]: _dropped, ...rest } = selection
      $artifactVersionSelection.set(rest)
    }

    return
  }

  $artifactVersionSelection.set({ ...selection, [artifactId]: clamped })
}

export function closeAllArtifactTabs() {
  $artifactTabs.set([])
  $artifactVersionSelection.set({})
}

export function clearArtifactRegistry() {
  $artifactRegistry.set({})
  closeAllArtifactTabs()
}
