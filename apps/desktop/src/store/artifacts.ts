import { atom } from 'nanostores'

import { artifactContentHash, type ArtifactDetection, type ArtifactKind, artifactSlug } from '@/lib/artifact-detect'

import { closeArtifactPreviewTabs, openPreview, type PreviewTarget } from './preview'

/**
 * ARTIFACT REGISTRY — substantial generated content (HTML pages, large SVGs,
 * long code) produced in the transcript, promoted out of the message flow into
 * versioned content the right rail can preview. The registry is authoritative
 * for artifact content; a rail tab only ever holds a reference to it, so a new
 * version shows up in an already-open tab.
 *
 * Identity: one artifact = one (session, slug) pair, where the slug derives
 * from kind + language + title. When the model regenerates "the dashboard"
 * three times in a session, that is ONE artifact with three versions, exactly
 * like a document the user keeps refining — not three cards.
 *
 * Memory-only: the transcript is the durable copy. Cards re-register as they
 * render, so a reload rebuilds the registry (and its version history) for free
 * instead of parking megabytes of generated HTML in localStorage.
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

export type ArtifactRegistry = Record<string, ArtifactRecord[]>

const MAX_ARTIFACTS_PER_SESSION = 24
const MAX_VERSIONS_PER_ARTIFACT = 20
const MAX_SESSIONS = 40
// The count caps alone still allow 40 × 24 × 20 = 19,200 full content strings,
// each a second strong copy of transcript text (#108172). This bounds the sum,
// in UTF-16 code units — what a JS string actually holds.
export const MAX_RETAINED_CONTENT_CHARS = 4 * 1024 * 1024

function retainedChars(registry: ArtifactRegistry): number {
  let total = 0

  for (const records of Object.values(registry)) {
    for (const record of records) {
      for (const version of record.versions) {
        total += version.content.length
      }
    }
  }

  return total
}

/**
 * Oldest historical versions go first; every artifact keeps its current
 * version. If current versions alone are over budget, the least recently
 * updated artifacts go whole — never the newest one, and content is never
 * truncated. Anything dropped re-registers when its card renders again.
 */
function enforceContentBudget(registry: ArtifactRegistry): ArtifactRegistry {
  let total = retainedChars(registry)

  if (total <= MAX_RETAINED_CONTENT_CHARS) {
    return registry
  }

  const records = Object.values(registry).flat()
  const dropVersion = new Set<ArtifactVersion>()
  const dropRecord = new Set<ArtifactRecord>()

  const history = records.flatMap(record => record.versions.slice(0, -1)).sort((a, b) => a.createdAt - b.createdAt)

  for (const version of history) {
    if (total <= MAX_RETAINED_CONTENT_CHARS) {
      break
    }

    dropVersion.add(version)
    total -= version.content.length
  }

  const byAge = [...records].sort((a, b) => a.updatedAt - b.updatedAt).slice(0, -1)

  for (const record of byAge) {
    if (total <= MAX_RETAINED_CONTENT_CHARS) {
      break
    }

    dropRecord.add(record)
    total -= record.versions.reduce((sum, v) => sum + (dropVersion.has(v) ? 0 : v.content.length), 0)
  }

  const next: ArtifactRegistry = {}

  for (const [sessionId, sessionRecords] of Object.entries(registry)) {
    const kept = sessionRecords
      .filter(record => !dropRecord.has(record))
      .map(record =>
        record.versions.some(v => dropVersion.has(v))
          ? { ...record, versions: record.versions.filter(v => !dropVersion.has(v)) }
          : record
      )

    if (kept.length > 0) {
      next[sessionId] = kept
    }
  }

  return next
}

/** Selection is an index; pruning shifts indices, so re-find each selected
 *  version by hash and drop selections whose version is gone (→ newest). */
function reconcileVersionSelection(before: ArtifactRegistry, after: ArtifactRegistry) {
  const selection = $artifactVersionSelection.get()
  const next: Record<string, number> = {}
  let changed = false

  for (const [artifactId, index] of Object.entries(selection)) {
    const hash = findArtifact(before, artifactId)?.versions[index]?.hash
    const versions = findArtifact(after, artifactId)?.versions ?? []
    const moved = hash ? versions.findIndex(v => v.hash === hash) : -1

    if (moved >= 0 && moved < versions.length - 1) {
      next[artifactId] = moved
    }

    changed ||= next[artifactId] !== index
  }

  if (changed) {
    $artifactVersionSelection.set(next)
  }
}

function commitRegistry(before: ArtifactRegistry, draft: ArtifactRegistry) {
  const after = enforceContentBudget(pruneRegistry(draft))

  $artifactRegistry.set(after)
  reconcileVersionSelection(before, after)

  return after
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

export const $artifactRegistry = atom<ArtifactRegistry>({})

/** Per-artifact selected version index; absent = newest. */
export const $artifactVersionSelection = atom<Record<string, number>>({})

/** Lookup against a registry value, for components that already subscribe to
 *  the atom and need the record to change identity when it does. */
export function findArtifact(registry: ArtifactRegistry, artifactId: string): ArtifactRecord | null {
  for (const records of Object.values(registry)) {
    const found = records.find(record => record.id === artifactId)

    if (found) {
      return found
    }
  }

  return null
}

export function getArtifact(artifactId: string): ArtifactRecord | null {
  return findArtifact($artifactRegistry.get(), artifactId)
}

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

    const after = commitRegistry(registry, {
      ...registry,
      [id]: records.map(record => (record.id === existing.id ? next : record))
    })

    return { artifactId: existing.id, record: findArtifact(after, existing.id) ?? next, versionAdded: true }
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

  const after = commitRegistry(registry, { ...registry, [id]: [...records, record] })

  return { artifactId: record.id, record: findArtifact(after, record.id) ?? record, versionAdded: true }
}

/** A rail tab for an artifact references the registry by id rather than
 *  carrying content, so an open tab follows the artifact as it gains versions. */
export function artifactPreviewTarget(record: ArtifactRecord): PreviewTarget {
  return { kind: 'artifact', label: record.title, source: record.id, url: record.id }
}

/** Open an artifact in the right rail at `versionIndex` (default: newest).
 *  User-initiated only (card click) — never called from streaming, per the
 *  no-hijack rule. */
export function openArtifact(artifactId: string, versionIndex?: number) {
  const record = getArtifact(artifactId)

  if (!record) {
    return
  }

  selectArtifactVersion(artifactId, versionIndex ?? record.versions.length - 1)
  openPreview(artifactPreviewTarget(record))
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

export function clearArtifactRegistry() {
  $artifactRegistry.set({})
  $artifactVersionSelection.set({})
  closeArtifactPreviewTabs()
}
