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
const MAX_ARTIFACT_CONTENT_LENGTH = 4 * 1024 * 1024
const MAX_VERSIONS_PER_ARTIFACT = 20
const MAX_SESSIONS = 40

function pruneRegistry(registry: ArtifactRegistry, newestArtifactId: string): ArtifactRegistry {
  const entries = Object.entries(registry)
    .map(([sessionId, records]) => {
      const trimmed = [...records]
        .sort(
          (a, b) =>
            b.updatedAt - a.updatedAt || Number(b.id === newestArtifactId) - Number(a.id === newestArtifactId)
        )
        .slice(0, MAX_ARTIFACTS_PER_SESSION)
        .sort((a, b) => a.createdAt - b.createdAt)

      return [sessionId, trimmed] as const
    })
    .filter(([, records]) => records.length > 0)
    .sort(([, a], [, b]) => {
      const latest = (records: readonly ArtifactRecord[]) => Math.max(...records.map(record => record.updatedAt))

      return (
        latest(b) - latest(a) ||
        Number(b.some(record => record.id === newestArtifactId)) -
          Number(a.some(record => record.id === newestArtifactId))
      )
    })
    .slice(0, MAX_SESSIONS)

  let retained = Object.fromEntries(entries)
  let retainedLength = Object.values(retained).reduce(
    (total, records) =>
      total +
      records.reduce(
        (recordsTotal, record) =>
          recordsTotal + record.versions.reduce((versionsTotal, version) => versionsTotal + version.content.length, 0),
        0
      ),
    0
  )

  if (retainedLength <= MAX_ARTIFACT_CONTENT_LENGTH) {
    return retained
  }

  const historicalVersions = Object.values(retained)
    .flatMap(records =>
      records.flatMap(record =>
        record.versions.slice(0, -1).map((version, index) => ({ index, record, version }))
      )
    )
    .sort(
      (a, b) =>
        a.version.createdAt - b.version.createdAt ||
        a.record.createdAt - b.record.createdAt ||
        a.record.id.localeCompare(b.record.id) ||
        a.index - b.index
    )
  const evictedVersions = new Set<ArtifactVersion>()

  for (const { version } of historicalVersions) {
    if (retainedLength <= MAX_ARTIFACT_CONTENT_LENGTH) {
      break
    }

    evictedVersions.add(version)
    retainedLength -= version.content.length
  }

  if (evictedVersions.size > 0) {
    retained = Object.fromEntries(
      Object.entries(retained).map(([sessionId, records]) => [
        sessionId,
        records.map(record => ({
          ...record,
          versions: record.versions.filter(version => !evictedVersions.has(version))
        }))
      ])
    )
  }

  if (retainedLength <= MAX_ARTIFACT_CONTENT_LENGTH) {
    return retained
  }

  const oldestRecords = Object.values(retained)
    .flat()
    .sort((a, b) => a.updatedAt - b.updatedAt || a.createdAt - b.createdAt || a.id.localeCompare(b.id))
  const protectedArtifactId = oldestRecords.some(record => record.id === newestArtifactId)
    ? newestArtifactId
    : oldestRecords.at(-1)?.id
  const evictedRecords = new Set<string>()

  for (const record of oldestRecords) {
    if (retainedLength <= MAX_ARTIFACT_CONTENT_LENGTH) {
      break
    }

    if (record.id === protectedArtifactId) {
      continue
    }

    evictedRecords.add(record.id)
    retainedLength -= record.versions.reduce((total, version) => total + version.content.length, 0)
  }

  return Object.fromEntries(
    Object.entries(retained)
      .map(([sessionId, records]) => [
        sessionId,
        records.filter(record => !evictedRecords.has(record.id))
      ] as const)
      .filter(([, records]) => records.length > 0)
  )
}

export const $artifactRegistry = atom<ArtifactRegistry>({})

/** Per-artifact selected version index; absent = newest. */
export const $artifactVersionSelection = atom<Record<string, number>>({})

function setPrunedRegistry(registry: ArtifactRegistry, newestArtifactId: string) {
  const selection = $artifactVersionSelection.get()
  const selectedHashes = Object.fromEntries(
    Object.entries(selection).flatMap(([artifactId, versionIndex]) => {
      const hash = findArtifact($artifactRegistry.get(), artifactId)?.versions[versionIndex]?.hash

      return hash ? [[artifactId, hash]] : []
    })
  )
  const pruned = pruneRegistry(registry, newestArtifactId)
  const reconciledSelection = Object.fromEntries(
    Object.entries(selectedHashes).flatMap(([artifactId, hash]) => {
      const record = findArtifact(pruned, artifactId)
      const versionIndex = record?.versions.findIndex(version => version.hash === hash) ?? -1

      return record && versionIndex >= 0 && versionIndex < record.versions.length - 1
        ? [[artifactId, versionIndex]]
        : []
    })
  )

  $artifactRegistry.set(pruned)

  if (
    Object.keys(selection).length !== Object.keys(reconciledSelection).length ||
    Object.entries(selection).some(([artifactId, versionIndex]) => reconciledSelection[artifactId] !== versionIndex)
  ) {
    $artifactVersionSelection.set(reconciledSelection)
  }
}

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

    setPrunedRegistry(
      {
        ...registry,
        [id]: records.map(record => (record.id === existing.id ? next : record))
      },
      existing.id
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

  setPrunedRegistry({ ...registry, [id]: [...records, record] }, record.id)

  return { artifactId: record.id, record, versionAdded: true }
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
