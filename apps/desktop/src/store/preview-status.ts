import { atom, computed } from 'nanostores'

import { Codecs, persistentAtom } from '@/lib/persisted'
import { previewName } from '@/lib/preview-targets'
import { normalizeProfileKey } from '@/store/profile'
import { sessionMatchesStoredId, sessionPinId } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

/**
 * Session-scoped feed of previewable artifacts (HTML files, localhost dev URLs)
 * a tool produced. Surfaced as compact links in the composer status stack —
 * NOT auto-opened and NOT a bulky inline card. Click opens the rail preview or
 * the browser; both are manual.
 *
 * Fed once per SessionView by preview-artifact-publisher.tsx using the same
 * target detection as the tool fallback model, so detection parity is exact.
 */
export interface PreviewArtifact {
  /** cwd captured at detection so a relative path still resolves on click. */
  cwd: string
  /** Dedupe key + display id (the raw target). */
  id: string
  label: string
  target: string
}

export interface PreviewArtifactPublication {
  cwd: string
  publicationKey: string
  target: string
}

const MAX_PER_SESSION = 4
const MAX_DISMISSED_PUBLICATIONS = 4_096
const DISMISSED_PUBLICATIONS_STORAGE_KEY = 'hermes.desktop.dismissedPreviewPublications.v1'
const PUBLICATION_KEY_PATTERN = /^p:[a-z0-9]{7}-[a-z0-9]{7}$/

export interface PreviewArtifactPublicationIdentity {
  baseUrl?: string
  mode: 'local' | 'remote'
  ownerId: string
  profile: string
  publicationId: string
}

const stableScopeHash = (value: string): string => {
  let first = 2166136261
  let second = 0x9e3779b9

  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index)
    first = Math.imul(first ^ code, 16777619)
    second = Math.imul(second ^ code, 2246822519)
  }

  return `${(first >>> 0).toString(36).padStart(7, '0')}-${(second >>> 0).toString(36).padStart(7, '0')}`
}

export function previewArtifactOwnerId(
  storedSessionId: string,
  sessions: readonly Pick<SessionInfo, '_lineage_root_id' | 'id' | 'profile'>[],
  profile = 'default'
): string | null {
  const profileKey = normalizeProfileKey(profile)
  let unscopedMatch: (typeof sessions)[number] | null = null

  for (const candidate of sessions) {
    if (!sessionMatchesStoredId(candidate, storedSessionId)) {
      continue
    }

    if (candidate.profile?.trim()) {
      if (normalizeProfileKey(candidate.profile) === profileKey) {
        return sessionPinId(candidate)
      }
    } else {
      unscopedMatch ??= candidate
    }
  }

  // Untagged rows support older current-profile-only runtimes. An unlisted
  // compression tip waits for its lineage row instead of minting a transient owner.
  return unscopedMatch ? sessionPinId(unscopedMatch) : null
}

export const previewArtifactPublicationKey = ({
  baseUrl = '',
  mode,
  ownerId,
  profile,
  publicationId
}: PreviewArtifactPublicationIdentity): string =>
  `p:${stableScopeHash(
    JSON.stringify([
      mode,
      mode === 'remote' ? baseUrl.trim() : '',
      normalizeProfileKey(profile),
      ownerId,
      publicationId
    ])
  )}`

const sanitizeDismissedPublications = (value: unknown): string[] => {
  if (!Array.isArray(value)) {
    return []
  }

  return [
    ...new Set(value.filter((key): key is string => typeof key === 'string' && PUBLICATION_KEY_PATTERN.test(key)))
  ].slice(-MAX_DISMISSED_PUBLICATIONS)
}

export const $previewStatusBySession = atom<Record<string, PreviewArtifact[]>>({})
export const $dismissedPreviewPublications = persistentAtom<string[]>(
  DISMISSED_PUBLICATIONS_STORAGE_KEY,
  [],
  Codecs.json(sanitizeDismissedPublications)
)

const publicationKeysByArtifact = new WeakMap<PreviewArtifact, Set<string>>()
const $dismissedPublicationKeys = computed($dismissedPreviewPublications, keys => new Set(keys))

const writePreviews = (sid: string, items: PreviewArtifact[]) => {
  const current = $previewStatusBySession.get()

  if (items.length === 0) {
    if (!current[sid]) {
      return
    }

    const next = { ...current }
    delete next[sid]
    $previewStatusBySession.set(next)

    return
  }

  $previewStatusBySession.set({ ...current, [sid]: items })
}

const sameArtifacts = (left: readonly PreviewArtifact[], right: readonly PreviewArtifact[]) =>
  left.length === right.length &&
  left.every(
    (artifact, index) =>
      artifact.cwd === right[index]?.cwd &&
      artifact.id === right[index]?.id &&
      artifact.label === right[index]?.label &&
      artifact.target === right[index]?.target
  )

/** Reconcile the capped feed from complete loaded history, oldest first. */
export function syncPreviewArtifacts(sid: string, publications: readonly PreviewArtifactPublication[]) {
  if (!sid) {
    return
  }

  const publicationKeysByTarget = new Map<string, Set<string>>()
  let artifacts: PreviewArtifact[] = []

  for (const publication of publications) {
    const raw = publication.target.trim()

    if (!raw || !PUBLICATION_KEY_PATTERN.test(publication.publicationKey)) {
      continue
    }

    const publicationKeys = publicationKeysByTarget.get(raw) ?? new Set<string>()
    publicationKeys.add(publication.publicationKey)
    publicationKeysByTarget.set(raw, publicationKeys)

    if ($dismissedPublicationKeys.get().has(publication.publicationKey) || artifacts.some(item => item.id === raw)) {
      continue
    }

    const artifact = { cwd: publication.cwd, id: raw, label: previewName(raw), target: raw }
    artifacts = [...artifacts, artifact].slice(-MAX_PER_SESSION)
  }

  const currentArtifacts = $previewStatusBySession.get()[sid]
  const visibleArtifacts = currentArtifacts && sameArtifacts(currentArtifacts, artifacts) ? currentArtifacts : artifacts

  for (const artifact of visibleArtifacts) {
    const publicationKeys = publicationKeysByTarget.get(artifact.id)

    if (publicationKeys) {
      publicationKeysByArtifact.set(artifact, publicationKeys)
    }
  }

  if (visibleArtifacts !== currentArtifacts) {
    writePreviews(sid, visibleArtifacts)
  }
}

export function dismissPreviewArtifact(sid: string, id: string) {
  const raw = id.trim()

  if (!sid || !raw) {
    return
  }

  const list = $previewStatusBySession.get()[sid]
  const artifact = list?.find(item => item.id === raw)

  if (!artifact) {
    return
  }

  const publicationKeys = publicationKeysByArtifact.get(artifact)

  if (publicationKeys?.size) {
    const dismissed = $dismissedPreviewPublications.get()
    const next = [...new Set([...dismissed, ...publicationKeys])].slice(-MAX_DISMISSED_PUBLICATIONS)

    if (next.length !== dismissed.length || next.some((key, index) => key !== dismissed[index])) {
      $dismissedPreviewPublications.set(next)
    }
  }

  writePreviews(
    sid,
    list.filter(item => item !== artifact)
  )
}

export function clearPreviewArtifacts(sid: string) {
  writePreviews(sid, [])
}
