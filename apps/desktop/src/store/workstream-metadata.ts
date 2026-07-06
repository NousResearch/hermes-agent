import { Codecs, persistentAtom } from '@/lib/persisted'
import type { WorkstreamState } from '@/store/workstream'

const WORKSTREAM_METADATA_KEY = 'hermes.desktop.workstream-metadata.v1'

export type WorkstreamLifecycle = 'active' | 'closed' | 'restart_required' | 'safe_delete'
export type StoredWorkstreamLifecycle = Exclude<WorkstreamLifecycle, 'active'>

export interface WorkstreamMetadataEntry {
  lifecycle: StoredWorkstreamLifecycle
  updatedAt: number
}

export type WorkstreamMetadataBySession = Record<string, WorkstreamMetadataEntry>

const STORED_LIFECYCLES = new Set<StoredWorkstreamLifecycle>(['closed', 'restart_required', 'safe_delete'])

function isStoredWorkstreamLifecycle(value: unknown): value is StoredWorkstreamLifecycle {
  return typeof value === 'string' && STORED_LIFECYCLES.has(value as StoredWorkstreamLifecycle)
}

function sanitizeMetadata(value: unknown): WorkstreamMetadataBySession {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {}
  }

  const entries = Object.entries(value).flatMap(([sessionId, rawEntry]) => {
    if (!sessionId || !rawEntry || typeof rawEntry !== 'object' || Array.isArray(rawEntry)) {
      return []
    }

    const lifecycle = (rawEntry as Partial<WorkstreamMetadataEntry>).lifecycle
    const updatedAt = (rawEntry as Partial<WorkstreamMetadataEntry>).updatedAt

    if (!isStoredWorkstreamLifecycle(lifecycle) || typeof updatedAt !== 'number' || !Number.isFinite(updatedAt)) {
      return []
    }

    return [[sessionId, { lifecycle, updatedAt } satisfies WorkstreamMetadataEntry] as const]
  })

  return Object.fromEntries(entries)
}

export const $workstreamMetadata = persistentAtom<WorkstreamMetadataBySession>(
  WORKSTREAM_METADATA_KEY,
  {},
  Codecs.json(sanitizeMetadata)
)

export function setWorkstreamLifecycle(sessionId: string, lifecycle: WorkstreamLifecycle, updatedAt = Date.now()) {
  if (!sessionId) {
    return
  }

  const current = $workstreamMetadata.get()

  if (lifecycle === 'active') {
    if (!(sessionId in current)) {
      return
    }

    const { [sessionId]: _removed, ...rest } = current
    $workstreamMetadata.set(rest)

    return
  }

  $workstreamMetadata.set({
    ...current,
    [sessionId]: { lifecycle, updatedAt }
  })
}

export function workstreamLifecycle(
  sessionId: string,
  metadataBySession: WorkstreamMetadataBySession = $workstreamMetadata.get()
): WorkstreamLifecycle {
  return metadataBySession[sessionId]?.lifecycle ?? 'active'
}

export function explicitStateForLifecycle(lifecycle: WorkstreamLifecycle): null | WorkstreamState {
  if (lifecycle === 'restart_required') {
    return 'restart'
  }

  if (lifecycle === 'closed' || lifecycle === 'safe_delete') {
    return 'close'
  }

  return null
}
