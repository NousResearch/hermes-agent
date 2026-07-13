import { profileSessionKey } from '@/store/session-identity'

export interface ComposerQueueKeys {
  queueRuntimeSessionKey: string | null
  queueSessionKey: string | null
}

export interface ComposerQueueScope {
  key: string
  provenance: 'runtime' | 'stored'
  runtimeKey: string | null
}

export interface ComposerQueueMigration {
  fromKey: string
  toKey: string
}

export function composerQueueKeys(
  profile: string | null | undefined,
  storedSessionId: string | null | undefined,
  runtimeSessionId: string | null | undefined
): ComposerQueueKeys {
  return {
    queueSessionKey: storedSessionId ? profileSessionKey(profile, storedSessionId) : null,
    queueRuntimeSessionKey: runtimeSessionId ? profileSessionKey(profile, runtimeSessionId) : null
  }
}

export function resolveComposerQueueScope(keys: ComposerQueueKeys): ComposerQueueScope | null {
  if (keys.queueSessionKey) {
    return {
      key: keys.queueSessionKey,
      provenance: 'stored',
      runtimeKey: keys.queueRuntimeSessionKey
    }
  }

  if (keys.queueRuntimeSessionKey) {
    return {
      key: keys.queueRuntimeSessionKey,
      provenance: 'runtime',
      runtimeKey: keys.queueRuntimeSessionKey
    }
  }

  return null
}

/** Re-key only the first persistence of the exact active runtime conversation. */
export function queueScopeMigration(
  previous: ComposerQueueScope | null,
  current: ComposerQueueScope | null
): ComposerQueueMigration | null {
  if (
    previous?.provenance !== 'runtime' ||
    current?.provenance !== 'stored' ||
    !previous.runtimeKey ||
    previous.runtimeKey !== current.runtimeKey ||
    previous.key === current.key
  ) {
    return null
  }

  return { fromKey: previous.key, toKey: current.key }
}
