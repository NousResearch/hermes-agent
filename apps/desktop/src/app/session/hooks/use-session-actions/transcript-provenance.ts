import type { ClientSessionState, PersistedDisplayTranscriptProvenance } from '../../../types'

export type TranscriptProvenanceScope =
  string | null | undefined | { connectionId?: string | null; profile?: string | null }

export function createPersistedDisplayTranscriptProvenance({
  lineageRootId,
  scope,
  storedSessionId
}: {
  storedSessionId: string
  lineageRootId: string | null
  scope: TranscriptProvenanceScope
}): PersistedDisplayTranscriptProvenance {
  const connectionId = typeof scope === 'object' && scope ? (scope.connectionId ?? '').trim() : ''
  const rawProfile = typeof scope === 'string' ? scope : scope?.profile

  return {
    connectionId,
    coverage: 'latest-page',
    lineageRootId,
    profile: rawProfile?.trim() || 'default',
    source: 'persisted-display',
    storedSessionId
  }
}

export function hasPersistedDisplayTranscriptProvenance(
  state: Pick<ClientSessionState, 'transcriptProvenance'>,
  expected: PersistedDisplayTranscriptProvenance
): boolean {
  const actual = state.transcriptProvenance

  return Boolean(
    actual &&
    actual.source === expected.source &&
    actual.connectionId === expected.connectionId &&
    actual.profile === expected.profile &&
    actual.storedSessionId === expected.storedSessionId &&
    actual.lineageRootId === expected.lineageRootId &&
    actual.coverage === expected.coverage
  )
}

export function withoutTranscriptProvenance(state: ClientSessionState): ClientSessionState {
  if (!state.transcriptProvenance) {
    return state
  }

  const { transcriptProvenance: _transcriptProvenance, ...withoutProvenance } = state

  return withoutProvenance
}

export function invalidatePersistedDisplayTranscriptAuthority(state: ClientSessionState): ClientSessionState {
  return {
    ...state,
    transcriptAuthorityEpoch: (state.transcriptAuthorityEpoch ?? 0) + 1,
    transcriptProvenance: undefined
  }
}

export function suppressTranscriptForView(state: ClientSessionState, suppress: boolean): ClientSessionState {
  if (!suppress || state.messages.length === 0) {
    return state
  }

  return { ...state, messages: [] }
}
