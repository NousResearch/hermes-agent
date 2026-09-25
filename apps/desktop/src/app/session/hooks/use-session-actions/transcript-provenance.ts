import type { ChatMessage } from '@/lib/chat-messages'

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

export interface TranscriptViewCutoff {
  cutoffIds: ReadonlySet<string>
  // Content fingerprints of the arm-time rows (optional). Compaction
  // re-sequences the cached tail with FRESH row ids mid-hold
  // (archive_and_compact: "consumers that reference durable row ids
  // re-resolve by content"), so an id-only cutoff would pass the whole
  // re-sequenced cached prefix as if it were live and paint the exact
  // compressed tail the hold exists to hide (#73646 via #117867).
  cutoffKeys?: ReadonlySet<string>
}

// Volatile-free content fingerprint: role + text of text parts, JSON of the
// rest. Deliberately ignores row ids and per-row timestamps so a re-sequenced
// copy of the same content fingerprints identically. Ceiling: a genuinely new
// row whose content is byte-identical to an arm-time row stays hidden until
// the hold releases (bounded by the REST window).
export function transcriptRowContentKey(message: ChatMessage): string {
  return `${message.role}:${(message.parts ?? [])
    .map(part => (part.type === 'text' ? `${part.type}:${part.text}` : JSON.stringify(part)))
    .join('|')}`
}

export function suppressTranscriptForView(
  state: ClientSessionState,
  cutoff: TranscriptViewCutoff | null
): ClientSessionState {
  if (cutoff === null) {
    return state
  }

  if (cutoff.cutoffIds.size === 0) {
    // Fail-closed: the gate was armed before any cached row existed, so there
    // is no unproven prefix to hide selectively — everything stays off the
    // view until REST authority lands (#73646).
    return { ...state, messages: [] }
  }

  const messages = state.messages.filter(
    message => !cutoff.cutoffIds.has(message.id) && !(cutoff.cutoffKeys?.has(transcriptRowContentKey(message)) ?? false)
  )

  if (messages.length === state.messages.length) {
    return state
  }

  return { ...state, messages }
}
