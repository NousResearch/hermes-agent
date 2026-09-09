import { type ChatMessage, toChatMessages } from '@/lib/chat-messages'
import { recordTranscriptTail } from '@/store/transcript-tail'
import type { TranscriptTailPagination } from '@/store/transcript-tail-cache'
import type { SessionMessagesResponse } from '@/types/hermes'

import type { PersistedDisplayTranscriptProvenance } from '../../../types'

import { markSessionOpen } from './session-open-marks'

export interface PersistedDisplayHydrationTarget {
  connectionId: string
  displayRevision: number | null
  /** The listed session says durable history exists, so legacy empty pages are inconclusive. */
  expectsPersistedHistory?: boolean
  lineageRootId: string | null
  profile: string
  resolvedTipId: string | null
  storedSessionId: string
}

export interface PersistedDisplayHydrationPaint {
  displayRevision: number | null
  hit: boolean
}

export interface PublishedPersistedDisplayHydration {
  kind: 'published'
  messages: ChatMessage[]
  provenance: PersistedDisplayTranscriptProvenance | null
  /** Exact authoritative REST page, before local/backfill/live reconciliation. */
  rawPageMessages: ChatMessage[]
  response: SessionMessagesResponse
}

export type PersistedDisplayHydrationResult =
  | { kind: 'inconclusive' }
  | { kind: 'unchanged' }
  | PublishedPersistedDisplayHydration

interface VerifiedTranscriptTail {
  messages: ChatMessage[]
  pagination?: TranscriptTailPagination
  provenance: PersistedDisplayTranscriptProvenance
}

export interface PersistedDisplayHydrationDependencies {
  commit: (messages: ChatMessage[], provenance: PersistedDisplayTranscriptProvenance | null) => void
  fetchLatest: (target: PersistedDisplayHydrationTarget, knownDisplayRevision?: number) => Promise<SessionMessagesResponse>
  isCurrent: () => boolean
  loadVerifiedTranscriptTail: (target: PersistedDisplayHydrationTarget) => VerifiedTranscriptTail | null
  nextFrame: () => Promise<void>
  readCurrentMessages: () => ChatMessage[]
  reconcile: (response: SessionMessagesResponse, current: ChatMessage[]) => ChatMessage[]
}

function exactCacheProof(
  target: PersistedDisplayHydrationTarget,
  provenance: PersistedDisplayTranscriptProvenance
): boolean {
  return Boolean(
    target.displayRevision !== null &&
      target.lineageRootId &&
      target.resolvedTipId &&
      provenance.connectionId === target.connectionId &&
      provenance.profile === target.profile &&
      provenance.storedSessionId === target.storedSessionId &&
      provenance.lineageRootId === target.lineageRootId &&
      provenance.resolvedTipId === target.resolvedTipId &&
      provenance.displayRevision === target.displayRevision
  )
}

/**
 * Paint only an exact v3 cache entry. This has no await by design: a verified
 * persisted transcript must become visible before session routing or resume can
 * yield to the runtime path.
 */
export function paintVerifiedTranscriptTail(
  target: PersistedDisplayHydrationTarget,
  deps: Pick<PersistedDisplayHydrationDependencies, 'commit' | 'loadVerifiedTranscriptTail'>
): PersistedDisplayHydrationPaint {
  const cached = deps.loadVerifiedTranscriptTail(target)

  if (!cached || !exactCacheProof(target, cached.provenance)) {
    return { displayRevision: null, hit: false }
  }

  deps.commit(cached.messages, cached.provenance)
  markSessionOpen('hermes.session.cache.commit')

  if (cached.provenance.coverage === 'latest-page') {
    recordTranscriptTail(
      target.storedSessionId,
      { messages: cached.messages, pagination: cached.pagination },
      { connectionId: target.connectionId, profile: target.profile }
    )
  }

  return {
    displayRevision: cached.provenance.coverage === 'latest-page' ? cached.provenance.displayRevision : null,
    hit: true
  }
}

function provenanceFromResponse(
  response: SessionMessagesResponse,
  target: PersistedDisplayHydrationTarget
): PersistedDisplayTranscriptProvenance | null {
  const displayRevision = response.display_revision
  const lineageRootId = response.lineage_root_id?.trim()
  const resolvedTipId = response.resolved_tip_id?.trim()

  if (
    typeof displayRevision !== 'number' ||
    !Number.isFinite(displayRevision) ||
    !Number.isInteger(displayRevision) ||
    displayRevision < 0 ||
    !lineageRootId ||
    !resolvedTipId
  ) {
    // Older backends legitimately return a durable transcript without a v3
    // proof. Publish it, but never invent proof or cache it as verified.
    return null
  }

  return {
    connectionId: target.connectionId,
    coverage: 'latest-page',
    displayRevision,
    lineageRootId,
    profile: target.profile,
    resolvedTipId,
    source: 'persisted-display',
    storedSessionId: target.storedSessionId
  }
}

function unchangedResponseMatchesExactPaint(
  response: SessionMessagesResponse,
  target: PersistedDisplayHydrationTarget,
  paint: PersistedDisplayHydrationPaint
): boolean {
  const displayRevision = response.display_revision

  return Boolean(
    paint.hit &&
      paint.displayRevision !== null &&
      target.displayRevision !== null &&
      paint.displayRevision === target.displayRevision &&
      typeof displayRevision === 'number' &&
      Number.isInteger(displayRevision) &&
      displayRevision === target.displayRevision &&
      response.lineage_root_id?.trim() === target.lineageRootId &&
      response.resolved_tip_id?.trim() === target.resolvedTipId &&
      response.session_id?.trim() === target.resolvedTipId
  )
}

/**
 * Starts persisted-display hydration without coupling it to runtime resume.
 * The response is gated both before and after exactly one paint frame, so a
 * route/edit/rewind that invalidates its authority cannot publish stale data.
 */
export function startPersistedDisplayHydration(
  target: PersistedDisplayHydrationTarget,
  deps: PersistedDisplayHydrationDependencies,
  paint: PersistedDisplayHydrationPaint
): Promise<PersistedDisplayHydrationResult | null> {
  return (async () => {
    try {
      const response = await deps.fetchLatest(target, paint.hit ? paint.displayRevision ?? undefined : undefined)

      if (!deps.isCurrent()) {
        return null
      }

      // `unchanged` only means "reuse the currently painted array" when the
      // request was conditioned on a verified exact cache hit and the response
      // repeats that same revision/root/tip proof. A miss sends no revision,
      // and an incomplete or contradictory proof is never authoritative.
      if (response.unchanged === true) {
        return unchangedResponseMatchesExactPaint(response, target, paint)
          ? { kind: 'unchanged' }
          : { kind: 'inconclusive' }
      }

      const provenance = provenanceFromResponse(response, target)

      // A pre-v3 backend cannot prove that an empty page is authoritative. If
      // its list row says history exists, preserve the current display and let
      // a later read establish truth. A v3 proof is different: a matching
      // revision/root/tip explicitly authorizes an empty durable transcript.
      if (response.messages.length === 0 && target.expectsPersistedHistory && !provenance) {
        return { kind: 'inconclusive' }
      }

      const rawPageMessages = toChatMessages(response.messages)
      const messages = deps.reconcile(response, deps.readCurrentMessages())

      await deps.nextFrame()

      if (!deps.isCurrent()) {
        return null
      }

      deps.commit(messages, provenance)
      markSessionOpen('hermes.session.rest.commit')

      return { kind: 'published', messages, provenance, rawPageMessages, response }
    } catch {
      // The cache paint (if any) remains usable; runtime resume owns its own
      // failure handling and must not be coupled to a display read failure.
      return null
    }
  })()
}
