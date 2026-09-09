import { describe, expect, it, vi } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { clearTranscriptTail, transcriptTailState } from '@/store/transcript-tail'
import { deferred } from '@/test/deferred'
import type { SessionMessagesResponse } from '@/types/hermes'

import type { PersistedDisplayTranscriptProvenance } from '../../../types'
import { applyRewindOptimistic } from '../use-prompt-actions/rewind'

import {
  paintVerifiedTranscriptTail,
  type PersistedDisplayHydrationDependencies,
  type PersistedDisplayHydrationTarget,
  startPersistedDisplayHydration
} from './persisted-display-hydration'
import { SESSION_OPEN_MARKS } from './session-open-marks'

const target: PersistedDisplayHydrationTarget = {
  connectionId: 'conn-1',
  displayRevision: 7,
  lineageRootId: 'root-1',
  profile: 'worker',
  resolvedTipId: 'tip-1',
  storedSessionId: 'stored-1'
}

const cached = [{ id: 'cached', parts: [{ text: 'cached', type: 'text' as const }], role: 'user' as const }]
const restored = [{ id: 'restored', parts: [{ text: 'restored', type: 'text' as const }], role: 'assistant' as const }]

function proof(revision = 8): PersistedDisplayTranscriptProvenance {
  return {
    connectionId: 'conn-1',
    coverage: 'latest-page',
    displayRevision: revision,
    lineageRootId: 'root-2',
    profile: 'worker',
    resolvedTipId: 'tip-2',
    source: 'persisted-display',
    storedSessionId: 'stored-1'
  }
}

function cacheProof(): PersistedDisplayTranscriptProvenance {
  return {
    ...proof(7),
    lineageRootId: 'root-1',
    resolvedTipId: 'tip-1'
  }
}

function changedResponse(overrides: Partial<SessionMessagesResponse> = {}): SessionMessagesResponse {
  return {
    display_revision: 8,
    lineage_root_id: 'root-2',
    messages: [{ content: 'restored', role: 'assistant', timestamp: 1 }],
    resolved_tip_id: 'tip-2',
    session_id: 'tip-2',
    ...overrides
  }
}

function harness(options: {
  current?: ChatMessage[]
  currentAuthority?: () => boolean
  fetch?: PersistedDisplayHydrationDependencies['fetchLatest']
  load?: () => { messages: ChatMessage[]; provenance: PersistedDisplayTranscriptProvenance } | null
  nextFrame?: () => Promise<void>
} = {}) {
  const commits: Array<[ChatMessage[], PersistedDisplayTranscriptProvenance | null]> = []
  const loadVerifiedTranscriptTail = vi.fn(options.load ?? (() => null))
  const commit = vi.fn((messages: ChatMessage[], provenance: PersistedDisplayTranscriptProvenance | null) => {
    commits.push([messages, provenance])
  })
  const reconcile = vi.fn((_response: SessionMessagesResponse, current: ChatMessage[]) =>
    current === cached ? restored : current
  )
  const fetchLatest = vi.fn<PersistedDisplayHydrationDependencies['fetchLatest']>(
    options.fetch ?? (() => Promise.resolve(changedResponse()))
  )
  const isCurrent = vi.fn(options.currentAuthority ?? (() => true))
  const nextFrame = vi.fn(options.nextFrame ?? (() => Promise.resolve()))
  const readCurrentMessages = vi.fn(() => options.current ?? cached)

  return {
    commits,
    commit,
    deps: { commit, fetchLatest, isCurrent, loadVerifiedTranscriptTail, nextFrame, readCurrentMessages, reconcile },
    fetchLatest,
    isCurrent,
    loadVerifiedTranscriptTail,
    nextFrame,
    reconcile
  }
}

describe('persisted display hydration', () => {
  it('marks only cache and REST commits at their visible publication points', async () => {
    for (const name of SESSION_OPEN_MARKS) {
      performance.clearMarks(name)
    }

    const state = harness({ load: () => ({ messages: cached, provenance: cacheProof() }) })
    const paint = paintVerifiedTranscriptTail(target, state.deps)
    await startPersistedDisplayHydration(target, state.deps, paint)

    expect(performance.getEntriesByName('hermes.session.cache.commit')).toHaveLength(1)
    expect(performance.getEntriesByName('hermes.session.rest.commit')).toHaveLength(1)
    expect(performance.getEntriesByName('hermes.session.select')).toHaveLength(0)

    for (const name of SESSION_OPEN_MARKS) {
      performance.clearMarks(name)
    }
  })

  it('paints an exact cache hit synchronously and rejects every stale authority completion', async () => {
    const state = harness({
      currentAuthority: (() => {
        let calls = 0
        return () => ++calls === 1
      })(),
      load: () => ({ messages: cached, provenance: cacheProof() })
    })

    const paint = paintVerifiedTranscriptTail(target, state.deps)
    expect(paint).toEqual({ displayRevision: 7, hit: true })
    expect(state.commit).toHaveBeenCalledWith(cached, cacheProof())

    await startPersistedDisplayHydration(target, state.deps, paint)
    expect(state.commit).toHaveBeenCalledTimes(1)
  })

  it('sends a known revision only after an exact cache hit', async () => {
    const hit = harness({ load: () => ({ messages: cached, provenance: cacheProof() }) })
    const hitPaint = paintVerifiedTranscriptTail(target, hit.deps)
    await startPersistedDisplayHydration(target, hit.deps, hitPaint)
    expect(hit.fetchLatest).toHaveBeenCalledWith(target, 7)

    const miss = harness()
    const missPaint = paintVerifiedTranscriptTail(target, miss.deps)
    await startPersistedDisplayHydration(target, miss.deps, missPaint)
    expect(miss.fetchLatest).toHaveBeenCalledWith(target, undefined)
  })

  it('paints a bounded cache suffix but does not condition REST on its revision', async () => {
    const tailProvenance = { ...cacheProof(), coverage: 'latest-page-tail' as never }
    const state = harness({ load: () => ({ messages: cached, provenance: tailProvenance }) })

    const paint = paintVerifiedTranscriptTail(target, state.deps)
    await startPersistedDisplayHydration(target, state.deps, paint)

    expect(paint).toEqual({ displayRevision: null, hit: true })
    expect(state.fetchLatest).toHaveBeenCalledWith(target, undefined)
  })

  it('restores earlier-page bookkeeping from an authoritative cached page before unchanged REST', () => {
    clearTranscriptTail(target.storedSessionId, {
      connectionId: target.connectionId,
      profile: target.profile
    })
    const state = harness({
      load: () => ({
        messages: cached,
        pagination: { limit: 1, offset: 0, order: 'latest', returned: 1 },
        provenance: cacheProof()
      })
    })

    paintVerifiedTranscriptTail(target, state.deps)

    expect(
      transcriptTailState(target.storedSessionId, {
        connectionId: target.connectionId,
        profile: target.profile
      })
    ).toMatchObject({ nextOffset: 1, possiblyTruncated: true })
  })

  it('commits changed REST without waiting for an unrelated runtime promise', async () => {
    const state = harness()

    const hydration = startPersistedDisplayHydration(target, state.deps, { hit: false, displayRevision: null })
    await hydration
    expect(state.commit).toHaveBeenCalledWith(restored, proof())
  })

  it('keeps the authoritative raw REST page separate from reconciled display rows', async () => {
    const optimistic = {
      id: 'optimistic-local',
      parts: [{ text: 'pending local turn', type: 'text' as const }],
      pending: true,
      role: 'user' as const
    }
    const rawPage = changedResponse({
      messages: [{ content: 'raw persisted row', role: 'assistant', timestamp: 11 }]
    })
    const display = [...restored, optimistic]
    const state = harness({ fetch: vi.fn().mockResolvedValue(rawPage) })
    state.reconcile.mockReturnValue(display)

    const result = await startPersistedDisplayHydration(
      target,
      state.deps,
      { hit: false, displayRevision: null }
    )

    expect(result).toMatchObject({
      kind: 'published',
      messages: display,
      rawPageMessages: [
        {
          parts: [{ text: 'raw persisted row', type: 'text' }],
          role: 'assistant',
          timestamp: 11
        }
      ]
    })
    expect(result?.kind === 'published' ? JSON.stringify(result.rawPageMessages) : '').not.toContain(
      'pending local turn'
    )
    expect(state.commit).toHaveBeenCalledWith(display, proof())
  })

  it('waits exactly one frame and validates route authority on both sides of it', async () => {
    const state = harness()

    await startPersistedDisplayHydration(target, state.deps, { hit: false, displayRevision: null })

    expect(state.isCurrent).toHaveBeenCalledTimes(2)
    expect(state.nextFrame).toHaveBeenCalledTimes(1)
  })

  it('does not publish a changed REST response after the route becomes stale', async () => {
    const state = harness({ currentAuthority: (() => {
      let calls = 0
      return () => ++calls === 1
    })() })

    await startPersistedDisplayHydration(target, state.deps, { hit: false, displayRevision: null })

    expect(state.commit).not.toHaveBeenCalled()
  })

  it('rejects hydration when a rewind changes authority after fetch', async () => {
    let authority = createClientSessionState('stored-1', cached)
    authority.transcriptAuthorityEpoch = 3
    const epochAtStart = authority.transcriptAuthorityEpoch
    const state = harness({
      currentAuthority: () => authority.transcriptAuthorityEpoch === epochAtStart,
      fetch: async () => {
        authority = applyRewindOptimistic(authority, 0)

        return changedResponse()
      }
    })

    await startPersistedDisplayHydration(target, state.deps, { hit: false, displayRevision: null })

    expect(state.nextFrame).not.toHaveBeenCalled()
    expect(state.commit).not.toHaveBeenCalled()
  })

  it('rejects hydration when a rewind changes authority before the paint frame', async () => {
    let authority = createClientSessionState('stored-1', cached)
    authority.transcriptAuthorityEpoch = 3
    const epochAtStart = authority.transcriptAuthorityEpoch
    const state = harness({
      currentAuthority: () => authority.transcriptAuthorityEpoch === epochAtStart,
      nextFrame: async () => {
        authority = applyRewindOptimistic(authority, 0)
      }
    })

    await startPersistedDisplayHydration(target, state.deps, { hit: false, displayRevision: null })

    expect(state.nextFrame).toHaveBeenCalledTimes(1)
    expect(state.commit).not.toHaveBeenCalled()
  })

  it('preserves message-array identity on an exact unchanged response without reconciling', async () => {
    const state = harness({ fetch: vi.fn().mockResolvedValue(changedResponse({
      display_revision: 7,
      lineage_root_id: 'root-1',
      messages: [],
      resolved_tip_id: 'tip-1',
      session_id: 'tip-1',
      unchanged: true
    })) })

    const paint = { displayRevision: 7, hit: true } as const

    const result = await startPersistedDisplayHydration(target, state.deps, paint)

    expect(result).toEqual({ kind: 'unchanged' })
    expect(state.reconcile).not.toHaveBeenCalled()
    expect(state.commit).not.toHaveBeenCalled()
  })

  it.each([
    ['display revision', { display_revision: 8 }],
    ['lineage root', { lineage_root_id: 'root-other' }],
    ['resolved tip', { resolved_tip_id: 'tip-other' }],
    ['session identity', { session_id: 'tip-other' }]
  ])('rejects an unchanged response with contradictory %s proof', async (_field, contradiction) => {
    const response = changedResponse({
      display_revision: 7,
      lineage_root_id: 'root-1',
      messages: [],
      resolved_tip_id: 'tip-1',
      session_id: 'tip-1',
      unchanged: true,
      ...contradiction
    })

    const state = harness({ fetch: vi.fn().mockResolvedValue(response) })

    const result = await startPersistedDisplayHydration(
      target,
      state.deps,
      { displayRevision: 7, hit: true }
    )

    expect(result).toEqual({ kind: 'inconclusive' })
    expect(state.reconcile).not.toHaveBeenCalled()
    expect(state.commit).not.toHaveBeenCalled()
  })

  it('reports an unchanged response on a cache miss as inconclusive instead of accepting a blank display', async () => {
    const state = harness({ fetch: vi.fn().mockResolvedValue(changedResponse({ messages: [], unchanged: true })) })

    const result = await startPersistedDisplayHydration(target, state.deps, { hit: false, displayRevision: null })

    expect(result).toEqual({ kind: 'inconclusive' })
    expect(state.reconcile).not.toHaveBeenCalled()
    expect(state.commit).not.toHaveBeenCalled()
  })

  it('keeps a cache paint when REST fails', async () => {
    const state = harness({
      fetch: vi.fn().mockRejectedValue(new Error('offline')),
      load: () => ({ messages: cached, provenance: cacheProof() })
    })
    const paint = paintVerifiedTranscriptTail(target, state.deps)

    await startPersistedDisplayHydration(target, state.deps, paint)
    expect(state.commit).toHaveBeenCalledTimes(1)
    expect(state.commit).toHaveBeenCalledWith(cached, cacheProof())
  })

  it('derives proof from response lineage, tip, and revision rather than stale list metadata', async () => {
    const state = harness()

    await startPersistedDisplayHydration(target, state.deps, { hit: false, displayRevision: null })

    expect(state.commit).toHaveBeenCalledWith(restored, proof(8))
  })

  it('publishes a legacy changed response as explicitly unproven', async () => {
    const state = harness({ fetch: vi.fn().mockResolvedValue(changedResponse({
      display_revision: undefined,
      lineage_root_id: undefined,
      resolved_tip_id: undefined
    })) })

    await startPersistedDisplayHydration(target, state.deps, { hit: false, displayRevision: null })

    expect(state.commit).toHaveBeenCalledWith(restored, null)
  })

  it('does not let an unproven empty legacy page erase a listed transcript', async () => {
    const state = harness({
      current: [],
      fetch: vi.fn().mockResolvedValue(changedResponse({
        display_revision: undefined,
        lineage_root_id: undefined,
        messages: [],
        resolved_tip_id: undefined
      }))
    })

    const listedTarget = { ...target, displayRevision: null, expectsPersistedHistory: true }
    const result = await startPersistedDisplayHydration(
      listedTarget,
      state.deps,
      { hit: false, displayRevision: null }
    )

    expect(result).toEqual({ kind: 'inconclusive' })
    expect(state.reconcile).not.toHaveBeenCalled()
    expect(state.commit).not.toHaveBeenCalled()
  })

  it('accepts an authoritative revisioned empty page and keeps it out of the tail cache path', async () => {
    const state = harness({
      current: [],
      fetch: vi.fn().mockResolvedValue(changedResponse({ messages: [] }))
    })

    const listedTarget = { ...target, expectsPersistedHistory: true }
    const result = await startPersistedDisplayHydration(
      listedTarget,
      state.deps,
      { hit: false, displayRevision: null }
    )

    expect(result).toMatchObject({ kind: 'published', messages: [], provenance: proof(8) })
    expect(state.commit).toHaveBeenCalledWith([], proof(8))
  })

  it('rejects a late A hydration after the user has opened session B', async () => {
    const pendingA = deferred<SessionMessagesResponse>()
    const commits: string[] = []
    let selected = 'A'
    const hydrationFor = (storedSessionId: string): PersistedDisplayHydrationDependencies => ({
      commit: messages => {
        commits.push(messages[0]?.parts[0]?.type === 'text' ? messages[0].parts[0].text : '')
      },
      fetchLatest: requestTarget =>
        requestTarget.storedSessionId === 'A'
          ? pendingA.promise
          : Promise.resolve({
              display_revision: 2,
              lineage_root_id: 'B',
              messages: [{ content: 'session B', role: 'assistant', timestamp: 1 }],
              resolved_tip_id: 'B',
              session_id: 'B'
            }),
      isCurrent: () => selected === storedSessionId,
      loadVerifiedTranscriptTail: () => null,
      nextFrame: () => Promise.resolve(),
      readCurrentMessages: () => [],
      reconcile: response => [
        {
          id: response.session_id,
          parts: [{ text: String(response.messages[0]?.content ?? ''), type: 'text' }],
          role: 'assistant'
        }
      ]
    })
    const targetA = { ...target, storedSessionId: 'A' }
    const targetB = {
      ...target,
      displayRevision: 1,
      lineageRootId: 'B',
      resolvedTipId: 'B',
      storedSessionId: 'B'
    }

    const openingA = startPersistedDisplayHydration(targetA, hydrationFor('A'), { hit: false, displayRevision: null })
    selected = 'B'
    await startPersistedDisplayHydration(targetB, hydrationFor('B'), { hit: false, displayRevision: null })
    pendingA.resolve({
      display_revision: 2,
      lineage_root_id: 'A',
      messages: [{ content: 'session A', role: 'assistant', timestamp: 1 }],
      resolved_tip_id: 'A',
      session_id: 'A'
    })
    await openingA

    expect(commits).toEqual(['session B'])
  })
})
