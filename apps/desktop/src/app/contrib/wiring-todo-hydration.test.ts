import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getLatestSessionMessages, getSessionMessages } from '@/hermes'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $statusItemsBySession } from '@/store/composer-status'
import { _resetSessionOwnerHintsForTests, setSessionOwnerHint, setSessions } from '@/store/session'
import { $sessionStates, $sessionTiles } from '@/store/session-states'
import {
  $todosBySession,
  _todoHydrationAuthorityStatsForTests,
  applyTodoContinuationSnapshot,
  captureTodoWriteFence,
  clearAllSessionTodos,
  clearAllTodoContinuations,
  setSessionTodos
} from '@/store/todos'
import { deferred } from '@/test/deferred'
import {
  LEGACY_TODO_CARRIER_277757,
  LEGACY_TODOS_277757,
  legacyEvidenceMessage277757
} from '@/test/evidence-row-277757'

import {
  hydrateSessionTodos,
  hydrateStoredSessionMessages,
  resolveStoredSessionTodoMessages
} from './wiring-todo-hydration'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getLatestSessionMessages: vi.fn(),
  getSessionMessages: vi.fn()
}))

const runtimeSessionId = 'runtime-target'
const carrierTodos = [
  { content: 'parent plan', id: 'plan', status: 'completed' as const },
  { content: 'active child', id: 'child', parent: 'plan', status: 'in_progress' as const },
  { content: 'cancelled parent', id: 'skip', status: 'cancelled' as const },
  { content: 'next child', id: 'next', parent: 'skip', status: 'pending' as const }
]

const unfinishedHistory = [
  {
    parts: [
      {
        args: { todos: [{ content: 'finish the task', id: 'task', status: 'in_progress' }] },
        toolCallId: 'todo-1',
        toolName: 'todo',
        type: 'tool-call'
      }
    ]
  }
]

const ordinaryFullTailMessages = Array.from({ length: 120 }, (_, index) => ({
  content: `ordinary post-turn tail ${index}`,
  role: 'assistant' as const,
  timestamp: index + 1
}))

type PostTurnHydration = (options: {
  attempts?: number
  isCurrent?: () => boolean
  publishTranscript?: (messages: ReturnType<typeof hydrateStoredSessionMessages>) => void
  runtimeSessionId: string
  storedSessionId: string
}) => Promise<null | ReturnType<typeof hydrateStoredSessionMessages>>

async function postTurnHydration(): Promise<PostTurnHydration> {
  const module = await import('./wiring-todo-hydration')
  const candidate = Reflect.get(module, 'hydratePostTurnStoredSession')

  expect(candidate).toBeTypeOf('function')

  return candidate as PostTurnHydration
}

function bindRuntime(runtimeId: string, storedSessionId: string): void {
  $sessionStates.set({ [runtimeId]: createClientSessionState(storedSessionId) })
}

function mockAuthoritativePostTurnState(): void {
  vi.mocked(getLatestSessionMessages).mockResolvedValue({
    messages: ordinaryFullTailMessages,
    session_id: 'stored-post-turn'
  } as never)
  vi.mocked(getSessionMessages).mockResolvedValue({
    messages: [legacyEvidenceMessage277757],
    pagination: { exhausted: true, has_more: false, limit: 32, next_before_id: null, returned: 1 },
    session_id: 'stored-post-turn'
  } as never)
}

beforeEach(() => {
  vi.clearAllMocks()
  setSessions([])
  $sessionStates.set({})
  $sessionTiles.set([])
  _resetSessionOwnerHintsForTests({ storage: true })
})

afterEach(() => {
  clearAllSessionTodos()
  clearAllTodoContinuations()
  setSessions([])
  $sessionStates.set({})
  $sessionTiles.set([])
  _resetSessionOwnerHintsForTests({ storage: true })
  vi.useRealTimers()
})

describe('wiring todo hydration caller', () => {
  it('bounds total requests and retained payload while an old valid carrier remains reachable', async () => {
    let legacyCandidatePage = 0

    vi.mocked(getSessionMessages).mockImplementation((_id, _profile, options) => {
      if (options?.projection === ('todo-state' as never)) {
        return Promise.resolve({
          messages: [legacyEvidenceMessage277757],
          pagination: { exhausted: true, has_more: false, limit: 2, next_before_id: null, returned: 1 },
          session_id: 'stored-long-history'
        } as never)
      }

      const page = legacyCandidatePage

      legacyCandidatePage += 1

      if (page < 8) {
        return Promise.resolve({
          messages: Array.from({ length: 32 }, (_, index) => ({
            content: `malformed tool-heavy candidate ${page}-${index} ${'x'.repeat(2_000)}`,
            role: 'assistant' as const,
            tool_calls: [
              {
                function: { arguments: '{}', name: 'not-todo' },
                id: `noise-${page}-${index}`,
                type: 'function'
              }
            ]
          })),
          pagination: {
            exhausted: false,
            has_more: true,
            limit: 32,
            next_before_id: 10_000 - (page + 1) * 32,
            returned: 32
          },
          session_id: 'stored-long-history'
        } as never)
      }

      return Promise.resolve({
        messages: [legacyEvidenceMessage277757],
        pagination: { exhausted: true, has_more: false, limit: 32, next_before_id: null, returned: 1 },
        session_id: 'stored-long-history'
      } as never)
    })

    const messages = await resolveStoredSessionTodoMessages('stored-long-history', undefined, [])

    expect(getSessionMessages).toHaveBeenCalledTimes(1)
    expect(getSessionMessages).toHaveBeenCalledWith('stored-long-history', undefined, {
      projection: 'todo-state'
    })
    expect(messages).toEqual([legacyEvidenceMessage277757])
    expect(messages).toHaveLength(1)
    expect(JSON.stringify(messages).length).toBeLessThan(7_000)
  })

  it.each([
    ['CJK', '界'.repeat(400_000)],
    ['supplementary-plane', '😀'.repeat(280_000)]
  ])('rejects a %s response whose UTF-8 bytes exceed the renderer boundary', async (_label, content) => {
    const messages = [{ content, role: 'assistant' as const }]

    expect(JSON.stringify(messages).length).toBeLessThan(1_100_000)
    expect(new TextEncoder().encode(JSON.stringify(messages)).byteLength).toBeGreaterThan(1_100_000)
    vi.mocked(getSessionMessages).mockResolvedValue({
      messages,
      pagination: { exhausted: true, has_more: false, limit: 2, next_before_id: null, returned: 1 },
      session_id: 'stored-multibyte'
    } as never)

    await expect(resolveStoredSessionTodoMessages('stored-multibyte', undefined, [])).rejects.toThrow(
      'Todo-state projection exceeded its bounded response contract'
    )
  })

  it('accepts a multibyte response below the UTF-8 renderer boundary', async () => {
    const messages = [{ content: '界'.repeat(350_000), role: 'assistant' as const }]

    expect(new TextEncoder().encode(JSON.stringify(messages)).byteLength).toBeLessThan(1_100_000)
    vi.mocked(getSessionMessages).mockResolvedValue({
      messages,
      pagination: { exhausted: true, has_more: false, limit: 2, next_before_id: null, returned: 1 },
      session_id: 'stored-multibyte-control'
    } as never)

    await expect(resolveStoredSessionTodoMessages('stored-multibyte-control', undefined, [])).resolves.toEqual(messages)
  })

  it('post-turn hydration keeps the exact remote connection and target profile through bounded discovery', async () => {
    const ownerRoute = {
      connectionId: 'remote-owner',
      mode: 'remote' as const,
      profile: 'desktop-profile',
      targetProfile: 'backend-profile'
    }

    bindRuntime('runtime-post-turn', 'stored-post-turn')
    setSessionOwnerHint('stored-post-turn', ownerRoute)
    mockAuthoritativePostTurnState()

    const hydratePostTurn = await postTurnHydration()
    await hydratePostTurn({ runtimeSessionId: 'runtime-post-turn', storedSessionId: 'stored-post-turn' })

    const scope = { connectionId: 'remote-owner', profile: 'backend-profile' }

    expect(getLatestSessionMessages).toHaveBeenCalledWith('stored-post-turn', scope)
    expect(getSessionMessages).toHaveBeenCalledWith('stored-post-turn', scope, {
      projection: 'todo-state'
    })
    expect($todosBySession.get()['runtime-post-turn']).toEqual(LEGACY_TODOS_277757)
  })

  it.each([
    [
      'profile',
      {
        connectionId: 'remote-owner-a',
        mode: 'remote' as const,
        profile: 'desktop-profile-a',
        targetProfile: 'backend-profile-a'
      },
      {
        connectionId: 'remote-owner-a',
        mode: 'remote' as const,
        profile: 'desktop-profile-b',
        targetProfile: 'backend-profile-b'
      }
    ],
    [
      'connectionId',
      {
        connectionId: 'remote-owner-a',
        mode: 'remote' as const,
        profile: 'desktop-profile',
        targetProfile: 'backend-profile'
      },
      {
        connectionId: 'remote-owner-b',
        mode: 'remote' as const,
        profile: 'desktop-profile',
        targetProfile: 'backend-profile'
      }
    ]
  ] as const)(
    'rejects a post-turn Todo response after the same runtime rebinds its %s and permits the new owner retry',
    async (_field, ownerA, ownerB) => {
      const projection = deferred<Awaited<ReturnType<typeof getSessionMessages>>>()

      bindRuntime('runtime-post-turn', 'stored-post-turn')
      $sessionTiles.set([
        { ownerRoute: ownerA, runtimeId: 'runtime-post-turn', storedSessionId: 'stored-post-turn' }
      ] as never)
      vi.mocked(getLatestSessionMessages).mockResolvedValue({
        messages: ordinaryFullTailMessages,
        session_id: 'stored-post-turn'
      } as never)
      vi.mocked(getSessionMessages).mockReturnValueOnce(projection.promise)

      const hydratePostTurn = await postTurnHydration()
      const staleHydration = hydratePostTurn({
        runtimeSessionId: 'runtime-post-turn',
        storedSessionId: 'stored-post-turn'
      })

      await vi.waitFor(() => expect(getSessionMessages).toHaveBeenCalledTimes(1))
      $sessionTiles.set([
        { ownerRoute: ownerB, runtimeId: 'runtime-post-turn', storedSessionId: 'stored-post-turn' }
      ] as never)
      projection.resolve({
        messages: [legacyEvidenceMessage277757],
        pagination: { exhausted: true, has_more: false, limit: 2, next_before_id: null, returned: 1 },
        session_id: 'stored-post-turn'
      } as never)

      await expect(staleHydration).resolves.toBeNull()
      expect($todosBySession.get()['runtime-post-turn']).toBeUndefined()
      expect(_todoHydrationAuthorityStatsForTests()).toMatchObject({
        activeSessionCount: 0,
        activeTokenCount: 0,
        latestSessionCount: 0
      })

      mockAuthoritativePostTurnState()
      const currentHydration = await hydratePostTurn({
        runtimeSessionId: 'runtime-post-turn',
        storedSessionId: 'stored-post-turn'
      })

      expect(currentHydration).not.toBeNull()
      expect($todosBySession.get()['runtime-post-turn']).toEqual(LEGACY_TODOS_277757)
      expect(getLatestSessionMessages).toHaveBeenLastCalledWith('stored-post-turn', {
        connectionId: ownerB.connectionId,
        profile: ownerB.targetProfile
      })
    }
  )

  it('rejects a superseded post-turn request immediately before Todo and transcript publication', async () => {
    const projection = deferred<Awaited<ReturnType<typeof getSessionMessages>>>()
    const publishTranscript = vi.fn()
    let currentRequest = 1

    bindRuntime('runtime-post-turn', 'stored-post-turn')
    setSessionOwnerHint('stored-post-turn', {
      connectionId: 'remote-generation',
      profile: 'desktop-generation',
      targetProfile: 'backend-generation'
    })
    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: ordinaryFullTailMessages,
      session_id: 'stored-post-turn'
    } as never)
    vi.mocked(getSessionMessages).mockReturnValueOnce(projection.promise)

    const hydratePostTurn = await postTurnHydration()
    const staleHydration = hydratePostTurn({
      isCurrent: () => currentRequest === 1,
      publishTranscript,
      runtimeSessionId: 'runtime-post-turn',
      storedSessionId: 'stored-post-turn'
    })

    await vi.waitFor(() => expect(getSessionMessages).toHaveBeenCalledTimes(1))
    currentRequest = 2
    projection.resolve({
      messages: [legacyEvidenceMessage277757],
      pagination: { exhausted: true, has_more: false, limit: 2, next_before_id: null, returned: 1 },
      session_id: 'stored-post-turn'
    } as never)

    await expect(staleHydration).resolves.toBeNull()
    expect(publishTranscript).not.toHaveBeenCalled()
    expect($todosBySession.get()['runtime-post-turn']).toBeUndefined()
    expect(_todoHydrationAuthorityStatsForTests()).toMatchObject({
      activeSessionCount: 0,
      activeTokenCount: 0,
      latestSessionCount: 0
    })

    mockAuthoritativePostTurnState()
    const currentHydration = await hydratePostTurn({
      isCurrent: () => currentRequest === 2,
      publishTranscript,
      runtimeSessionId: 'runtime-post-turn',
      storedSessionId: 'stored-post-turn'
    })

    expect(currentHydration).not.toBeNull()
    expect(publishTranscript).toHaveBeenCalledTimes(1)
    expect($todosBySession.get()['runtime-post-turn']).toEqual(LEGACY_TODOS_277757)
  })

  it.each([
    ['secondary local profile', 'research', 'research'],
    ['unknown legacy owner', null, undefined]
  ] as const)('post-turn hydration preserves %s fallback routing', async (_label, storedProfile, expectedScope) => {
    bindRuntime('runtime-post-turn', 'stored-post-turn')

    if (storedProfile) {
      setSessions([{ id: 'stored-post-turn', profile: storedProfile }] as never)
    }

    mockAuthoritativePostTurnState()

    const hydratePostTurn = await postTurnHydration()
    await hydratePostTurn({ runtimeSessionId: 'runtime-post-turn', storedSessionId: 'stored-post-turn' })

    expect(getLatestSessionMessages).toHaveBeenCalledWith('stored-post-turn', expectedScope)
    expect(getSessionMessages).toHaveBeenCalledWith('stored-post-turn', expectedScope, {
      projection: 'todo-state'
    })
  })

  it('post-turn retries retain the same exact owner scope', async () => {
    vi.useFakeTimers()
    bindRuntime('runtime-post-turn', 'stored-post-turn')
    setSessionOwnerHint('stored-post-turn', {
      connectionId: 'remote-retry',
      profile: 'desktop-retry',
      targetProfile: 'backend-retry'
    })
    vi.mocked(getLatestSessionMessages)
      .mockRejectedValueOnce(new Error('transient post-turn read failure'))
      .mockResolvedValueOnce({ messages: ordinaryFullTailMessages, session_id: 'stored-post-turn' } as never)
    vi.mocked(getSessionMessages).mockResolvedValue({
      messages: [legacyEvidenceMessage277757],
      pagination: { exhausted: true, has_more: false, limit: 32, next_before_id: null, returned: 1 },
      session_id: 'stored-post-turn'
    } as never)

    const hydratePostTurn = await postTurnHydration()
    const hydration = hydratePostTurn({
      attempts: 2,
      runtimeSessionId: 'runtime-post-turn',
      storedSessionId: 'stored-post-turn'
    })

    await vi.runAllTimersAsync()
    await hydration

    const scope = { connectionId: 'remote-retry', profile: 'backend-retry' }

    expect(getLatestSessionMessages).toHaveBeenNthCalledWith(1, 'stored-post-turn', scope)
    expect(getLatestSessionMessages).toHaveBeenNthCalledWith(2, 'stored-post-turn', scope)
    expect($todosBySession.get()['runtime-post-turn']).toEqual(LEGACY_TODOS_277757)
  })

  it('post-turn same-ID collision follows the runtime-bound tile owner', async () => {
    setSessions([
      { connection_id: 'remote-a', id: 'stored-post-turn', profile: 'profile-a' },
      { connection_id: 'remote-b', id: 'stored-post-turn', profile: 'profile-b' }
    ] as never)
    $sessionTiles.set([
      {
        ownerRoute: {
          connectionId: 'remote-b',
          mode: 'remote',
          profile: 'profile-b',
          targetProfile: 'backend-b'
        },
        runtimeId: 'runtime-b',
        storedSessionId: 'stored-post-turn'
      }
    ] as never)
    mockAuthoritativePostTurnState()

    const hydratePostTurn = await postTurnHydration()
    await hydratePostTurn({ runtimeSessionId: 'runtime-b', storedSessionId: 'stored-post-turn' })

    const scope = { connectionId: 'remote-b', profile: 'backend-b' }

    expect(getLatestSessionMessages).toHaveBeenCalledWith('stored-post-turn', scope)
    expect(getSessionMessages).toHaveBeenCalledWith('stored-post-turn', scope, {
      projection: 'todo-state'
    })
  })

  it('post-turn clears stale Todo state only after genuine candidate exhaustion', async () => {
    bindRuntime('runtime-post-turn', 'stored-post-turn')
    setSessionTodos('runtime-post-turn', [{ content: 'stale post-turn Todo', id: 'stale', status: 'in_progress' }])
    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: ordinaryFullTailMessages,
      session_id: 'stored-post-turn'
    } as never)
    vi.mocked(getSessionMessages).mockResolvedValue({
      messages: [],
      pagination: { exhausted: true, has_more: false, limit: 32, next_before_id: null, returned: 0 },
      session_id: 'stored-post-turn'
    } as never)

    const hydratePostTurn = await postTurnHydration()
    await hydratePostTurn({ runtimeSessionId: 'runtime-post-turn', storedSessionId: 'stored-post-turn' })

    expect(getSessionMessages).toHaveBeenCalledWith('stored-post-turn', undefined, {
      projection: 'todo-state'
    })
    expect($todosBySession.get()['runtime-post-turn']).toBeUndefined()
  })

  it.each([
    ['decoded metadata', { todo_snapshot: { todos: carrierTodos } }],
    ['raw JSON metadata', JSON.stringify({ todo_snapshot: { todos: carrierTodos } })]
  ])('restores carrier-only structured state from %s', (_label, displayMetadata) => {
    hydrateSessionTodos(runtimeSessionId, [
      {
        content: 'opaque todo carrier',
        display_kind: 'hidden',
        display_metadata: displayMetadata,
        role: 'user'
      }
    ])

    expect($todosBySession.get()[runtimeSessionId]).toEqual(carrierTodos)
  })

  it('hydrates post-turn state from the raw carrier before transcript projection', () => {
    const messages = hydrateStoredSessionMessages(runtimeSessionId, [
      {
        content:
          '[Your active task list was preserved across context compression]\n- [>] child. active child (in_progress)',
        display_kind: 'hidden',
        display_metadata: { todo_snapshot: { todos: carrierTodos } },
        role: 'user'
      }
    ])

    expect(messages).toEqual([])
    expect($todosBySession.get()[runtimeSessionId]).toEqual(carrierTodos)
  })

  it('projects legacy standalone carrier prose into structured todo state', () => {
    hydrateSessionTodos(runtimeSessionId, [
      {
        content: [
          '[Your active task list was preserved across context compression]',
          '- [x] plan. parent plan (completed)',
          '  - [>] child. active child (in_progress)',
          '- [~] skip. cancelled parent (cancelled)',
          '  - [ ] next. next child (pending)',
          '',
          '[Skills pruned during compression — reload before acting on these tasks]',
          "The task list above crossed the compression boundary verbatim, but the skill instructions that governed it were pruned. Before executing any preserved task that depends on these skills, reload them first: skill_view(name='software-development/systematic-debugging'). After reloading, re-check that each pending task is still justified — findings recorded before the boundary may have invalidated it."
        ].join('\n'),
        display_kind: null,
        role: 'user'
      }
    ])

    expect($todosBySession.get()[runtimeSessionId]).toEqual(carrierTodos)
    expect($statusItemsBySession.get()[runtimeSessionId]?.filter(item => item.type === 'todo')).toHaveLength(4)
    expect($statusItemsBySession.get()[runtimeSessionId]?.every(item => item.todoPresentation === 'restored')).toBe(
      true
    )
  })

  it('replays evidence row 277757 after post-turn hydration and immediately restores the missing panel', () => {
    expect(LEGACY_TODO_CARRIER_277757).toHaveLength(3067)

    const messages = hydrateStoredSessionMessages(runtimeSessionId, [legacyEvidenceMessage277757])

    expect(messages).toEqual([])
    expect($todosBySession.get()[runtimeSessionId]).toEqual(LEGACY_TODOS_277757)
    expect($statusItemsBySession.get()[runtimeSessionId]).toHaveLength(11)
    expect($statusItemsBySession.get()[runtimeSessionId]?.every(item => item.todoPresentation === 'restored')).toBe(
      true
    )
  })

  it.each(['active', 'paused'] as const)(
    'preserves unfinished history for the exact runtime when continuation is %s',
    state => {
      applyTodoContinuationSnapshot(runtimeSessionId, { revision: 1, state })

      hydrateSessionTodos(runtimeSessionId, unfinishedHistory)

      expect($todosBySession.get()[runtimeSessionId]).toEqual([
        { content: 'finish the task', id: 'task', status: 'in_progress' }
      ])
    }
  )

  it('does not use another runtime continuation to restore unfinished history', () => {
    applyTodoContinuationSnapshot('runtime-other', { revision: 1, state: 'active' })

    hydrateSessionTodos(runtimeSessionId, unfinishedHistory)

    expect($todosBySession.get()[runtimeSessionId]).toBeUndefined()
  })

  it.each(['absent', 'none'] as const)('clears stale unfinished history when continuation is %s', state => {
    if (state === 'none') {
      applyTodoContinuationSnapshot(runtimeSessionId, { revision: 1, state: 'none' })
    }

    hydrateSessionTodos(runtimeSessionId, unfinishedHistory)

    expect($todosBySession.get()[runtimeSessionId]).toBeUndefined()
  })

  it('accepts stored hydration after an older finished-list linger fires while the read is pending', async () => {
    vi.useFakeTimers()
    setSessionTodos(runtimeSessionId, [{ content: 'finished task', id: 'finished', status: 'completed' }])
    applyTodoContinuationSnapshot(runtimeSessionId, { revision: 1, state: 'active' })

    const read = deferred<typeof unfinishedHistory>()
    const token = captureTodoWriteFence(runtimeSessionId)
    const hydration = read.promise.then(messages => hydrateSessionTodos(runtimeSessionId, messages, token))

    vi.advanceTimersByTime(5_000)
    expect($todosBySession.get()[runtimeSessionId]).toBeUndefined()

    read.resolve(unfinishedHistory)
    await hydration

    expect($todosBySession.get()[runtimeSessionId]).toEqual([
      { content: 'finish the task', id: 'task', status: 'in_progress' }
    ])
  })
})
