/**
 * The shared canonical agent-chat identity-resolution flow — the ONE
 * implementation both Bot Mode and the SDK's `host.openCanonicalAgentChat`
 * invoke. Exercises every failure/authorization/concurrency path the
 * resolver is responsible for; caller-specific presentation (workspace
 * scoping, kickoff, tile navigation) is NOT tested here — those live in
 * `hermes-bots/canonical-chat.test.ts`-family files and the SDK's own
 * `profile-routing.test.ts`, which supply real callback implementations and
 * assert the resolver is what they call.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  type AuthorizedRoster,
  type CanonicalAgentChatCallbacks,
  type CanonicalAgentChatTarget,
  isAuthorizedCanonicalChatTarget,
  isCanonicalAgentChatRow,
  isTitleConflictError,
  resolveCanonicalAgentChat,
} from './canonical-agent-chat'

const LOCAL_TARGET: CanonicalAgentChatTarget = { connectionId: null, profile: 'architect', targetProfile: 'architect' }

const REMOTE_ALIASED_TARGET: CanonicalAgentChatTarget = {
  connectionId: 'mac-mini',
  profile: 'moxie',
  targetProfile: 'default',
}

function makeCallbacks(overrides: Partial<CanonicalAgentChatCallbacks> = {}): CanonicalAgentChatCallbacks {
  return {
    lookup: vi.fn(async () => null),
    create: vi.fn(async () => ({ runtimeId: 'runtime-1', storedId: 'stored-1' })),
    titleSession: vi.fn(async () => {}),
    openExisting: vi.fn(async (_target, _openedId, _row, _canNavigate) => {}),
    openFresh: vi.fn(async (_target, _storedId, _canNavigate) => {}),
    ...overrides,
  }
}

describe('isCanonicalAgentChatRow', () => {
  it('matches by root_title when present', () => {
    expect(isCanonicalAgentChatRow({ root_title: 'Bot Chat', title: 'Something else' })).toBe(true)
  })

  it('falls back to title when root_title is absent', () => {
    expect(isCanonicalAgentChatRow({ title: 'Bot Chat' })).toBe(true)
  })

  it('rejects a non-canonical title', () => {
    expect(isCanonicalAgentChatRow({ title: 'Not Bot Chat' })).toBe(false)
  })
})

describe('isTitleConflictError', () => {
  it('recognizes an "already in use" message', () => {
    expect(isTitleConflictError(new Error('title already in use'))).toBe(true)
  })

  it('rejects an unrelated error', () => {
    expect(isTitleConflictError(new Error('network timeout'))).toBe(false)
  })
})

describe('isAuthorizedCanonicalChatTarget', () => {
  it('authorizes a target present in the current roster by the full descriptor', () => {
    const roster = { agents: [{ connectionId: null, profile: 'architect', targetProfile: 'architect' }] }

    expect(isAuthorizedCanonicalChatTarget(LOCAL_TARGET, roster)).toBe(true)
  })

  it('rejects an arbitrary non-blank target absent from the roster', () => {
    const roster = { agents: [{ connectionId: null, profile: 'someone-else', targetProfile: 'someone-else' }] }

    expect(isAuthorizedCanonicalChatTarget(LOCAL_TARGET, roster)).toBe(false)
  })

  it('rejects an empty roster', () => {
    expect(isAuthorizedCanonicalChatTarget(LOCAL_TARGET, { agents: [] })).toBe(false)
  })

  it('authorizes a target whose caller-projected profile already reflects the established alias identity', () => {
    // The roster the CALLER hands in has already projected the alias's own
    // display name into `profile` (Bot Mode's `assertAuthorizedTarget` does
    // this via `aliasIdentityFor` before calling this generic check) — the
    // raw host.agents() backend row never carries the Desktop-local alias
    // name itself, only its own backend profile.
    const roster = { agents: [{ connectionId: 'mac-mini', profile: 'moxie', targetProfile: 'default' }] }

    expect(isAuthorizedCanonicalChatTarget(REMOTE_ALIASED_TARGET, roster)).toBe(true)
  })

  it('rejects an aliased target when the roster still reports the raw backend profile, not the alias', () => {
    // Without the caller projecting the alias identity first, the generic
    // full-descriptor check must not silently authorize on targetProfile
    // alone — profile is never ignored.
    const roster = { agents: [{ connectionId: 'mac-mini', profile: 'default', targetProfile: 'default' }] }

    expect(isAuthorizedCanonicalChatTarget(REMOTE_ALIASED_TARGET, roster)).toBe(false)
  })

  it('never authorizes a mismatched connectionId even with a matching profile name', () => {
    const roster = { agents: [{ connectionId: 'other-connection', profile: 'architect', targetProfile: 'architect' }] }

    expect(isAuthorizedCanonicalChatTarget(LOCAL_TARGET, roster)).toBe(false)
  })

  it('rejects a mismatched backend targetProfile even when connectionId and profile appear valid', () => {
    const roster = { agents: [{ connectionId: null, profile: 'architect', targetProfile: 'some-other-backend-name' }] }

    expect(isAuthorizedCanonicalChatTarget(LOCAL_TARGET, roster)).toBe(false)
  })

  it('rejects a mismatched displayed profile even when connectionId and targetProfile appear valid', () => {
    const roster = { agents: [{ connectionId: null, profile: 'some-other-display-name', targetProfile: 'architect' }] }

    expect(isAuthorizedCanonicalChatTarget(LOCAL_TARGET, roster)).toBe(false)
  })

  it('rejects a roster row that omits its own targetProfile — never substitutes profile for it (Architect corrective, 2026-09-02, sixth pass)', () => {
    // A row lacking the required third field cannot vouch for ANY target,
    // even one whose profile matches exactly — collapsing a missing
    // targetProfile onto profile let such a row silently authorize. Cast
    // through `unknown` to model a caller/bridge sending a legacy or
    // malformed row at runtime (not just proving the type checker rejects
    // it) — the runtime guard is what this test actually exercises.
    const roster = {
      agents: [{ connectionId: null, profile: 'architect' }],
    } as unknown as AuthorizedRoster

    expect(isAuthorizedCanonicalChatTarget(LOCAL_TARGET, roster)).toBe(false)
  })

  it('rejects a roster row whose targetProfile is present but blank', () => {
    const roster = { agents: [{ connectionId: null, profile: 'architect', targetProfile: '   ' }] }

    expect(isAuthorizedCanonicalChatTarget(LOCAL_TARGET, roster)).toBe(false)
  })
})

describe('resolveCanonicalAgentChat', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('adopts an existing canonical session without creating', async () => {
    const callbacks = makeCallbacks({
      lookup: vi.fn(async () => ({ id: 'existing-1', title: 'Bot Chat' })),
    })

    const result = await resolveCanonicalAgentChat(LOCAL_TARGET, callbacks)

    expect(result).toEqual({ registryId: 'existing-1', openedId: 'existing-1' })
    expect(callbacks.create).not.toHaveBeenCalled()
    expect(callbacks.openExisting).toHaveBeenCalledWith(LOCAL_TARGET, 'existing-1', { id: 'existing-1', title: 'Bot Chat' }, true)
  })

  it('opens the compression-lineage tip while the registry id names the chat', async () => {
    const callbacks = makeCallbacks({
      lookup: vi.fn(async () => ({ id: 'root-1', resolved_id: 'tip-9', title: 'Bot Chat' })),
    })

    const result = await resolveCanonicalAgentChat(LOCAL_TARGET, callbacks)

    expect(result).toEqual({ registryId: 'root-1', openedId: 'tip-9' })
    expect(callbacks.openExisting).toHaveBeenCalledWith(LOCAL_TARGET, 'tip-9', expect.objectContaining({ id: 'root-1' }), true)
  })

  it('creates exactly as the current contract requires when no canonical row exists', async () => {
    const callbacks = makeCallbacks()

    const result = await resolveCanonicalAgentChat(LOCAL_TARGET, callbacks)

    expect(result).toEqual({ registryId: 'stored-1', openedId: 'stored-1' })
    expect(callbacks.create).toHaveBeenCalledWith(LOCAL_TARGET)
    expect(callbacks.titleSession).toHaveBeenCalledWith(LOCAL_TARGET, 'runtime-1')
    expect(callbacks.openFresh).toHaveBeenCalledWith(LOCAL_TARGET, 'stored-1', true)
  })

  it('rejects and performs no creation when lookup fails', async () => {
    const callbacks = makeCallbacks({
      lookup: vi.fn(async () => {
        throw new Error('Could not check registry')
      }),
    })

    await expect(resolveCanonicalAgentChat(LOCAL_TARGET, callbacks)).rejects.toThrow('Could not check registry')
    expect(callbacks.create).not.toHaveBeenCalled()
  })

  it('fails closed on a non-conflict title-persistence failure — does not return the untitled session', async () => {
    const callbacks = makeCallbacks({
      titleSession: vi.fn(async () => {
        throw new Error('database is locked')
      }),
    })

    await expect(resolveCanonicalAgentChat(LOCAL_TARGET, callbacks)).rejects.toThrow('database is locked')
    expect(callbacks.openFresh).not.toHaveBeenCalled()
  })

  it('proceeds with the untitled compat path only when the caller explicitly opts in', async () => {
    const callbacks = makeCallbacks({
      titleSession: vi.fn(async () => {
        throw new Error('unknown method session.title')
      }),
    })

    const result = await resolveCanonicalAgentChat(LOCAL_TARGET, callbacks, {
      allowUntitledCompat: async () => true,
    })

    expect(result).toEqual({ registryId: 'stored-1', openedId: 'stored-1' })
    expect(callbacks.openFresh).toHaveBeenCalledWith(LOCAL_TARGET, 'stored-1', true)
  })

  it('a real create → title conflict → re-lookup → adopt-winner scenario', async () => {
    // The FIRST lookup (adoption check) genuinely finds nothing — no
    // canonical row exists yet from this caller's point of view.
    const lookup = vi
      .fn<CanonicalAgentChatCallbacks['lookup']>()
      .mockResolvedValueOnce(null) // adoption check
      .mockResolvedValueOnce({ id: 'winner-1', resolved_id: 'winner-tip', title: 'Bot Chat' }) // post-conflict re-lookup

    const callbacks = makeCallbacks({
      lookup,
      // The create path genuinely proceeds (lookup found nothing), and its
      // OWN title write is the one that hits the conflict — a second
      // writer won the race between our lookup and our title write.
      titleSession: vi.fn(async () => {
        throw new Error('title already in use')
      }),
    })

    const result = await resolveCanonicalAgentChat(LOCAL_TARGET, callbacks)

    expect(result).toEqual({ registryId: 'winner-1', openedId: 'winner-tip' })
    expect(callbacks.create).toHaveBeenCalledTimes(1)
    expect(lookup).toHaveBeenCalledTimes(2)
    expect(callbacks.openExisting).toHaveBeenCalledWith(LOCAL_TARGET, 'winner-tip', expect.objectContaining({ id: 'winner-1' }), true)
    // The stray lazy session created before the conflict is never opened —
    // only the winner is.
    expect(callbacks.openFresh).not.toHaveBeenCalled()
  })

  it('fails closed when a title conflict is reported but re-lookup finds no canonical winner', async () => {
    const lookup = vi
      .fn<CanonicalAgentChatCallbacks['lookup']>()
      .mockResolvedValueOnce(null)
      .mockResolvedValueOnce(null) // re-lookup after conflict still finds nothing

    const callbacks = makeCallbacks({
      lookup,
      titleSession: vi.fn(async () => {
        throw new Error('title already in use')
      }),
    })

    await expect(resolveCanonicalAgentChat(LOCAL_TARGET, callbacks)).rejects.toThrow(/conflict/i)
    expect(callbacks.openFresh).not.toHaveBeenCalled()
  })

  it('concurrent callers on the same target do not create duplicate canonical sessions', async () => {
    let lookupCalls = 0
    let createCalls = 0
    let resolveCreate!: (value: { runtimeId: string; storedId: string }) => void

    const createPromise = new Promise<{ runtimeId: string; storedId: string }>(resolve => {
      resolveCreate = resolve
    })

    const callbacks = makeCallbacks({
      lookup: vi.fn(async () => {
        lookupCalls += 1

        return null
      }),
      create: vi.fn(async () => {
        createCalls += 1

        return createPromise
      }),
    })

    const first = resolveCanonicalAgentChat(LOCAL_TARGET, callbacks)
    // Give the first caller's synchronous lookup+flight-registration a tick
    // to run before the second caller arrives, so the second genuinely
    // observes an in-flight creation rather than racing the map write.
    await Promise.resolve()
    await Promise.resolve()
    const second = resolveCanonicalAgentChat(LOCAL_TARGET, callbacks)

    resolveCreate({ runtimeId: 'runtime-shared', storedId: 'stored-shared' })

    const [firstResult, secondResult] = await Promise.all([first, second])

    expect(createCalls).toBe(1)
    expect(firstResult).toEqual({ registryId: 'stored-shared', openedId: 'stored-shared' })
    expect(secondResult).toEqual({ registryId: 'stored-shared', openedId: 'stored-shared' })
    // Both callers still get their own open — the flight created (never
    // adopted), so BOTH the winner and the joiner open via openFresh,
    // matching that callback's documented "freshly created" semantics.
    expect(callbacks.openFresh).toHaveBeenCalledWith(LOCAL_TARGET, 'stored-shared', true)
    expect(callbacks.openFresh).toHaveBeenCalledTimes(2)
    expect(callbacks.openExisting).not.toHaveBeenCalled()
  })

  it('a different target key creates independently — concurrency protection is per-target, not global', async () => {
    const otherTarget: CanonicalAgentChatTarget = { connectionId: null, profile: 'builder', targetProfile: 'builder' }
    const callbacksA = makeCallbacks()
    const callbacksB = makeCallbacks()

    const [resultA, resultB] = await Promise.all([
      resolveCanonicalAgentChat(LOCAL_TARGET, callbacksA),
      resolveCanonicalAgentChat(otherTarget, callbacksB),
    ])

    expect(resultA).toEqual({ registryId: 'stored-1', openedId: 'stored-1' })
    expect(resultB).toEqual({ registryId: 'stored-1', openedId: 'stored-1' })
    expect(callbacksA.create).toHaveBeenCalledTimes(1)
    expect(callbacksB.create).toHaveBeenCalledTimes(1)
  })

  it('reports canNavigate=false but still invokes the open callback when the caller reports opening is no longer current', async () => {
    const callbacks = makeCallbacks({
      lookup: vi.fn(async () => ({ id: 'existing-1', title: 'Bot Chat' })),
    })

    const result = await resolveCanonicalAgentChat(LOCAL_TARGET, callbacks, { openingStillCurrent: () => false })

    expect(result).toEqual({ registryId: 'existing-1', openedId: 'existing-1' })
    expect(callbacks.openExisting).toHaveBeenCalledWith(LOCAL_TARGET, 'existing-1', expect.objectContaining({ id: 'existing-1' }), false)
  })

  it('returns null and creates nothing when create reports no session was made', async () => {
    const callbacks = makeCallbacks({
      create: vi.fn(async () => ({ runtimeId: null, storedId: null })),
    })

    const result = await resolveCanonicalAgentChat(LOCAL_TARGET, callbacks)

    expect(result).toBeNull()
    expect(callbacks.titleSession).not.toHaveBeenCalled()
  })
})
