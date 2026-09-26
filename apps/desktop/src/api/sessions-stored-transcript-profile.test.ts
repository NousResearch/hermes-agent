import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/lib/gateway-rpc', () => ({ isMissingRestEndpoint: () => false }))
vi.mock('@/store/transcript-tail', () => ({ recordTranscriptTail: vi.fn() }))
vi.mock('@/store/connection-registry-state', () => ({ $connectionsRegistry: { get: () => ({ connections: [] }) } }))
vi.mock('./client', () => ({
  ambientOwnerConnectionId: vi.fn(() => 'local'),
  capabilityScoped: vi.fn(),
  connectionScoped: vi.fn(() => ({})),
  getApiRequestConnection: vi.fn(() => 'local'),
  getApiRequestProfile: vi.fn(() => null),
  hermesApi: vi.fn(),
  profileScoped: vi.fn(() => ({}))
}))

const client = await import('./client')

const { fetchStoredTranscriptAcrossBackends } = await import('./sessions')

const hermesApi = vi.mocked(client.hermesApi)

/** The real capabilityScoped turns a profile string into `{ profile }`; the
 *  restore probe's URL is built straight off that return. An absent scope must
 *  stay empty — that is the unknown-owner case the id-only sweep exists for. */
function scopeTo(scope: unknown) {
  if (typeof scope === 'string') {
    return { profile: scope }
  }

  if (scope && typeof scope === 'object') {
    return { ...(scope as Record<string, unknown>) }
  }

  return {}
}

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(client.getApiRequestConnection).mockReturnValue('local')
  vi.mocked(client.getApiRequestProfile).mockReturnValue(null)
  vi.mocked(client.capabilityScoped).mockImplementation(scope => scopeTo(scope))
})

/**
 * A restore that probes a stored transcript by ID ALONE resolves against the
 * ACTIVE gateway's home — the `default` profile — regardless of which profile
 * owns the session. For a multi-profile user every tile that is not in the
 * foreground profile then 404s on a row that is alive in its own state.db, and
 * the resume path reads that 404 as "session gone" and drops the tab.
 *
 * The contract: when the caller knows the owning profile, the ambient probe
 * must carry it, so the read lands on the backend that holds the row.
 */
describe('fetchStoredTranscriptAcrossBackends profile scoping', () => {
  it('carries the owning profile on the ambient probe', async () => {
    hermesApi.mockResolvedValue({ messages: [] } as never)

    await fetchStoredTranscriptAcrossBackends('sess-redapple', 'redapple')

    expect(hermesApi).toHaveBeenCalledTimes(1)
    expect(hermesApi.mock.calls[0][0].path).toContain('profile=redapple')
  })

  it('still probes id-only when the owner is unknown', async () => {
    // The unknown-owner case must keep working: probing every registered
    // backend by id is the recovery this function exists for. Scoping it to a
    // guessed profile would be worse than not scoping it at all.
    hermesApi.mockResolvedValue({ messages: [] } as never)

    await fetchStoredTranscriptAcrossBackends('sess-orphan')

    expect(hermesApi).toHaveBeenCalledTimes(1)
    expect(hermesApi.mock.calls[0][0].path).not.toContain('profile=')
  })

  it('treats a null owner as unknown rather than defaulting to a profile', async () => {
    hermesApi.mockResolvedValue({ messages: [] } as never)

    await fetchStoredTranscriptAcrossBackends('sess-orphan', null)

    expect(hermesApi).toHaveBeenCalledTimes(1)
    expect(hermesApi.mock.calls[0][0].path).not.toContain('profile=')
  })

  it('falls through to the connection sweep when the scoped probe misses', async () => {
    // A miss on the owning profile must not swallow the cross-backend sweep:
    // the probe is retried, and with no registered remote connection the sweep
    // legitimately resolves to null rather than throwing.
    hermesApi.mockRejectedValue(new Error('404'))

    const stored = await fetchStoredTranscriptAcrossBackends('sess-redapple', 'redapple')

    expect(hermesApi).toHaveBeenCalledTimes(1)
    expect(stored).toBeNull()
  })
})
