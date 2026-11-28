import { beforeEach, describe, expect, it, vi } from 'vitest'

// A session page belongs to the source that SERVED it. The stamping helper used
// to read the ambient request connection when the response landed, so a source
// switch that finished while a page was in flight re-tagged the outgoing
// source's rows with the incoming one: the sidebar rendered this machine's
// chats under the other gateway's profile group, and a later resume followed
// that wrong owner. These tests pin the stamp to the request-time scope.

vi.mock('./client', () => ({
  ambientOwnerConnectionId: vi.fn(() => null),
  capabilityScoped: vi.fn(() => ({})),
  connectionScoped: vi.fn(() => ({})),
  getApiRequestConnection: vi.fn(() => null),
  getApiRequestProfile: vi.fn(() => null),
  hermesApi: vi.fn(),
  profileScoped: vi.fn(() => ({})),
  sessionReadOwnerPin: vi.fn(() => ({}))
}))

vi.mock('@/lib/gateway-rpc', () => ({ isMissingRestEndpoint: () => false }))
vi.mock('@/lib/legacy-session-owner-backfill', () => ({ maybeBackfillLegacySessionOwners: vi.fn() }))
vi.mock('@/store/transcript-tail', () => ({ pageHonorsLatestOrder: () => true, recordTranscriptTail: vi.fn() }))

const client = await import('./client')

const { listAllProfileSessions, listSidebarSessions, listSessions } = await import('./sessions')

const hermesApi = vi.mocked(client.hermesApi)
const requestConnection = vi.mocked(client.getApiRequestConnection)

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(settle => {
    resolve = settle
  })

  return { promise, resolve }
}

function row(id: string): Record<string, unknown> {
  return { id, title: id, profile: 'default', last_active: 1, message_count: 1, source: 'desktop' }
}

beforeEach(() => {
  vi.clearAllMocks()
  requestConnection.mockReturnValue(null)
})

describe('session pages keep the source that served them', () => {
  it('leaves a local page unowned when the window moves to another source mid-flight', async () => {
    const page = deferred<Record<string, unknown>>()

    hermesApi.mockReturnValue(page.promise as never)

    const inflight = listSessions(10)

    // The pick lands while the page is still on the wire.
    requestConnection.mockReturnValue('nailde')
    page.resolve({ sessions: [row('local-1')], total: 1 })

    const result = await inflight

    expect(result.sessions[0]?.connection_id).toBeUndefined()
  })

  it('keeps a remote page owned by that source after the window returns to the local pool', async () => {
    const page = deferred<Record<string, unknown>>()

    hermesApi.mockReturnValue(page.promise as never)
    requestConnection.mockReturnValue('nailde')

    const inflight = listSessions(10)

    requestConnection.mockReturnValue(null)
    page.resolve({ sessions: [row('remote-1')], total: 1 })

    const result = await inflight

    expect(result.sessions[0]?.connection_id).toBe('nailde')
  })

  it('never overrides an owner the serving backend already stamped', async () => {
    const page = deferred<Record<string, unknown>>()

    hermesApi.mockReturnValue(page.promise as never)

    const inflight = listSessions(10)

    requestConnection.mockReturnValue('nailde')
    page.resolve({ sessions: [{ ...row('tagged'), connection_id: 'homelab' }], total: 1 })

    const result = await inflight

    expect(result.sessions[0]?.connection_id).toBe('homelab')
  })

  it('keeps the serving source on the unified all-profiles page', async () => {
    const page = deferred<Record<string, unknown>>()

    hermesApi.mockReturnValue(page.promise as never)
    requestConnection.mockReturnValue('nailde')

    const inflight = listAllProfileSessions(10)

    requestConnection.mockReturnValue(null)
    page.resolve({ sessions: [row('remote-2')], total: 1 })

    const result = await inflight

    expect(result.sessions[0]?.connection_id).toBe('nailde')
  })

  it('stamps every batched sidebar slice with the source that served the batch', async () => {
    const page = deferred<Record<string, unknown>>()

    hermesApi.mockReturnValue(page.promise as never)
    requestConnection.mockReturnValue('nailde')

    const inflight = listSidebarSessions({
      cronLimit: 5,
      messagingExclude: [],
      messagingLimit: 5,
      recentsExclude: [],
      recentsLimit: 10,
      recentsProfile: 'default'
    })

    requestConnection.mockReturnValue(null)
    page.resolve({
      cron: { sessions: [row('cron-1')] },
      messaging: { sessions: [row('msg-1')] },
      recents: { sessions: [row('remote-3')] }
    })

    const result = await inflight

    expect(result.recents.sessions[0]?.connection_id).toBe('nailde')
    expect(result.cron.sessions[0]?.connection_id).toBe('nailde')
    expect(result.messaging.sessions[0]?.connection_id).toBe('nailde')
  })
})
