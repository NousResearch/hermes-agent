import { describe, expect, it } from 'vitest'

import {
  $inbox,
  beginInboxRequest,
  clearInbox,
  commitInboxRequest,
  filterInboxItems,
  type InboxEntry,
  isInboxRequestCurrent,
  parseInboxSnapshot,
  refreshInbox,
  selectInboxBadge,
  selectInboxNeedsCount
} from './inbox'

function backendSnapshot(overrides: Record<string, unknown> = {}) {
  return {
    coverage: {
      approval_scope: 'live gateway approval queue',
      clarify_scope: 'live open sessions only',
      connection_scope: 'active connection and profile only',
      errors: [],
      partial: false,
      profile: 'default',
      scanned_sessions: 2
    },
    items: [
      {
        cwd: 'C:/w/x',
        lanes: ['needs_you', 'running'],
        goal: { status: 'active', title: 'Ship the inbox', turns_used: 1, max_turns: 4 },
        loop: null,
        heartbeat: null,
        pending_approval: { command_redacted: true, count: 1, description: 'run deploy' },
        pending_clarify: null,
        session_key: 'sess-1',
        source: 'cli',
        title: 'Inbox session'
      },
      {
        cwd: '',
        lanes: ['scheduled'],
        goal: null,
        loop: null,
        heartbeat: { prompt: 'Poll', status: 'active', fire_count: 3 },
        pending_approval: null,
        pending_clarify: null,
        session_key: 'sess-2',
        source: 'cli',
        title: 'Heartbeat session'
      }
    ],
    counts: { needs_you: 1, running: 1, waiting: 0, scheduled: 1, total: 2 },
    badge: 'amber',
    ...overrides
  }
}

function fakeRequest(result: unknown, error?: unknown) {
  return async <R>(_method: string): Promise<R> => {
    if (error) {throw error}

    return result as R
  }
}

describe('parseInboxSnapshot', () => {
  it('accepts a valid backend payload into a typed snapshot', () => {
    const parsed = parseInboxSnapshot({ inbox: backendSnapshot() })
    expect(parsed).not.toBeNull()

    if (!parsed) {return}
    expect(parsed.badge).toBe('amber')
    expect(parsed.counts.needs_you).toBe(1)
    expect(parsed.items).toHaveLength(2)
    expect(parsed.items[0].lanes).toEqual(['needs_you', 'running'])
    expect(parsed.items[0].pending_approval?.count).toBe(1)
    expect(parsed.coverage.profile).toBe('default')
    expect(parsed.coverage.partial).toBe(false)
  })

  it('parses metadata-only pending_clarify (count without question)', () => {
    const snapshot = backendSnapshot()
    const item = snapshot.items[0] as Record<string, unknown>

    item.pending_clarify = { count: 1 }
    const parsed = parseInboxSnapshot({ inbox: snapshot })

    expect(parsed).not.toBeNull()
    expect(parsed!.items[0].pending_clarify).toEqual({ count: 1 })
  })

  it('rejects a payload with pending_clarify missing count', () => {
    const snapshot = backendSnapshot()
    const item = snapshot.items[0] as Record<string, unknown>

    item.pending_clarify = { question: 'What?' }
    expect(parseInboxSnapshot({ inbox: snapshot })).toBeNull()
  })

  it('rejects a whole-wire payload that lacks the inbox envelope', () => {
    expect(parseInboxSnapshot({ items: [] })).toBeNull()
    expect(parseInboxSnapshot(null)).toBeNull()
    expect(parseInboxSnapshot(undefined)).toBeNull()
  })

  it('rejects a payload with an unknown lane', () => {
    const bad = backendSnapshot()

    ;(bad.items as Array<Record<string, unknown>>)[0].lanes = ['fabricated']
    expect(parseInboxSnapshot({ inbox: bad })).toBeNull()
  })

  it('rejects a payload with a malformed coverage or counts block', () => {
    const noCounts = backendSnapshot()

    ;(noCounts as Record<string, unknown>).counts = null
    expect(parseInboxSnapshot({ inbox: noCounts })).toBeNull()

    const badCoverage = backendSnapshot()

    ;(badCoverage as Record<string, unknown>).coverage = { partial: true }
    expect(parseInboxSnapshot({ inbox: badCoverage })).toBeNull()
  })

  it('rejects a non-object item entry', () => {
    const bad = backendSnapshot()

    ;(bad as Record<string, unknown>).items = ['not-an-object']
    expect(parseInboxSnapshot({ inbox: bad })).toBeNull()
  })
})

describe('inbox request tokens (A-B-A protection)', () => {
  it('begin/commit only accepts the current generation and consumes it', () => {
    const token = beginInboxRequest('conn\\u0000default')
    expect(isInboxRequestCurrent(token)).toBe(true)
    expect(commitInboxRequest(token, parseInboxSnapshot({ inbox: backendSnapshot() })!)).toBe(true)
    // The token was consumed: a duplicate completion cannot publish again.
    expect(commitInboxRequest(token, parseInboxSnapshot({ inbox: backendSnapshot() })!)).toBe(false)
  })

  it('a superseded (stale) request can never commit', () => {
    const stale = beginInboxRequest('conn\\u0000default')
    beginInboxRequest('conn\\u0000default') // newer request in same scope
    expect(isInboxRequestCurrent(stale)).toBe(false)
    expect(commitInboxRequest(stale, parseInboxSnapshot({ inbox: backendSnapshot() })!)).toBe(false)
  })
})

describe('refreshInbox', () => {
  it('publishes a supported snapshot from the gateway', async () => {
    clearInbox()
    const request = fakeRequest({ inbox: backendSnapshot() })
    const response = await refreshInbox('default', request)
    expect(response.published).toBe(true)
    const entry = $inbox.get()
    expect(entry.capability).toBe('supported')
    expect(entry.error).toBeNull()
    expect(entry.snapshot?.counts.needs_you).toBe(1)
  })

  it('marks unsupported (never all-clear) on method-not-found', async () => {
    clearInbox()
    const request = fakeRequest(undefined, { code: -32601, message: 'Method not found' })
    await refreshInbox('default', request)
    expect($inbox.get().capability).toBe('unsupported')
    expect($inbox.get().snapshot).toBeNull()
  })

  it('records a read error and never fabricates a snapshot', async () => {
    clearInbox()
    const request = fakeRequest(undefined, new Error('gateway reset'))
    await refreshInbox('default', request)
    const entry = $inbox.get()
    expect(entry.capability).toBe('unknown')
    expect(entry.error).toBeTruthy()
    expect(entry.snapshot).toBeNull()
  })

  it('ignores a late response after the store is cleared (gateway switch)', async () => {
    clearInbox()
    let resolveFn: (value: { inbox: unknown }) => void = () => undefined

    const pending = new Promise<{ inbox: unknown }>(resolve => {
      resolveFn = resolve
    })

    const request = (async () => pending) as never
    const inflight = refreshInbox('default', request)
    clearInbox() // simulate the gateway-switch seam mid-flight
    resolveFn({ inbox: backendSnapshot() })
    const response = await inflight
    expect(response.published).toBe(false)
    expect($inbox.get().capability).toBe('unknown')
    expect($inbox.get().snapshot).toBeNull()
  })

  it('never starts an overlapping refresh while one is in flight', async () => {
    clearInbox()
    let calls = 0

    const request = (async () => {
      calls += 1

      return { inbox: backendSnapshot() }
    }) as never

    const first = refreshInbox('default', request)
    const second = refreshInbox('default', request)
    await Promise.all([first, second])
    expect(calls).toBeLessThan(2)
  })
})

describe('selectors', () => {
  it('selects needs-you count and badge from the published snapshot', () => {
    clearInbox()
    $inbox.set({
      capability: 'supported',
      error: null,
      loading: false,
      snapshot: parseInboxSnapshot({ inbox: backendSnapshot() })!
    })
    expect(selectInboxNeedsCount($inbox.get())).toBe(1)
    expect(selectInboxBadge($inbox.get())).toBe('amber')
  })

  it('surfaces red when the backend reports a coverage error -- never all-clear', () => {
    const withErrors: InboxEntry = {
      capability: 'supported',
      error: null,
      loading: false,
      snapshot: parseInboxSnapshot({
        inbox: backendSnapshot({ coverage: { ...backendSnapshot().coverage, errors: ['sess-x: snapshot read failed'] }, items: [], badge: 'red' })
      })!
    }

    expect(selectInboxBadge(withErrors)).toBe('red')
  })

  it('returns none/zero for the empty cleared store', () => {
    clearInbox()
    expect(selectInboxNeedsCount($inbox.get())).toBe(0)
    expect(selectInboxBadge($inbox.get())).toBe('none')
  })

  it('filters inbox items by title / key / cwd, ignoring case', () => {
    const parsed = parseInboxSnapshot({ inbox: backendSnapshot() })!
    expect(filterInboxItems(parsed.items, '')).toHaveLength(2)
    expect(filterInboxItems(parsed.items, 'inbox session')).toHaveLength(1)
    expect(filterInboxItems(parsed.items, 'HEARTBEAT')).toHaveLength(1)
    expect(filterInboxItems(parsed.items, 'sess-2')).toHaveLength(1)
    expect(filterInboxItems(parsed.items, 'C:/w/x')).toHaveLength(1)
    expect(filterInboxItems(parsed.items, 'no-match')).toHaveLength(0)
  })
})

describe('regression: stale data after gateway/profile switch', () => {
  it('allows a new scope immediately while an old request is pending', async () => {
    clearInbox()
    let resolveOld!: (value: unknown) => void
    const old = refreshInbox('old', () => new Promise(resolve => { resolveOld = resolve }))
    expect($inbox.get().loading).toBe(true)
    clearInbox()
    const fresh = await refreshInbox('new', fakeRequest({ inbox: backendSnapshot() }))
    expect(fresh.published).toBe(true)
    resolveOld({ inbox: backendSnapshot() })
    expect((await old).published).toBe(false)
    expect($inbox.get().loading).toBe(false)
  })

  it('marks truncated coverage as incomplete even without read errors', () => {
    const snapshot = parseInboxSnapshot({ inbox: backendSnapshot() })!
    snapshot.coverage.partial = true
    expect(selectInboxBadge({ capability: 'supported', error: null, loading: false, snapshot })).toBe('red')
  })

  it('clearInbox wipes snapshot so stale rows are never presented as current', async () => {
    clearInbox()
    const request = fakeRequest({ inbox: backendSnapshot() })
    await refreshInbox('default', request)

    const before = $inbox.get()
    expect(before.snapshot).not.toBeNull()
    expect(before.snapshot!.items).toHaveLength(2)

    // Simulate gateway going null
    clearInbox()

    const after = $inbox.get()
    expect(after.snapshot).toBeNull()
    expect(after.capability).toBe('unknown')
    expect(after.error).toBeNull()
  })

  it('late response after clearInbox is rejected (A-B-A)', async () => {
    clearInbox()

    let resolveFn: (value: { inbox: unknown }) => void = () => undefined
    const pending = new Promise<{ inbox: unknown }>(resolve => { resolveFn = resolve })
    const request = (async () => pending) as never

    const inflight = refreshInbox('default', request)

    // Profile switch happens mid-flight
    clearInbox()

    // Late response arrives from old backend
    resolveFn({ inbox: backendSnapshot() })
    const response = await inflight

    expect(response.published).toBe(false)
    expect($inbox.get().snapshot).toBeNull()
  })

  it('profile switch clears old items — new profile gets fresh state', async () => {
    clearInbox()

    // Populate from profile A
    const reqA = fakeRequest({
      inbox: backendSnapshot({
        coverage: { ...backendSnapshot().coverage, profile: 'profileA' },
        items: [{ ...backendSnapshot().items[0], session_key: 'sess-a-from-A', title: 'Profile A only' }]
      })
    })

    await refreshInbox('profileA', reqA)

    expect($inbox.get().snapshot!.items[0].session_key).toBe('sess-a-from-A')

    // Profile switch
    clearInbox()
    expect($inbox.get().snapshot).toBeNull()

    // New profile populates — old items are gone
    const reqB = fakeRequest({
      inbox: backendSnapshot({
        coverage: { ...backendSnapshot().coverage, profile: 'profileB' },
        items: [{ ...backendSnapshot().items[0], session_key: 'sess-b-from-B', title: 'Profile B only' }]
      })
    })

    await refreshInbox('profileB', reqB)

    const entry = $inbox.get()
    expect(entry.snapshot!.coverage.profile).toBe('profileB')
    expect(entry.snapshot!.items).toHaveLength(1)
    expect(entry.snapshot!.items[0].session_key).toBe('sess-b-from-B')
  })
})
