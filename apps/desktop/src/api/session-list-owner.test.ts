import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { resolveResumeOwner } from '@/app/session/hooks/use-session-actions/resume-owner'
import type { HermesApiRequest } from '@/global'
import { sessionOwnerRouteFromRow } from '@/store/session-request-router'
import { deferred } from '@/test/deferred'
import { installRestBridge } from '@/test/rest-bridge'
import { makeSessionInfo } from '@/test/session-info'

import { setApiRequestConnection, setApiRequestLocalMode } from './client'
import { listAllProfileSessions, listSessions, listSidebarSessions } from './sessions'

// A local primary whose `work` profile Electron resolves to a legacy
// per-profile remote override: the untagged list request is answered by that
// remote, so its rows must keep the profile-door route that reaches it again.
const overrideRow = makeSessionInfo({ id: 'served-by-override', profile: 'work' })

const sidebar = {
  recentsProfile: 'all',
  recentsLimit: 50,
  recentsExclude: [],
  cronLimit: 50,
  messagingLimit: 100,
  messagingExclude: []
}

const sidebarPage = {
  recents: { sessions: [overrideRow] },
  cron: { sessions: [{ ...overrideRow, source: 'cron' }] },
  messaging: { sessions: [{ ...overrideRow, source: 'telegram' }] }
}

function resumeRoute(row: ReturnType<typeof makeSessionInfo>) {
  // The ambient resume connection of a legacy local primary is the profile door ('').
  return resolveResumeOwner(sessionOwnerRouteFromRow(row), row, '')
}

beforeEach(() => {
  setApiRequestLocalMode(true)
  setApiRequestConnection(null)
})

afterEach(() => {
  setApiRequestLocalMode(false)
  setApiRequestConnection(null)
})

describe('session lists on a local primary with a legacy per-profile remote override', () => {
  it.each([
    ['listSessions', () => listSessions()],
    ['listAllProfileSessions', () => listAllProfileSessions()]
  ])('%s keeps served rows on the profile door', async (_name, list) => {
    const api = installRestBridge(() => ({ sessions: [overrideRow] }))

    const [row] = (await list()).sessions

    expect((api.mock.calls[0][0] as HermesApiRequest).connectionId).toBeUndefined()
    expect(resumeRoute(row)).toEqual({ resolvedConnectionId: '', sessionOwner: 'work' })
  })

  it('keeps every sidebar slice on the profile door', async () => {
    installRestBridge(() => sidebarPage)

    const result = await listSidebarSessions(sidebar)

    for (const row of [...result.recents.sessions, ...result.cron.sessions, ...result.messaging.sessions]) {
      expect(resumeRoute(row)).toEqual({ resolvedConnectionId: '', sessionOwner: 'work' })
    }
  })

  it('does not hand a delayed page to the gateway selected while it was in flight', async () => {
    const pending = deferred<object>()
    installRestBridge(() => pending.promise)

    const request = listSessions()
    setApiRequestConnection('remote-b')
    pending.resolve({ sessions: [overrideRow] })

    expect((await request).sessions[0].connection_id).toBeUndefined()
  })

  it('keeps a delayed page fetched through an explicit local pin on this machine', async () => {
    setApiRequestConnection('local')
    const pending = deferred<object>()
    installRestBridge(() => pending.promise)

    const request = listSessions()
    setApiRequestConnection('remote-b')
    pending.resolve({ sessions: [overrideRow] })

    expect(sessionOwnerRouteFromRow((await request).sessions[0])).toEqual({ connectionId: 'local', profile: 'work' })
  })

  it('still stamps rows served by an active registry source', async () => {
    setApiRequestConnection('remote-b')
    installRestBridge(() => ({ sessions: [overrideRow] }))

    const [row] = (await listSessions()).sessions

    expect(sessionOwnerRouteFromRow(row)).toEqual({ connectionId: 'remote-b', profile: 'work' })
  })
})
