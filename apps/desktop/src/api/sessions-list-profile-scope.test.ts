/**
 * A Bot Mode list read names the selected bot's profile in its selector
 * (`recents_profile` / `profile`). The request's own profile must name the
 * same one: Electron adds the ambient `?profile=` beside the selector, and
 * the gateway's grant check refuses a disagreeing pair with
 * 403 profile_scope_mismatch (#126732 review).
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  listAllProfileSessions,
  listSidebarSessions,
  resetSidebarBatchCapability,
  setApiRequestProfile
} from '@/hermes'

describe('single-profile list reads carry that profile as the request scope', () => {
  let api: ReturnType<typeof vi.fn>

  beforeEach(() => {
    resetSidebarBatchCapability()
    api = vi.fn().mockResolvedValue({
      cron: { sessions: [] },
      limit: 0,
      messaging: { sessions: [] },
      offset: 0,
      recents: { sessions: [] },
      sessions: [],
      total: 0
    })
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })
    setApiRequestProfile('default')
  })

  afterEach(() => {
    setApiRequestProfile(null)
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('sidebar batch for a bot profile is requested as that profile', async () => {
    await listSidebarSessions({
      cronLimit: 1,
      messagingExclude: [],
      messagingLimit: 1,
      recentsExclude: [],
      recentsLimit: 1,
      recentsProfile: 'mybot'
    })

    expect(api.mock.calls[0][0]).toMatchObject({ profile: 'mybot' })
  })

  it('per-slice list for a bot profile is requested as that profile', async () => {
    await listAllProfileSessions(5, 1, 'exclude', 'recent', 'mybot')

    expect(api.mock.calls[0][0]).toMatchObject({ profile: 'mybot' })
  })

  it('the all-profiles fan-out keeps the ambient scope', async () => {
    await listAllProfileSessions(5, 1, 'exclude', 'recent', 'all')

    expect(api.mock.calls[0][0]).toMatchObject({ profile: 'default' })
  })
})
