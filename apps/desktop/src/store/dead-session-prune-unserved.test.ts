import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesModule from '@/hermes'
import type { ProfileInfo, SessionInfo } from '@/types/hermes'

const getSessionMock = vi.fn<(id: string, profile?: null | string) => Promise<SessionInfo>>()

vi.mock('@/hermes', async importOriginal => {
  const actual = await importOriginal<typeof HermesModule>()

  return {
    ...actual,
    getSession: (id: string, profile?: null | string) => getSessionMock(id, profile)
  }
})

import { $queuedPromptsBySession } from '@/store/composer-queue'
import { $pinnedSessionIds } from '@/store/layout'
import { $activeGatewayProfile, $profiles } from '@/store/profile'
import { $sessions } from '@/store/session'

import { __runDeadSessionPrunePass, noteServedProfiles, resetDeadSessionPrune } from './dead-session-prune'

const row = (id: string): SessionInfo =>
  ({ id, message_count: 1, source: 'cli', started_at: 0, title: id }) as SessionInfo

const profile = (name: string): ProfileInfo =>
  ({
    has_env: false,
    is_default: name === 'default',
    model: null,
    name,
    path: name,
    provider: null,
    skill_count: 0
  }) as ProfileInfo

const notFound = () => new Error('404: {"detail":"Session not found"}')

// How a by-id read for a profile the gateway does not serve fails. On main the ticket mint
// collapses the refusal into a generic message; #142911 names it. Neither is a 404, so neither
// proves the chat dead, and the prune must not depend on either wording.
const REFUSALS = {
  'today (generic ticket error)': 'Invalid gateway ticket response',
  'with #142911 (named refusal)': "The gateway does not serve profile 'qa' (/home/u/.hermes/profiles/qa)"
}

const probesOf = (profileName: string) =>
  getSessionMock.mock.calls.filter(([, calledFor]) => calledFor === profileName).length

// The gateway is up: it answers 'default' and 'work' with a real "not found", and refuses 'qa'.
const qaRefused = (message: string) =>
  getSessionMock.mockImplementation(async (_id, profileName) => {
    if (profileName === 'qa') {
      throw new Error(message)
    }

    throw notFound()
  })

beforeAll(() => {
  ;(globalThis as { window?: unknown }).window ??= {}
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {}
})

beforeEach(() => {
  window.localStorage.clear()
  $sessions.set([row('unrelated')])
  $pinnedSessionIds.set([])
  $queuedPromptsBySession.set({})
  $profiles.set([profile('default'), profile('work'), profile('qa')])
  $activeGatewayProfile.set('default')
  getSessionMock.mockReset()
  resetDeadSessionPrune()
})

afterEach(() => {
  vi.useRealTimers()
  $sessions.set([])
  $pinnedSessionIds.set([])
  $queuedPromptsBySession.set({})
})

describe('dead-session prune and a profile the gateway does not serve', () => {
  describe('no served list known (standalone gateway, older backend, or not read yet)', () => {
    for (const [wording, message] of Object.entries(REFUSALS)) {
      it(`does not re-ask a refusing profile on the next sweep, and keeps its chat — ${wording}`, async () => {
        $pinnedSessionIds.set(['qa-chat'])
        qaRefused(message)

        await __runDeadSessionPrunePass()
        expect(probesOf('qa')).toBe(1)

        // Ten seconds later the sidebar refreshes and the sweep runs again. Nothing about 'qa'
        // has changed, so asking it again only adds another error to the log.
        getSessionMock.mockClear()
        await __runDeadSessionPrunePass()

        expect(probesOf('qa')).toBe(0)
        // A refusal is not a verdict: the pin stays.
        expect($pinnedSessionIds.get()).toEqual(['qa-chat'])
      })
    }

    it('re-asks a refusing profile only after a growing wait (30 s, then 1 min)', async () => {
      vi.useFakeTimers()
      $pinnedSessionIds.set(['qa-chat'])
      qaRefused(REFUSALS['today (generic ticket error)'])

      await __runDeadSessionPrunePass()
      getSessionMock.mockClear()

      vi.setSystemTime(Date.now() + 29_000)
      await __runDeadSessionPrunePass()
      expect(probesOf('qa')).toBe(0)

      vi.setSystemTime(Date.now() + 2_000)
      await __runDeadSessionPrunePass()
      expect(probesOf('qa')).toBe(1)

      // Refused again: the next wait is longer than the first.
      getSessionMock.mockClear()
      vi.setSystemTime(Date.now() + 31_000)
      await __runDeadSessionPrunePass()
      expect(probesOf('qa')).toBe(0)

      vi.setSystemTime(Date.now() + 30_000)
      await __runDeadSessionPrunePass()
      expect(probesOf('qa')).toBe(1)
      expect($pinnedSessionIds.get()).toEqual(['qa-chat'])
    })

    it('several chats refused in one sweep still start with a 30 s wait, not a longer one', async () => {
      // QA had three saved chats. One sweep checks up to four at once, so the profile refuses
      // several times in the same sweep: that is one refusal of the profile, not three.
      vi.useFakeTimers()
      $pinnedSessionIds.set(['qa-chat-1', 'qa-chat-2', 'qa-chat-3'])
      qaRefused(REFUSALS['today (generic ticket error)'])

      await __runDeadSessionPrunePass()
      getSessionMock.mockClear()

      vi.setSystemTime(Date.now() + 31_000)
      await __runDeadSessionPrunePass()

      expect(probesOf('qa')).toBeGreaterThan(0)
      expect($pinnedSessionIds.get()).toEqual(['qa-chat-1', 'qa-chat-2', 'qa-chat-3'])
    })

    it('still retries on the next sweep when the whole gateway is not answering', async () => {
      // Every probe fails, including the active profile's: that is the gateway, not one profile.
      // Existing behavior (dead-session-prune.test.ts "defers on non-404 probe failures") stays.
      $pinnedSessionIds.set(['flaky-pin'])
      getSessionMock.mockRejectedValue(new Error('gateway restarting'))

      await __runDeadSessionPrunePass()
      getSessionMock.mockClear()
      await __runDeadSessionPrunePass()

      expect(getSessionMock).toHaveBeenCalledWith('flaky-pin', undefined)
    })

    it('a gateway switch forgets the refusal, so the new gateway is asked', async () => {
      $pinnedSessionIds.set(['qa-chat'])
      qaRefused(REFUSALS['today (generic ticket error)'])

      await __runDeadSessionPrunePass()
      resetDeadSessionPrune()
      getSessionMock.mockClear()
      await __runDeadSessionPrunePass()

      expect(probesOf('qa')).toBe(1)
    })
  })

  describe('served list known (gateway status `gateway_shared_with`)', () => {
    it('never asks a profile the gateway does not serve, and keeps its chat', async () => {
      noteServedProfiles(['default', 'work'])
      $pinnedSessionIds.set(['qa-chat'])
      getSessionMock.mockRejectedValue(notFound())

      await __runDeadSessionPrunePass()
      await __runDeadSessionPrunePass()

      expect(probesOf('qa')).toBe(0)
      // 404 on every served profile is not enough: the unserved profile might own the chat.
      expect($pinnedSessionIds.get()).toEqual(['qa-chat'])
    })

    it('asks again as soon as the profile is back on the served list', async () => {
      noteServedProfiles(['default', 'work'])
      $pinnedSessionIds.set(['qa-chat'])
      getSessionMock.mockRejectedValue(notFound())

      await __runDeadSessionPrunePass()
      expect(probesOf('qa')).toBe(0)

      noteServedProfiles(['default', 'work', 'qa'])
      getSessionMock.mockImplementation(async (id, profileName) => {
        if (profileName === 'qa') {
          return row(id)
        }

        throw notFound()
      })
      await __runDeadSessionPrunePass()

      expect(probesOf('qa')).toBe(1)
      expect($pinnedSessionIds.get()).toEqual(['qa-chat'])
    })

    it('a profile coming back clears an earlier refusal without waiting out the backoff', async () => {
      $pinnedSessionIds.set(['qa-chat'])
      qaRefused(REFUSALS['today (generic ticket error)'])

      await __runDeadSessionPrunePass()
      noteServedProfiles(['default', 'work', 'qa'])
      getSessionMock.mockClear()
      await __runDeadSessionPrunePass()

      expect(probesOf('qa')).toBe(1)
    })

    it('still prunes a chat that is gone from every profile when every profile is served', async () => {
      noteServedProfiles(['default', 'work', 'qa'])
      $pinnedSessionIds.set(['gone-pin'])
      getSessionMock.mockRejectedValue(notFound())

      await __runDeadSessionPrunePass()

      expect($pinnedSessionIds.get()).toEqual([])
    })

    it('an unknown list (null) means every profile counts as served, as before', async () => {
      noteServedProfiles(['default', 'work'])
      noteServedProfiles(null)
      $pinnedSessionIds.set(['gone-pin'])
      getSessionMock.mockRejectedValue(notFound())

      await __runDeadSessionPrunePass()

      expect(probesOf('qa')).toBe(1)
      expect($pinnedSessionIds.get()).toEqual([])
    })
  })
})
