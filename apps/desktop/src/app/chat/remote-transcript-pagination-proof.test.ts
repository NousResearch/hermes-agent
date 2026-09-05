/**
 * Provenance: independently reviewed test-only artifact from worker commit
 * 1e9a784975b3134610cfe448f0872362f7c9a233 on
 * codex/hermes-remote-transcript-pagination-192035-20260904. The worker
 * commit's parent production pin is 2ddd96aff552b7aa8c48a5543b1687a41cf26c02;
 * only this file is carried into the existing PR #101435 owner branch. This
 * proves renderer pagination/scope behavior with a deterministic stand-in;
 * it does not claim live Electron, HTTP, or SQLite proof.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getLatestSessionMessages, type ProfileScope, setApiRequestConnection } from '@/hermes'
import { type ChatMessage, chatMessageText, toChatMessages } from '@/lib/chat-messages'
import { $transcriptTailBySessionId } from '@/store/transcript-tail'
import type { SessionMessage, SessionMessagesResponse } from '@/types/hermes'

import {
  _resetTranscriptBackfillForTests,
  backfillOlderTranscriptPage,
  mergeOlderTranscriptPage
} from './transcript-backfill'

const REMOTE_CONNECTION = 'remote-hermes'
const SESSION_ID = 'same-session-id'
const PAGE_LIMIT = 120
const INITIAL_MESSAGE_COUNT = 241
const PROFILES = ['profile-a', 'profile-b'] as const

type ProfileName = (typeof PROFILES)[number]

type RemoteApiRequest = {
  connectionId?: string
  path: string
  profile?: string
}

type RemoteApiCall = RemoteApiRequest

/**
 * A disposable remote state.db stand-in. It intentionally implements the
 * backend's `latest` paging contract: offset is measured back from the newest
 * row, while each page is returned in chronological order.
 */
function createDisposableRemoteSessions() {
  const rowsByProfile = new Map<ProfileName, SessionMessage[]>(
    PROFILES.map(profile => [
      profile,
      Array.from({ length: INITIAL_MESSAGE_COUNT }, (_, index) => createMessage(profile, index + 1))
    ])
  )

  return {
    append(profile: ProfileName) {
      const rows = rowsByProfile.get(profile)

      if (!rows) {
        throw new Error(`unknown disposable profile ${profile}`)
      }

      rows.push(createMessage(profile, rows.length + 1))
    },

    dispose() {
      rowsByProfile.clear()
    },

    page(profile: ProfileName, limit: number, offset: number): SessionMessagesResponse {
      const rows = rowsByProfile.get(profile)

      if (!rows) {
        throw new Error(`unknown disposable profile ${profile}`)
      }

      const end = Math.max(0, rows.length - offset)
      const start = Math.max(0, end - limit)
      const messages = rows.slice(start, end)

      return {
        messages,
        pagination: { limit, offset, order: 'latest', returned: messages.length },
        session_id: SESSION_ID
      }
    }
  }
}

function createMessage(profile: ProfileName, ordinal: number): SessionMessage {
  return {
    content: `${profile}/message-${ordinal}`,
    id: ordinal,
    role: ordinal % 2 === 0 ? 'assistant' : 'user',
    timestamp: 10_000 + ordinal
  }
}

function isProfileName(value: string | null): value is ProfileName {
  return value === 'profile-a' || value === 'profile-b'
}

function rowIds(messages: ChatMessage[]): number[] {
  return messages.map(message => message.rowId).filter((id): id is number => id !== undefined)
}

describe('remote transcript pagination with same-id profile twins', () => {
  let api: ReturnType<typeof vi.fn>
  let calls: RemoteApiCall[]
  let remote: ReturnType<typeof createDisposableRemoteSessions>

  beforeEach(() => {
    $transcriptTailBySessionId.set({})
    _resetTranscriptBackfillForTests()
    setApiRequestConnection(null)
    calls = []
    remote = createDisposableRemoteSessions()

    api = vi.fn(async (request: RemoteApiRequest): Promise<SessionMessagesResponse> => {
      calls.push(request)

      const url = new URL(request.path, 'http://desktop.test')
      const profile = url.searchParams.get('profile')
      const limit = Number(url.searchParams.get('limit'))
      const offset = Number(url.searchParams.get('offset'))

      if (url.pathname !== `/api/sessions/${SESSION_ID}/messages`) {
        throw new Error(`unexpected remote path ${request.path}`)
      }

      if (!isProfileName(profile)) {
        throw new Error('remote transcript request lost its profile')
      }

      if (request.connectionId !== REMOTE_CONNECTION || request.profile !== profile) {
        throw new Error(`remote transcript request crossed scope: ${JSON.stringify(request)}`)
      }

      if (url.searchParams.get('order') !== 'latest' || url.searchParams.get('include_compacted') !== 'true') {
        throw new Error(`remote transcript request lost pagination semantics: ${request.path}`)
      }

      return remote.page(profile, limit, offset)
    })

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api }
    })
  })

  afterEach(() => {
    remote.dispose()
    _resetTranscriptBackfillForTests()
    $transcriptTailBySessionId.set({})
    setApiRequestConnection(null)
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('reconstructs overlapping remote pages independently for same-id sessions in two profiles', async () => {
    const scopes: Record<ProfileName, ProfileScope> = {
      'profile-a': { connectionId: REMOTE_CONNECTION, profile: 'profile-a' },
      'profile-b': { connectionId: REMOTE_CONNECTION, profile: 'profile-b' }
    }

    const messagesByProfile = new Map<ProfileName, ChatMessage[]>()

    // Both profiles expose the same stored id, but each remote state.db has
    // different transcript content. Hydrate each through the real REST helper.
    for (const profile of PROFILES) {
      const tail = await getLatestSessionMessages(SESSION_ID, scopes[profile])

      expect(tail.messages).toHaveLength(PAGE_LIMIT)
      messagesByProfile.set(profile, toChatMessages(tail.messages))
    }

    // A row arriving after hydration shifts the newest-relative offset. The
    // next older page therefore overlaps the tail at durable row 122; the
    // following page must still recover the two rows skipped by that shift.
    for (const profile of PROFILES) {
      remote.append(profile)
    }

    for (const profile of PROFILES) {
      const applyOlderPage = (olderPage: ChatMessage[]) => {
        messagesByProfile.set(profile, mergeOlderTranscriptPage(messagesByProfile.get(profile) ?? [], olderPage))
      }

      await expect(
        backfillOlderTranscriptPage({
          applyOlderPage,
          isCurrent: () => true,
          profile: scopes[profile],
          storedSessionId: SESSION_ID
        })
      ).resolves.toBe(true)

      // The first older page is full, so the actual tail bookkeeping must
      // advance to 240 and make a second request instead of stopping or
      // replaying the overlapping page.
      await expect(
        backfillOlderTranscriptPage({
          applyOlderPage,
          isCurrent: () => true,
          profile: scopes[profile],
          storedSessionId: SESSION_ID
        })
      ).resolves.toBe(true)
    }

    for (const profile of PROFILES) {
      const messages = messagesByProfile.get(profile) ?? []
      const ids = rowIds(messages)

      expect(ids).toEqual(Array.from({ length: INITIAL_MESSAGE_COUNT }, (_, index) => index + 1))
      expect(new Set(ids).size).toBe(INITIAL_MESSAGE_COUNT)
      expect(messages.every(message => chatMessageText(message).startsWith(`${profile}/`))).toBe(true)
    }

    // Each profile made one tail read followed by the two older-page reads.
    // The request-level assertions above make missing/wrong scope fail at the
    // mocked remote boundary; these offsets prove overlap was traversed rather
    // than silently omitted or duplicated.
    for (const profile of PROFILES) {
      const profileCalls = calls.filter(call => call.profile === profile)

      const offsets = profileCalls.map(call =>
        Number(new URL(call.path, 'http://desktop.test').searchParams.get('offset'))
      )

      expect(profileCalls).toHaveLength(3)
      expect(offsets).toEqual([0, PAGE_LIMIT, 240])
      expect(profileCalls.every(call => call.connectionId === REMOTE_CONNECTION)).toBe(true)
      expect(
        profileCalls.every(call => new URL(call.path, 'http://desktop.test').searchParams.get('profile') === profile)
      ).toBe(true)
    }
  })
})
