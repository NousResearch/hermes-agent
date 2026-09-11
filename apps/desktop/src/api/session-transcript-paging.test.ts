import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  backfillOlderTranscriptPage,
  mergeOlderTranscriptPage,
  transcriptBackfillAvailable
} from '@/app/chat/transcript-backfill'
import { toChatMessages } from '@/lib/chat-messages'
import { $transcriptTailBySessionId, transcriptTailState } from '@/store/transcript-tail'
import type { SessionMessagesResponse } from '@/types/hermes'

import { setApiRequestConnection, setApiRequestLocalMode, setApiRequestProfile } from './client'
import { getLatestSessionMessages, LATEST_SESSION_MESSAGES_LIMIT } from './sessions'

// Zero timestamps use the conversion-time clock for IDs; keep fixtures stable.
const row = (id: number) => ({ id, role: 'user' as const, content: `message ${id}`, timestamp: 1_000 + id })

const page = (messages: ReturnType<typeof row>[], offset = 0): SessionMessagesResponse => ({
  session_id: 'stored-session',
  profile: 'default',
  messages,
  pagination: { limit: LATEST_SESSION_MESSAGES_LIMIT, offset, order: 'latest', returned: messages.length }
})

describe('session transcript pagination ownership', () => {
  const api = vi.fn()

  beforeEach(() => {
    api.mockReset()
    $transcriptTailBySessionId.set({})
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })
    setApiRequestProfile('default')
    setApiRequestLocalMode(true)
  })

  afterEach(() => {
    setApiRequestLocalMode(false)
    setApiRequestConnection(null)
    setApiRequestProfile(null)
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it.each(['local', null])(
    'keeps older history reachable across scope spellings on connection %s',
    async connectionId => {
      setApiRequestConnection(connectionId)
      const owner = { connectionId: 'local', profile: 'default' }
      const older = Array.from({ length: 79 }, (_, index) => row(index + 1))
      const tail = Array.from({ length: LATEST_SESSION_MESSAGES_LIMIT }, (_, index) => row(older.length + index + 1))
      api.mockResolvedValue(page(tail))

      // Resume, post-turn hydration, and background refresh use different scope
      // spellings, but all three requests go to the same owning backend.
      for (const scope of [owner, 'default', undefined]) {
        await getLatestSessionMessages('stored-session', scope)
      }

      expect(Object.keys($transcriptTailBySessionId.get())).toHaveLength(1)
      expect(transcriptBackfillAvailable('stored-session', owner)).toBe(true)
      expect(transcriptBackfillAvailable('stored-session')).toBe(true)

      let messages = toChatMessages(tail)
      api.mockResolvedValueOnce(page(older, tail.length))
      expect(
        await backfillOlderTranscriptPage({
          storedSessionId: 'stored-session',
          profile: owner,
          isCurrent: () => true,
          applyOlderPage: earlier => {
            messages = mergeOlderTranscriptPage(messages, earlier)
          }
        })
      ).toBe(true)

      // Resolve the owner for lookup, but replay the exact ambient read route.
      expect(api).toHaveBeenLastCalledWith({
        ...(connectionId ? { connectionId } : {}),
        path: `/api/sessions/stored-session/messages?limit=${LATEST_SESSION_MESSAGES_LIMIT}&offset=${tail.length}&order=latest&include_compacted=true`
      })
      expect(messages.map(message => message.rowId)).toEqual([...older, ...tail].map(message => message.id))
      expect(transcriptBackfillAvailable('stored-session', owner)).toBe(false)
    }
  )

  it('coalesces spellings against a backend that predates the profile field', async () => {
    setApiRequestConnection(null)
    const owner = { connectionId: 'local', profile: 'default' }
    const tail = Array.from({ length: LATEST_SESSION_MESSAGES_LIMIT }, (_, index) => row(index + 1))
    const { profile: _omitted, ...legacyPage } = page(tail)
    api.mockResolvedValue(legacyPage)

    for (const scope of [owner, 'default', undefined]) {
      await getLatestSessionMessages('stored-session', scope)
    }

    // The legacy response cannot name its profile; the ambient profile stands in.
    expect(Object.keys($transcriptTailBySessionId.get())).toHaveLength(1)
    expect(transcriptBackfillAvailable('stored-session', owner)).toBe(true)
    expect(transcriptBackfillAvailable('stored-session')).toBe(true)
  })

  it('preserves legacy named-profile routing while the foreground is local', async () => {
    setApiRequestConnection(null)
    const profile = 'remote-alias'
    const owner = { connectionId: 'local', profile }
    const tail = Array.from({ length: LATEST_SESSION_MESSAGES_LIMIT }, (_, index) => row(index + 2))
    api.mockResolvedValueOnce({ ...page(tail), profile: 'work' })

    await getLatestSessionMessages('stored-session', profile)

    // An explicit local pin would bypass Electron's per-profile remote override.
    expect.soft(api).toHaveBeenLastCalledWith({
      profile,
      path: `/api/sessions/stored-session/messages?profile=${profile}&limit=${LATEST_SESSION_MESSAGES_LIMIT}&order=latest&include_compacted=true`
    })
    expect(transcriptBackfillAvailable('stored-session', owner)).toBe(true)

    const applyOlderPage = vi.fn()
    api.mockResolvedValueOnce({ ...page([row(1)], tail.length), profile: 'work' })
    expect(
      await backfillOlderTranscriptPage({
        storedSessionId: 'stored-session',
        profile: owner,
        isCurrent: () => true,
        applyOlderPage
      })
    ).toBe(true)

    expect(api).toHaveBeenLastCalledWith({
      profile,
      path: `/api/sessions/stored-session/messages?profile=${profile}&limit=${LATEST_SESSION_MESSAGES_LIMIT}&offset=${tail.length}&order=latest&include_compacted=true`
    })
    expect(applyOlderPage).toHaveBeenCalledWith(toChatMessages([row(1)]))
    expect(transcriptTailState('stored-session', owner)).toMatchObject({
      nextOffset: tail.length + 1,
      possiblyTruncated: false,
      profile: { profile }
    })
  })

  it.each(['work', null])(
    'isolates an ambient %s profile across explicit reads and gateway switches',
    async servingProfile => {
      setApiRequestConnection('source-a')
      const tail = page(Array.from({ length: LATEST_SESSION_MESSAGES_LIMIT }, (_, index) => row(index + 1)))
      let resolve!: (value: SessionMessagesResponse) => void
      api.mockReturnValueOnce(
        new Promise<SessionMessagesResponse>(done => {
          resolve = done
        })
      )
      const pending = getLatestSessionMessages('stored-session')
      api.mockResolvedValueOnce(page([row(1)]))
      await getLatestSessionMessages('stored-session', { connectionId: 'source-a', profile: 'default' })

      setApiRequestConnection('source-b')
      api.mockResolvedValueOnce(page([row(1)]))
      await getLatestSessionMessages('stored-session', { connectionId: 'source-b', profile: 'default' })
      resolve({ ...tail, profile: servingProfile })
      await pending

      const ownerA = { connectionId: 'source-a', profile: servingProfile || undefined }
      const ownerB = { connectionId: 'source-b', profile: 'default' }
      expect(transcriptTailState('stored-session', ownerA)).toMatchObject({
        possiblyTruncated: true,
        profile: { connectionId: 'source-a' }
      })
      expect(transcriptTailState('stored-session', { connectionId: 'source-a', profile: 'default' })).toMatchObject({
        possiblyTruncated: false
      })
      expect(transcriptTailState('stored-session', ownerB)).toMatchObject({ possiblyTruncated: false, profile: ownerB })
      expect(transcriptTailState('stored-session')).toBeUndefined()

      const defaultA = transcriptTailState('stored-session', { connectionId: 'source-a', profile: 'default' })
      const defaultB = transcriptTailState('stored-session', ownerB)
      const applyOlderPage = vi.fn()
      api.mockResolvedValueOnce({ ...page([row(0)], tail.messages.length), profile: servingProfile })
      expect(
        await backfillOlderTranscriptPage({
          storedSessionId: 'stored-session',
          profile: ownerA,
          isCurrent: () => true,
          applyOlderPage
        })
      ).toBe(true)

      // Serving-profile metadata selects the cache entry, not the request route.
      expect(api).toHaveBeenLastCalledWith({
        connectionId: 'source-a',
        path: `/api/sessions/stored-session/messages?limit=${LATEST_SESSION_MESSAGES_LIMIT}&offset=${tail.messages.length}&order=latest&include_compacted=true`
      })
      expect(applyOlderPage).toHaveBeenCalledWith(toChatMessages([row(0)]))
      expect(transcriptTailState('stored-session', ownerA)).toEqual({
        nextOffset: tail.messages.length + 1,
        possiblyTruncated: false,
        profile: { connectionId: 'source-a' }
      })
      expect(transcriptTailState('stored-session', { connectionId: 'source-a', profile: 'default' })).toBe(defaultA)
      expect(transcriptTailState('stored-session', ownerB)).toBe(defaultB)
    }
  )
})
