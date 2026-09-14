/**
 * First-click home bounce (community report, Aug 2026): clicking a bot whose
 * canonical Bot Chat had been COMPRESSED landed on the Bots home instead of
 * the chat, and only a second click got through.
 *
 * Root cause: openRosterBot claimed the center with the durable registry id,
 * but the session-focus edge that the open itself fired reports the
 * compression-lineage TIP. releaseStaleOpenBotChat compared tip !== registry
 * id, called the claim stale, released it, and the home reasserted over the
 * freshly opened chat. The second click "worked" only because the tip was
 * already focused, so no new focus edge fired to sabotage it.
 *
 * The fix is to carry BOTH identities: openBotCanonicalChat returns
 * registryId + openedId, the claim stores both, and focus on either one keeps
 * it.
 *
 * Ported from tests/bot-open-focus-identity.test.mjs.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { RosterRow } from './types'

const { ackStoredSessionId, openBotCanonicalChat, prepareBotSource } = vi.hoisted(() => ({
  ackStoredSessionId: vi.fn(),
  openBotCanonicalChat: vi.fn(),
  prepareBotSource: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return { ...sdk, ackStoredSessionId }
})

vi.mock('./canonical-chat', () => ({
  CANONICAL_CHAT_TITLE: 'Bot Chat',
  ensureBotMetadata: vi.fn(async () => ({})),
  notifyBotOpenFailure: vi.fn(),
  openBotCanonicalChat,
  prepareBotSource,
  PROFILE_SESSION_LIST_LIMIT: 200
}))

const { host } = await import('@hermes/plugin-sdk')
const { $openBotChat } = await import('./bot-state')
const { openRosterBot } = await import('./roster-actions')
const { releaseStaleOpenBotChat } = await import('./roster-pane')

const bot = {
  canonical_session: { id: 'reg-1', last_active: 100 },
  connectionId: 'local',
  name: 'ops'
} as RosterRow

function withFocusApi(impl: () => null | string) {
  const original = host.focusOpenWorkspaceSession

  host.focusOpenWorkspaceSession = impl

  return () => {
    host.focusOpenWorkspaceSession = original
  }
}

beforeEach(() => {
  vi.clearAllMocks()
  $openBotChat.set(null)
  prepareBotSource.mockResolvedValue(undefined)
  openBotCanonicalChat.mockResolvedValue({ openedId: 'tip-9', registryId: 'reg-1' })
})

describe('an open claims both identities of the chat it opened', () => {
  it('records the registry row and the lineage tip it actually navigated to', async () => {
    openBotCanonicalChat.mockResolvedValue({ openedId: 'tip-9', registryId: 'reg-1' })

    await openRosterBot(bot)

    expect($openBotChat.get()).toMatchObject({ openedRegistryId: 'reg-1', openedSessionId: 'tip-9' })
  })
})

describe('unread acknowledgement follows a successful foreground transition', () => {
  it('acknowledges only after the canonical chat opens', async () => {
    openBotCanonicalChat.mockImplementation(async () => {
      expect(ackStoredSessionId).not.toHaveBeenCalled()

      return { openedId: 'tip-9', registryId: 'reg-1' }
    })

    await expect(openRosterBot(bot)).resolves.toBe(true)

    expect(ackStoredSessionId).toHaveBeenCalledWith('reg-1', 'ops')
  })

  it('acknowledges after an existing bot tab successfully fronts', async () => {
    const restore = withFocusApi(() => {
      expect(ackStoredSessionId).not.toHaveBeenCalled()

      return 'reg-1'
    })

    try {
      await expect(openRosterBot(bot)).resolves.toBe(true)

      expect(openBotCanonicalChat).not.toHaveBeenCalled()
      expect(ackStoredSessionId).toHaveBeenCalledWith('reg-1', 'ops')
    } finally {
      restore()
    }
  })

  it('preserves unread activity when source preparation fails', async () => {
    prepareBotSource.mockRejectedValue(new Error('gateway unavailable'))

    await expect(openRosterBot(bot)).resolves.toBe(false)

    expect(ackStoredSessionId).not.toHaveBeenCalled()
  })

  it('preserves unread activity when the canonical chat fails to open', async () => {
    openBotCanonicalChat.mockRejectedValue(new Error('open failed'))

    await expect(openRosterBot(bot)).resolves.toBe(false)

    expect(ackStoredSessionId).not.toHaveBeenCalled()
  })
})

describe('focus on either owned identity keeps the claim', () => {
  it('survives the tip focus edge the open itself fires', () => {
    $openBotChat.set({ key: 'k', openedRegistryId: 'reg-1', openedSessionId: 'tip-9' })

    releaseStaleOpenBotChat('tip-9')

    expect($openBotChat.get()).not.toBeNull()
  })

  it('survives focus reported as the durable registry id', () => {
    $openBotChat.set({ key: 'k', openedRegistryId: 'reg-1', openedSessionId: 'tip-9' })

    releaseStaleOpenBotChat('reg-1')

    expect($openBotChat.get()).not.toBeNull()
  })

  it('releases on a genuinely foreign session', () => {
    $openBotChat.set({ key: 'k', openedRegistryId: 'reg-1', openedSessionId: 'tip-9' })

    releaseStaleOpenBotChat('other-session')

    expect($openBotChat.get()).toBeNull()
  })

  it('releases a legacy draft claim as soon as any session takes focus', () => {
    // The newChat fallback has no registry id to compare against, so it only
    // yields once something is actually focused.
    $openBotChat.set({ key: 'k', openedRegistryId: '' })

    releaseStaleOpenBotChat('any')

    expect($openBotChat.get()).toBeNull()
  })
})
