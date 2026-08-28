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

const { ackStoredSessionId, openBotCanonicalChat } = vi.hoisted(() => ({
  ackStoredSessionId: vi.fn(),
  openBotCanonicalChat: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return { ...sdk, ackStoredSessionId }
})

vi.mock('./canonical-chat', () => ({
  ensureBotMetadata: vi.fn(async () => ({})),
  notifyBotOpenFailure: vi.fn(),
  openBotCanonicalChat,
  prepareBotSource: vi.fn(async () => undefined),
  PROFILE_SESSION_LIST_LIMIT: 200
}))

const { $openBotChat } = await import('./bot-state')
const { openRosterBot } = await import('./roster-actions')
const { releaseStaleOpenBotChat } = await import('./roster-pane')

beforeEach(() => {
  vi.clearAllMocks()
  $openBotChat.set(null)
})

describe('an open claims both identities of the chat it opened', () => {
  it('records the registry row and the lineage tip it actually navigated to', async () => {
    openBotCanonicalChat.mockResolvedValue({ openedId: 'tip-9', registryId: 'reg-1' })

    await openRosterBot({ connectionId: 'local', name: 'ops' } as RosterRow)

    expect($openBotChat.get()).toMatchObject({ openedRegistryId: 'reg-1', openedSessionId: 'tip-9' })
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
