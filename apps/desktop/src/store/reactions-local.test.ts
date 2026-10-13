import { beforeEach, describe, expect, it, vi } from 'vitest'

import { handleDesktopBridgeEvent } from '@/app/session/hooks/use-message-stream/gateway-event/desktop-bridge'
import type { GatewayEventContext } from '@/app/session/hooks/use-message-stream/gateway-event/types'
import { wipeSessionListsForGatewaySwitch } from '@/store/gateway-switch'
import { $activeGatewayProfile } from '@/store/profile'
import {
  $agentReactions,
  $localReactions,
  agentLiveReactions,
  clearLiveReactionOverlays,
  mergeReactions,
  recordAgentReaction,
  setLocalReaction
} from '@/store/reactions-local'
import { $messages } from '@/store/session'
import type { MessageReaction } from '@/types/hermes'

vi.mock('@/store/profile', async () => {
  const { atom: nanoAtom } = await import('nanostores')

  return {
    $activeGatewayProfile: nanoAtom('default'),
    invalidateProfileListFetches: vi.fn(),
    normalizeProfileKey: (value: string | null | undefined) => (value ?? '').trim() || 'default'
  }
})

vi.mock('@/store/reactions', () => ({
  QUICK_REACTIONS: ['❤️'],
  applyReaction: (_list: unknown, emoji: null | string, author: string) => (emoji ? [{ author, emoji }] : [])
}))

vi.mock('@/store/session', async () => {
  const { atom: nanoAtom } = await import('nanostores')
  const $messages = nanoAtom<unknown[]>([])

  return {
    $messages,
    $sessions: nanoAtom<unknown[]>([]),
    $unreadFinishedSessionIds: nanoAtom<string[]>([]),
    setActiveSessionId: vi.fn(),
    setCronSessions: vi.fn(),
    setCurrentBranch: vi.fn(),
    setCurrentCwdTransient: vi.fn(),
    setFreshDraftReady: vi.fn(),
    // Apply the updater for real: recordAgentReaction runs inside it.
    setMessages: (next: unknown) => {
      $messages.set(
        typeof next === 'function' ? (next as (m: unknown[]) => unknown[])($messages.get()) : (next as unknown[])
      )
    },
    setMessagingListServer: vi.fn(),
    setMessagingPlatformTotals: vi.fn(),
    setMessagingSessions: vi.fn(),
    setMessagingTruncated: vi.fn(),
    setSelectedStoredSessionId: vi.fn(),
    setSessionProfilesTruncated: vi.fn(),
    setSessionProfilesUsage: vi.fn(),
    setSessions: vi.fn(),
    setSessionsLoadError: vi.fn(),
    setSessionsLoading: vi.fn()
  }
})

vi.mock('@/app/right-sidebar/terminal/agent-terminal-stream', () => ({ writeAgentTerminalChunk: vi.fn() }))
vi.mock('@/app/right-sidebar/terminal/terminals', () => ({ closeAgentTerminalByProc: vi.fn() }))
vi.mock('@/store/pane-focus', () => ({ applyDesktopLayoutPreset: vi.fn(), revealDesktopPane: vi.fn() }))
vi.mock('@/store/tips', () => ({ $tipsEnabled: { get: () => false }, agentTipId: vi.fn(), showTip: vi.fn() }))
vi.mock('@/app/contrib/hooks/use-background-sync', () => ({ resetLiveRuntimeTracking: vi.fn() }))
vi.mock('@/hermes', () => ({ resetSidebarBatchCapability: vi.fn() }))
vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))
vi.mock('@/store/artifacts', () => ({ clearArtifactRegistry: vi.fn() }))
vi.mock('@/store/cron', () => ({ invalidateCronJobsRequests: vi.fn(), setCronJobs: vi.fn() }))
vi.mock('@/store/layout', () => ({ resetSessionsLimit: vi.fn() }))
vi.mock('@/store/live-sync', () => ({ resetLiveSync: vi.fn() }))
vi.mock('@/store/session-control', () => ({ clearAllSessionControl: vi.fn() }))
vi.mock('@/store/session-pin-sync', () => ({ resetSessionPinMirror: vi.fn() }))
vi.mock('@/store/session-states', async () => {
  const { atom: nanoAtom } = await import('nanostores')

  return {
    $sessionStates: nanoAtom({}),
    clearAllSessionStates: vi.fn()
  }
})
vi.mock('@/store/transcript-tail', () => ({ clearTranscriptTailPaging: vi.fn() }))
vi.mock('@/store/transcript-tail-cache', () => ({ clearTranscriptTails: vi.fn() }))

const AGENT_THUMBS_UP: MessageReaction[] = [{ at: 1, author: 'agent', emoji: '👍' }]
const SOURCE_A = 'conn:source-a::default'
const SOURCE_B = 'conn:source-b::default'

const reactionEvent = (overrides: Partial<GatewayEventContext> = {}): GatewayEventContext =>
  ({
    event: { connectionId: 'source-a', profile: 'default', type: 'message.reaction' },
    isActiveEvent: true,
    fromActiveSource: () => true,
    payload: { row_id: 4242, reactions: AGENT_THUMBS_UP, role: 'assistant' }
  }) as unknown as GatewayEventContext

describe('live reaction overlay scope', () => {
  beforeEach(() => {
    clearLiveReactionOverlays()
  })

  it('a real message.reaction event records into the overlay scoped to its source, and merge prefers it over persisted', () => {
    // An optimistic bubble awaiting its durable row id is the stamping target.
    $messages.set([{ id: 'm1', role: 'assistant' }] as never)

    expect(handleDesktopBridgeEvent(reactionEvent())).toBe(true)

    const overlay = $agentReactions.get()[4242]

    expect(overlay?.scope).toBe(SOURCE_A)
    expect(overlay?.reactions).toEqual(AGENT_THUMBS_UP)
    expect($messages.get()).toEqual([{ id: 'm1', reactions: AGENT_THUMBS_UP, role: 'assistant', rowId: 4242 }])
    expect(
      mergeReactions(
        [{ at: 0, author: 'agent', emoji: '😴' }],
        undefined,
        agentLiveReactions($agentReactions.get(), 4242, SOURCE_A)
      )
    ).toEqual(AGENT_THUMBS_UP)
  })

  it('a foreign source never sees the overlay: its persisted reaction at the same row id wins', () => {
    // Focusing a session tile from connection B does not change the ambient
    // profile, so an overlay recorded while source A was on screen used to
    // override B's persisted reaction at the coincidental same row id. The
    // read keys the displayed session's own source (connection + profile),
    // so B falls back to what its transcript carried.
    recordAgentReaction(4242, AGENT_THUMBS_UP, SOURCE_A)

    const persistedForB: MessageReaction[] = [{ at: 0, author: 'agent', emoji: '😴' }]

    expect(agentLiveReactions($agentReactions.get(), 4242, SOURCE_B)).toBeUndefined()
    expect(mergeReactions(persistedForB, undefined, agentLiveReactions($agentReactions.get(), 4242, SOURCE_B))).toEqual(
      persistedForB
    )
    // The owning source still sees its live overlay.
    expect(agentLiveReactions($agentReactions.get(), 4242, SOURCE_A)).toEqual(AGENT_THUMBS_UP)
  })

  it('a profile swap clears the overlay so a foreign row id cannot repaint', () => {
    recordAgentReaction(4242, AGENT_THUMBS_UP, 'default')
    setLocalReaction('msg-1', '❤️')
    expect($agentReactions.get()[4242]).toBeDefined()

    $activeGatewayProfile.set('work')

    expect($agentReactions.get()).toEqual({})
    expect($localReactions.get()).toEqual({})
    // The durable reaction survives the wipe: merge falls back to persisted.
    expect(
      mergeReactions(
        [{ at: 0, author: 'agent', emoji: '😴' }],
        undefined,
        agentLiveReactions($agentReactions.get(), 4242, 'work')
      )
    ).toEqual([{ at: 0, author: 'agent', emoji: '😴' }])
  })

  it('a same-profile set is not a swap and keeps live overlays', () => {
    recordAgentReaction(4242, AGENT_THUMBS_UP, 'default')

    $activeGatewayProfile.set($activeGatewayProfile.get())

    expect(agentLiveReactions($agentReactions.get(), 4242, 'default')).toEqual(AGENT_THUMBS_UP)
  })

  it('a connection switch wipes the overlays along with the rest of the outgoing transcript', () => {
    recordAgentReaction(4242, AGENT_THUMBS_UP, SOURCE_A)
    setLocalReaction('msg-1', '❤️')

    wipeSessionListsForGatewaySwitch()

    expect($agentReactions.get()).toEqual({})
    expect($localReactions.get()).toEqual({})
  })
})
