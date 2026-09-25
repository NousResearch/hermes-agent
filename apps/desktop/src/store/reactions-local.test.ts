import { beforeEach, describe, expect, it, vi } from 'vitest'

import { handleDesktopBridgeEvent } from '@/app/session/hooks/use-message-stream/gateway-event/desktop-bridge'
import type { GatewayEventContext } from '@/app/session/hooks/use-message-stream/gateway-event/types'
import { wipeSessionListsForGatewaySwitch } from '@/store/gateway-switch'
import { $activeGatewayProfile } from '@/store/profile'
import {
  $agentReactions,
  $localReactions,
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
  applyReaction: (_list: unknown, emoji: null | string, author: string) =>
    emoji ? [{ author, emoji }] : []
}))

vi.mock('@/store/session', async () => {
  const { atom: nanoAtom } = await import('nanostores')
  const $messages = nanoAtom<unknown[]>([])

  return {
    $messages,
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
    setMessagingPlatformTotals: vi.fn(),
    setMessagingSessions: vi.fn(),
    setMessagingTruncated: vi.fn(),
    setSelectedStoredSessionId: vi.fn(),
    setSessionProfilesTruncated: vi.fn(),
    setSessionProfilesUsage: vi.fn(),
    setSessions: vi.fn(),
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
vi.mock('@/store/session-states', () => ({ clearAllSessionStates: vi.fn() }))
vi.mock('@/store/transcript-tail', () => ({ clearTranscriptTailPaging: vi.fn() }))
vi.mock('@/store/transcript-tail-cache', () => ({ clearTranscriptTails: vi.fn() }))

const AGENT_THUMBS_UP: MessageReaction[] = [{ at: 1, author: 'agent', emoji: '👍' }]

const reactionEvent = (isActiveEvent: boolean): GatewayEventContext =>
  ({
    event: { type: 'message.reaction' },
    isActiveEvent,
    payload: { row_id: 4242, reactions: AGENT_THUMBS_UP, role: 'assistant' }
  }) as unknown as GatewayEventContext

describe('live reaction overlay scope', () => {
  beforeEach(() => {
    clearLiveReactionOverlays()
  })

  it('a real message.reaction event records into the overlay, and merge prefers it over persisted', () => {
    // An optimistic bubble awaiting its durable row id is the stamping target.
    $messages.set([{ id: 'm1', role: 'assistant' }] as never)

    expect(handleDesktopBridgeEvent(reactionEvent(true))).toBe(true)
    expect($agentReactions.get()[4242]).toEqual(AGENT_THUMBS_UP)
    expect($messages.get()).toEqual([{ id: 'm1', reactions: AGENT_THUMBS_UP, role: 'assistant', rowId: 4242 }])
    expect(
      mergeReactions([{ at: 0, author: 'agent', emoji: '😴' }], undefined, $agentReactions.get()[4242])
    ).toEqual(AGENT_THUMBS_UP)
  })

  it('a profile swap clears the overlay so a foreign row id cannot repaint', () => {
    recordAgentReaction(4242, AGENT_THUMBS_UP)
    setLocalReaction('msg-1', '❤️')
    expect($agentReactions.get()[4242]).toBeDefined()

    $activeGatewayProfile.set('work')

    expect($agentReactions.get()).toEqual({})
    expect($localReactions.get()).toEqual({})
    // The durable reaction survives the wipe: merge falls back to persisted.
    expect(
      mergeReactions([{ at: 0, author: 'agent', emoji: '😴' }], undefined, $agentReactions.get()[4242])
    ).toEqual([{ at: 0, author: 'agent', emoji: '😴' }])
  })

  it('a same-profile set is not a swap and keeps live overlays', () => {
    recordAgentReaction(4242, AGENT_THUMBS_UP)

    $activeGatewayProfile.set($activeGatewayProfile.get())

    expect($agentReactions.get()[4242]).toEqual(AGENT_THUMBS_UP)
  })

  it('a connection switch wipes the overlays along with the rest of the outgoing transcript', () => {
    recordAgentReaction(4242, AGENT_THUMBS_UP)
    setLocalReaction('msg-1', '❤️')

    wipeSessionListsForGatewaySwitch()

    expect($agentReactions.get()).toEqual({})
    expect($localReactions.get()).toEqual({})
  })
})
