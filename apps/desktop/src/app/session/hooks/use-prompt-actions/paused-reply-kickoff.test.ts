import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $queuedPromptsBySession, getQueuedPrompts } from '@/store/composer-queue'
import { continueReplyPause, endReplyPause, markReplyPaused } from '@/store/paused-replies'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $sessions } from '@/store/session'
import { $sessionStates } from '@/store/session-states'

import { queueKickoffIfSessionBusy } from './queue-if-busy'
import { useSlashCommand } from './slash'
import type { GatewayRequest, SubmitTextOptions } from './utils'

// A reply paused for its automatic continuation (store/paused-replies.ts) holds what the user sends until
// the continuation has completed. A typed slash command is not held: most are controls (/yolo, /status,
// /new) and run at once. Some produce a prompt (a skill, /plan, /goal <text>): the backend answers with
// the prompt to send. That prompt is a new turn, so it waits in the chat's queue like a typed message,
// as it already does behind a running turn.

const STORED = 'stored-paused-kickoff'
const RUNTIME = 'runtime-paused-kickoff'

const skill = { type: 'skill', name: 'work', message: 'Load the work skill, then: fix it', display: '/work fix it' }
const plan = { type: 'send', message: 'Write a plan for: the migration' }
const goal = { type: 'send', notice: '⊙ Goal set', message: 'ship the release' }

function renderSlash(answer: unknown) {
  const submitPromptText = vi.fn<(rawText: string, options?: SubmitTextOptions) => Promise<boolean>>(async () => true)
  const appendSessionTextMessage = vi.fn()
  const requestGateway = vi.fn(async (method: string) => (method === 'slash.exec' ? answer : {}))

  const { result } = renderHook(() =>
    useSlashCommand({
      activeSessionIdRef: { current: RUNTIME },
      selectedStoredSessionIdRef: { current: STORED },
      busyRef: { current: false },
      copy: en.desktop,
      createBackendSessionForSend: async () => null,
      getRoutedStoredSessionId: () => STORED,
      getRuntimeIdForStoredSession: () => RUNTIME,
      requestGateway: requestGateway as GatewayRequest,
      resumeStoredSession: vi.fn(),
      updateSessionState: vi.fn((_sid, updater) => updater(createClientSessionState())),
      submitPromptText,
      appendSessionTextMessage,
      branchCurrentSession: async () => true,
      handleSkinCommand: () => '',
      handoffSession: async () => ({ ok: true }),
      openMemoryGraph: vi.fn(),
      refreshSessions: async () => undefined,
      startFreshSessionDraft: vi.fn()
    })
  )

  const type = (command: string) => act(async () => void (await result.current(command)))
  const output = () => appendSessionTextMessage.mock.calls.flat().filter(value => typeof value === 'string')

  return { output, submitPromptText, type }
}

const setBusy = (busy: boolean) =>
  $sessionStates.set({ [RUNTIME]: { ...createClientSessionState(), busy, storedSessionId: STORED } })

const queued = () => getQueuedPrompts(STORED).map(({ displayText, text }) => ({ displayText, text }))

describe('a slash command typed while the reply is paused for its continuation', () => {
  beforeEach(() => {
    // The queue store merges over what it saved (cross-window sync): start each case from an empty one.
    window.localStorage.clear()
    $sessions.set([])
    $connection.set(null)
    $activeGatewayProfile.set('default')
    $queuedPromptsBySession.set({})
    setBusy(false)
  })

  afterEach(() => {
    cleanup()
    endReplyPause(STORED)
    $queuedPromptsBySession.set({})
    $sessionStates.set({})
    vi.restoreAllMocks()
  })

  it.each([
    ['a skill', '/work fix it', skill, { displayText: '/work fix it', text: skill.message }],
    ['/plan', '/plan the migration', plan, { displayText: undefined, text: plan.message }],
    ['/goal <text>', '/goal ship the release', goal, { displayText: undefined, text: goal.message }]
  ])('holds the prompt of %s while the reply waits for its continuation', async (_label, command, answer, row) => {
    markReplyPaused(STORED)
    const slash = renderSlash(answer)

    await slash.type(command)

    // Not started ahead of the continuation: it waits in the chat's queue, and the command says so.
    expect(slash.submitPromptText).not.toHaveBeenCalled()
    expect(queued()).toEqual([row])
    expect(slash.output().some(text => text.includes('queued'))).toBe(true)
  })

  it('holds it while the continuation runs, as behind any running turn', async () => {
    markReplyPaused(STORED)
    continueReplyPause(STORED)
    setBusy(true)
    const slash = renderSlash(skill)

    await slash.type('/work fix it')

    expect(slash.submitPromptText).not.toHaveBeenCalled()
    expect(queued()).toEqual([{ displayText: '/work fix it', text: skill.message }])
  })

  it('holds it for a chat paused under its stored id whose restart has not published a runtime yet', () => {
    $sessionStates.set({})
    markReplyPaused(STORED)

    expect(
      queueKickoffIfSessionBusy({ sessionId: 'runtime-after-restart', storedSessionId: STORED, text: 'go on' })
    ).toBe('queued')
    expect(queued()).toEqual([{ displayText: undefined, text: 'go on' }])
  })

  it('still runs a command that produces no prompt at once', async () => {
    markReplyPaused(STORED)
    const slash = renderSlash({ output: 'tools: 12 enabled' })

    await slash.type('/tools')

    expect(slash.output().some(text => text.includes('tools: 12 enabled'))).toBe(true)
    expect(slash.submitPromptText).not.toHaveBeenCalled()
    expect(queued()).toEqual([])
  })

  it('sends the prompt at once when no reply is paused, and again once the pause has ended', async () => {
    const slash = renderSlash(skill)

    await slash.type('/work fix it')
    expect(slash.submitPromptText).toHaveBeenCalledTimes(1)

    markReplyPaused(STORED)
    endReplyPause(STORED)
    await slash.type('/work fix it')

    expect(slash.submitPromptText).toHaveBeenCalledTimes(2)
    expect(slash.submitPromptText.mock.calls[1]![0]).toBe(skill.message)
    expect(queued()).toEqual([])
  })
})
