import { afterEach, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { $setupSession } from '@/components/onboarding-chat/setup-profile'
import { assistantTextPart } from '@/lib/chat-messages'
import { $activeGatewayProfile } from '@/store/profile'
import { $activeSessionId, $messages, $selectedStoredSessionId } from '@/store/session'
import { $sessionStates } from '@/store/session-states'

import { adoptGuideSession } from './onboarding-kickoff'

const messages = [{ id: 'greeting', role: 'assistant' as const, parts: [assistantTextPart('Welcome back')] }]
const setupProfile = 'setup-provisioned'

function publishGuide(storedSessionId: string, visible = true) {
  const state: ClientSessionState = {
    storedSessionId,
    messages,
    branch: '',
    cwd: '',
    model: '',
    provider: '',
    reasoningEffort: '',
    serviceTier: '',
    fast: false,
    yolo: false,
    personality: '',
    busy: false,
    awaitingResponse: false,
    streamId: null,
    sawAssistantPayload: false,
    adoptedRunningTurn: false,
    pendingBranchGroup: null,
    interrupted: false,
    interimBoundaryPending: false,
    needsInput: false,
    runtimeStartedAt: Date.now(),
    turnStartedAt: null,
    turnLive: false,
    usage: null
  }

  $sessionStates.set({ 'runtime-guide': state })
  $activeSessionId.set('runtime-guide')
  $selectedStoredSessionId.set(storedSessionId)
  $activeGatewayProfile.set(setupProfile)
  $messages.set(visible ? messages : [])
}

afterEach(() => {
  $sessionStates.set({})
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  $activeGatewayProfile.set('default')
  $messages.set([])
  $setupSession.set(null)
})

it('adopts the resumed runtime only after the correct saved transcript is visible', async () => {
  const request = vi.fn(async () => {
    throw new Error('Unexpected configuration request')
  })

  await adoptGuideSession(
    setupProfile,
    { id: 'guide', resolved_id: 'guide-tip' },
    false,
    async () => {
      publishGuide('guide-tip')
    },
    request
  )
  expect($setupSession.get()).toMatchObject({ storedId: 'guide', runtimeId: 'runtime-guide', profile: setupProfile })
  expect(request).not.toHaveBeenCalled()
  expect($messages.get()).toEqual(messages)
})

it.each(['empty', 'wrong-session', 'wrong-profile', 'no-runtime'])(
  'rejects a settled resume with %s instead of releasing startup',
  async failure => {
    const request = vi.fn(async () => {
      throw new Error('Unexpected configuration request')
    })

    await expect(
      adoptGuideSession(
        setupProfile,
        { id: 'guide' },
        false,
        async () => {
          publishGuide(failure === 'wrong-session' ? 'other' : 'guide', failure !== 'empty')

          if (failure === 'wrong-profile') {
            $activeGatewayProfile.set('default')
          }

          if (failure === 'no-runtime') {
            $activeSessionId.set(null)
          }
        },
        request
      )
    ).rejects.toThrow('could not be loaded')
    expect($setupSession.get()).toBeNull()
    expect(request).not.toHaveBeenCalled()
  }
)
