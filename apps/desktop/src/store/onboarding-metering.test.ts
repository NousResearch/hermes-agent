import { afterEach, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { blockContinuationSend } from '@/store/free-tier-continuation'
import { requestGatewayForAgent } from '@/store/gateway'
import { $onboardingAnswers } from '@/store/onboarding-answers'
import { setSessionOwnerHint } from '@/store/session'
import { $sessionStates } from '@/store/session-states'

vi.mock('@/store/gateway', async importOriginal => ({
  ...await importOriginal(), requestGatewayForAgent: vi.fn()
}))

const initial = $onboardingAnswers.get()
const prompt = { id: 'choose', role: 'assistant', parts: [{ type: 'text', text: 'What should we make?\n::onboarding{step="first" options="A garden planner|A journal"}' }] }

function seed() {
  setSessionOwnerHint('guide', { connectionId: 'remote-guide', profile: 'hermes-setup' })
  $sessionStates.set({ guide: { messages: [prompt] } as unknown as ClientSessionState })
  vi.mocked(requestGatewayForAgent).mockImplementation(async (_connection, _profile, method) =>
    (method === 'free_tier.choose_onboarding_task' ? { chosen: true } : { continuation_required: false }) as never)
}

afterEach(() => {
  $onboardingAnswers.set(initial)
  $sessionStates.set({})
  vi.clearAllMocks()
})

it('leaves layout alone and signals a typed or clicked first-task answer before submitting it', async () => {
  seed()
  $onboardingAnswers.set({ ...initial, committed: ['layout'] })
  expect(await blockContinuationSend('guide', { text: '[setup] layout: Basic', hidden: true })).toBe(false)
  expect(requestGatewayForAgent).toHaveBeenCalledTimes(1)
  expect(vi.mocked(requestGatewayForAgent).mock.calls[0][2]).toBe('free_tier.status')
  vi.clearAllMocks()
  expect(await blockContinuationSend('guide', { text: "Let's figure it out together" })).toBe(false)
  expect(requestGatewayForAgent).toHaveBeenCalledTimes(1)
  vi.clearAllMocks()
  expect(await blockContinuationSend('guide', { text: 'A garden planner' })).toBe(false)
  expect(vi.mocked(requestGatewayForAgent).mock.calls.map(call => call[2])).toEqual(['free_tier.choose_onboarding_task', 'free_tier.status'])
  expect(requestGatewayForAgent).toHaveBeenCalledWith('remote-guide', 'hermes-setup', 'free_tier.choose_onboarding_task', { session_id: 'guide' })
})

it('retains the draft when the task-choice marker cannot be confirmed', async () => {
  seed()
  vi.mocked(requestGatewayForAgent).mockRejectedValue(new Error('offline'))
  expect(await blockContinuationSend('guide', { text: 'A garden planner' })).toBe(true)
  expect(requestGatewayForAgent).toHaveBeenCalledTimes(1)
  expect($sessionStates.get().guide.messages).toEqual([prompt])
})
