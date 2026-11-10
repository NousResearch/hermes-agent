import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { DEMO_LAYOUT_ID, registerLayoutPresets } from '@/app/contrib/layout-presets'
import { endChatOnboardingSolo } from '@/components/onboarding/first-run/assembly'
import { $introCopy, skipIntro } from '@/components/onboarding/first-run/intro'
import { group } from '@/components/pane-shell/tree/model'
import { applyLayoutPreset } from '@/components/pane-shell/tree/presets'
import { $activePresetId, $layoutTree } from '@/components/pane-shell/tree/store'
import { setPrimaryGateway } from '@/store/gateway'
import { $chatOnboardingSolo, $introView } from '@/store/onboarding-intro'
import { $onboardingPhase } from '@/store/onboarding-phase'
import {
  $onboardingChatOpener,
  noteOnboardingEnded,
  type OnboardingRequest,
  skipOnboarding
} from '@/store/onboarding-run'
import { $activeSessionId, $connection, $selectedStoredSessionId } from '@/store/session'

import { OnboardingSettings } from './onboarding-settings'

const started = { resumed: false, run_id: 'run-2', session_id: 'rt-2', stored_session_id: 'st-2' }

type RequestParams = Parameters<OnboardingRequest>[1]

const calls: [string, RequestParams][] = []

const answerGateway: OnboardingRequest = async <T,>(method: string, params?: RequestParams): Promise<T> => {
  calls.push([method, params])

  // SAFETY: the fake answers onboarding.start / onboarding.state with the fields the start reads.
  return (
    method === 'onboarding.state' ? { eligible: true, run: true, run_id: 'run-2', session_id: 'st-2' } : started
  ) as T
}

const size = vi.fn()

// The shell's opener (wiring.tsx::adoptSessionRoute): the stored chat becomes the selected, routed one.
const adopt = vi.fn(async (storedId: string) => {
  $selectedStoredSessionId.set(storedId)
  $activeSessionId.set('rt-2')
})

beforeAll(() => {
  registerLayoutPresets()
})

beforeEach(() => {
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { chatOnboarding: { size }, guestOnboardingEnabled: true }
  })
  // SAFETY: the Settings button's gateway request calls only `request` on the primary socket.
  setPrimaryGateway({ request: answerGateway } as never)
  // SAFETY: factsSource reads only `mode`.
  $connection.set({ mode: 'local' } as never)
  $onboardingChatOpener.set(adopt)
  $onboardingPhase.set('idle')
  $introView.set('off')
  // Out of the last case's demo, if it left one up.
  endChatOnboardingSolo()
  // An existing user's own arrangement, from before the rerun.
  applyLayoutPreset('mine', group(['workspace']))
  // The chat the user was on.
  $selectedStoredSessionId.set('st-old')
  $activeSessionId.set('rt-old')
  calls.length = 0
  size.mockClear()
})

afterEach(() => {
  cleanup()
  setPrimaryGateway(null)
})

async function runAgain() {
  const mine = $layoutTree.get()
  const views: string[] = []
  const stop = $introView.listen(view => views.push(view))

  render(<OnboardingSettings />)
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Run again' })))
  stop()

  return { mine, views }
}

describe('Settings > Run onboarding again', () => {
  it('replays the whole first-run intro in a new chat', async () => {
    const { views } = await runAgain()

    expect(calls.find(([method]) => method === 'onboarding.start')?.[1]).toMatchObject({ rerun: true })
    expect(adopt).toHaveBeenCalledWith('st-2', true)
    // The boot overlay holds while the chat opens, then the intro plays over the demo layout.
    expect(views).toEqual(['starting', 'intro'])
    expect(size).toHaveBeenCalledWith('onboarding')
    expect($chatOnboardingSolo.get()).toBe(true)
    expect($activePresetId.get()).toBe(DEMO_LAYOUT_ID)
    // The new chat is blank, so the intro copy plays.
    expect($introCopy.get()).toBe('playing')
  })

  it("puts the user's layout and window size back when the run ends", async () => {
    const { mine } = await runAgain()

    act(() => noteOnboardingEnded({ outcome: 'done', run_id: 'run-2', stored_session_id: 'st-2' }))

    expect($chatOnboardingSolo.get()).toBe(false)
    expect(size).toHaveBeenLastCalledWith('normal')
    expect($layoutTree.get()).toEqual(mine)
    expect($activePresetId.get()).toBe('mine')
  })

  it("puts the user's layout and window size back on Skip setup", async () => {
    const { mine } = await runAgain()

    // What the Skip setup button does.
    await act(async () => skipOnboarding(answerGateway, 'rt-2').then(skipIntro))

    expect($chatOnboardingSolo.get()).toBe(false)
    expect(size).toHaveBeenLastCalledWith('normal')
    expect($layoutTree.get()).toEqual(mine)
    expect($activePresetId.get()).toBe('mine')
  })
})
