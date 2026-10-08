import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import type { HermesConnection } from '@/global'
import { $questionnaireDecided } from '@/store/onboarding-presence'
import { setConnection } from '@/store/session'

import { $questionnaireAvailable, decideQuestionnaire, type OnboardingRequester, readRunState, runSetupAgain } from './due'
import { $questionnaire, closeQuestionnaire } from './store'

const answering =
  (value: null | Record<string, boolean | number | string>): OnboardingRequester =>
  async <T>() =>
    // SAFETY: the due check validates the answer's shape before reading it.
    value as T

const failing: OnboardingRequester = async () => {
  throw Object.assign(new Error('method not found'), { code: -32601 })
}

// SAFETY: availability reads only mode and baseUrl from the connection; the rest is window chrome.
const LOCAL = { baseUrl: 'http://127.0.0.1:9119', isFullscreen: false, mode: 'local' } as HermesConnection
// SAFETY: as above.
const REMOTE = { baseUrl: 'https://vps.example:9119', isFullscreen: false, mode: 'remote' } as HermesConnection
// SAFETY: as above. A pooled local profile runs its own backend on another port.
const LOCAL_WORK = { baseUrl: 'http://127.0.0.1:9120', isFullscreen: false, mode: 'local', profile: 'work' } as HermesConnection

beforeEach(() => {
  $questionnaireDecided.set(false)
  setConnection(LOCAL)
})

afterEach(() => {
  closeQuestionnaire('skipped')
  setConnection(null)
})

describe('questionnaire due check', () => {
  it('opens when the backend says run and eligible, and releases the waiters', async () => {
    expect(await decideQuestionnaire(answering({ eligible: true, run: true }))).toBe(true)
    expect($questionnaire.get().phase).toBe('shown')
    expect($questionnaireDecided.get()).toBe(true)
    expect($questionnaireAvailable.get()).toBe(true)
  })

  it.each([
    ['not eligible', { eligible: false, run: true }],
    ['not due', { eligible: true, run: false }],
    ['the old shape under the same name', { eligible: true, failed_starts: 0, intro: 'unseen' }],
    ['a non-bool run', { eligible: true, run: 'yes' }],
    ['no answer', null]
  ])('is not due for %s', async (_case, value) => {
    expect(await decideQuestionnaire(answering(value))).toBe(false)
    expect($questionnaire.get().phase).not.toBe('shown')
    expect($questionnaireDecided.get()).toBe(true)
  })

  it('is not due when the backend lacks the method', async () => {
    expect(await decideQuestionnaire(failing)).toBe(false)
    expect($questionnaireDecided.get()).toBe(true)
    expect($questionnaireAvailable.get()).toBe(false)
  })

  it('stops offering Run setup again once the window moves to a remote backend, and refuses to run it there', async () => {
    const sent: string[] = []

    const request: OnboardingRequester = async <T>(method: string) => {
      sent.push(method)

      // SAFETY: only onboarding.state is read back, and it answers that shape.
      return { eligible: true, run: false } as T
    }

    await decideQuestionnaire(request)
    expect($questionnaireAvailable.get()).toBe(true)

    setConnection(REMOTE)
    expect($questionnaireAvailable.get()).toBe(false)

    await runSetupAgain(request)
    expect(sent).toEqual(['onboarding.state'])
    expect($questionnaire.get().phase).not.toBe('shown')
  })

  it('keeps offering Run setup again across local profiles, whose pooled backends answer on other ports', async () => {
    await decideQuestionnaire(answering({ eligible: true, run: false }))

    setConnection(LOCAL_WORK)
    expect($questionnaireAvailable.get()).toBe(true)

    setConnection(REMOTE)
    expect($questionnaireAvailable.get()).toBe(false)

    setConnection(LOCAL)
    expect($questionnaireAvailable.get()).toBe(true)
  })

  it('reads only the new shape', () => {
    expect(readRunState({ eligible: true, run: false })).toEqual({ eligible: true, run: false })
    const agenticStateAnswer = { eligible: true, intro: 'seen' }

    expect(readRunState(agenticStateAnswer)).toBeNull()
  })
})
