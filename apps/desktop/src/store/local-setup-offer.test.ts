import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $questionnaireDecided, setOnboardingSurfaceActive } from '@/store/onboarding-presence'

import { $localSetupOffer, acceptLocalSetupOffer, noteHandoffSession, reportLocalSetupTurnComplete } from './local-setup-offer'

beforeEach(() => {
  $questionnaireDecided.set(false)
  window.__hermesTips?.reset()
})

afterEach(() => setOnboardingSurfaceActive('questionnaire', false))

describe('local-setup offer arming', () => {
  it('waits for the due check, then arms when the questionnaire is not due', () => {
    expect($localSetupOffer.get().state).toBe('unarmed')

    $questionnaireDecided.set(true)

    expect($localSetupOffer.get().state).toBe('armed')
  })

  it('waits for the questionnaire to close', () => {
    setOnboardingSurfaceActive('questionnaire', true)
    $questionnaireDecided.set(true)

    expect($localSetupOffer.get().state).toBe('unarmed')

    setOnboardingSurfaceActive('questionnaire', false)

    expect($localSetupOffer.get().state).toBe('armed')
  })

  it('keeps the local step answer: accepted stays final when the questionnaire closes', () => {
    setOnboardingSurfaceActive('questionnaire', true)
    $questionnaireDecided.set(true)
    acceptLocalSetupOffer()
    setOnboardingSurfaceActive('questionnaire', false)

    expect($localSetupOffer.get().state).toBe('accepted')
  })

  it('skips the first finished turn of the handoff chat', () => {
    $questionnaireDecided.set(true)
    noteHandoffSession('handoff-1')

    reportLocalSetupTurnComplete({ failed: false, sessionId: 'handoff-1' })

    expect($localSetupOffer.get()).toMatchObject({ sessionId: null, state: 'armed' })
  })
})
