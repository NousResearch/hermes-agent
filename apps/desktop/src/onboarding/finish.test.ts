import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $freeTierStatus } from '@/store/free-tier'
import { $localSetupOffer } from '@/store/local-setup-offer'
import { markQuestionnaireDecided, onboardingSurfaceActive } from '@/store/onboarding-presence'
import type { FreeTierStatus } from '@/types/hermes'

import type { OnboardingRequester } from './due'
import { answers, FIXTURES } from './fixtures.test-util'
import { finish, type HandoffDeps, skipSetup } from './handoff'
import type { InferenceClock } from './inference'
import { $questionnaire, closeQuestionnaire, openQuestionnaire } from './store'

const MINTED: FreeTierStatus = {
  available: true,
  enabled: true,
  has_guest: true,
  label: 'Nous · free tier',
  model: 'nous/welcome',
  notice_pending: true
}

const REFUSED: FreeTierStatus = {
  ...MINTED,
  available: false,
  error_code: 'anon_gate_closed',
  has_guest: false,
  retryable: false
}

/** Instant clock: each sleep moves time forward without waiting. */
function fakeClock(): InferenceClock {
  let now = 0

  return {
    now: () => now,
    sleep: async ms => {
      now += ms
    }
  }
}

function harness({ ready = true, launchProfile = 'default' } = {}) {
  const log: string[] = []

  const replies = new Map<string, object>([
    ['free_tier.ack_notice', { acked: true }],
    ['free_tier.status', { ...MINTED, notice_pending: false }],
    ['setup.runtime_check', { ok: ready }],
    ['setup.status', { free_tier_account: true, free_tier_route: true, provider_configured: true, ready }]
  ])

  const request: OnboardingRequester = async <T>(method: string, params?: Parameters<OnboardingRequester>[1]) => {
    log.push(params && 'run' in params ? `${method}:${params.run}:${params.mark_profile_offered}` : method)

    // SAFETY: each method answers the shape the code under test reads back.
    return (replies.get(method) ?? {}) as T
  }

  const deps: HandoffDeps = {
    clock: fakeClock(),
    launchProfile,
    openDefaultChat: async text => {
      log.push(`chat:${text.split('\n')[0]}`)

      return 'runtime-1'
    },
    openLandedChat: async () => false,
    refreshReadiness: async () => {
      log.push('readiness')
    },
    request,
    runTour: async () => {
      log.push(`tour:open=${onboardingSurfaceActive()}`)
    },
    startQuickstart: async model => {
      log.push(`quickstart:${model.id}`)
    }
  }

  return { deps, log }
}

beforeEach(() => {
  markQuestionnaireDecided()
  window.__hermesTips?.reset()
  $freeTierStatus.set(MINTED)
  openQuestionnaire()
})

afterEach(() => {
  closeQuestionnaire('skipped')
  $freeTierStatus.set(null)
})

describe('finish (Start)', () => {
  it('settles the flags, closes, tours, then opens the first chat and starts the quickstart', async () => {
    const { deps, log } = harness()
    const picked = answers({ local: 'yes', task: { id: 'tidy' }, tour: 'quick' })

    expect(await finish(FIXTURES.spark, picked, deps)).toBe('started')

    expect(log).toEqual([
      'setup.status',
      'setup.runtime_check',
      'onboarding.set_run:false:true',
      'free_tier.ack_notice',
      'free_tier.status',
      'readiness',
      'tour:open=false',
      'chat:Tidy my Downloads folder. Show me a dry-run plan before you move anything.',
      'quickstart:qwen3.8-27b'
    ])
    expect($questionnaire.get().phase).toBe('done')
    expect($localSetupOffer.get().state).toBe('accepted')
  })

  it('adopts the free account into default when launched under another profile', async () => {
    const { deps, log } = harness({ launchProfile: 'work' })

    await finish(FIXTURES.mac, answers(), deps)

    expect(log.indexOf('free_tier.provision')).toBeGreaterThan(log.indexOf('setup.runtime_check'))
    expect(log.indexOf('free_tier.provision')).toBeLessThan(log.indexOf('onboarding.set_run:false:true'))
  })

  it('skips the tour and quickstart when not picked', async () => {
    const { deps, log } = harness()

    await finish(FIXTURES.spark, answers({ local: 'no', tour: 'none' }), deps)

    expect(log.some(line => line.startsWith('tour') || line.startsWith('quickstart'))).toBe(false)
    expect($localSetupOffer.get().state).toBe('dismissed')
  })

  it('falls back to the picker after the wait, dropping the prompt and leaving the notice owed', async () => {
    const { deps, log } = harness({ ready: false })

    expect(await finish(FIXTURES.spark, answers({ task: { id: 'tidy' } }), deps)).toBe('failed')

    expect(log).toContain('onboarding.set_run:false:false')
    expect(log).not.toContain('free_tier.ack_notice')
    expect(log.some(line => line.startsWith('chat:'))).toBe(false)
    expect($questionnaire.get().phase).toBe('failed')
  })

  it('stops waiting at once on a terminal free-tier failure', async () => {
    const { deps, log } = harness({ ready: false })

    $freeTierStatus.set(REFUSED)

    expect(await finish(FIXTURES.spark, answers(), deps)).toBe('failed')
    expect(log.filter(line => line === 'setup.status')).toHaveLength(1)
  })
})

describe('skipSetup', () => {
  it('writes run=false without marking the profile offer and leaves the notice owed', async () => {
    const { deps, log } = harness()

    await skipSetup(deps)

    expect(log).toEqual(['onboarding.set_run:false:false', 'readiness'])
    expect($questionnaire.get().phase).toBe('skipped')
  })

  it('holds on Preparing while the free account is still being made', async () => {
    const { deps } = harness()
    const minting = { ...MINTED, available: false, has_guest: false }

    $freeTierStatus.set(minting)
    // A clock that never reaches the Start deadline: only the account settling ends the hold.
    const skipping = skipSetup({ ...deps, clock: { now: () => 0, sleep: () => new Promise<void>(() => {}) } })

    await Promise.resolve()
    await Promise.resolve()
    expect($questionnaire.get().view).toBe('preparing')
    expect(onboardingSurfaceActive()).toBe(true)

    $freeTierStatus.set(MINTED)
    await skipping

    expect($questionnaire.get().phase).toBe('skipped')
  })

  it('stops holding on Preparing after the Start wait when the account keeps failing to be made', async () => {
    const { deps } = harness()
    const rateLimited = { ...MINTED, available: false, error_code: 'anon_rate_limited', has_guest: false, retryable: true }

    $freeTierStatus.set(rateLimited)
    await skipSetup(deps)

    expect($questionnaire.get().phase).toBe('skipped')
    expect(onboardingSurfaceActive()).toBe(false)
  })
})
