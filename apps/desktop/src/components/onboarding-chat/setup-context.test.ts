import { describe, expect, it, vi } from 'vitest'

import type { OnboardingAnswers } from '@/store/onboarding-answers'

import { buildFirstTaskRunbook, ensureSetupProfile, SETUP_PROFILE } from './setup-profile'

/**
 * What Setup learns has to reach the session Setup hands to.
 *
 * The whole promise of the handoff is that the work carries on where the
 * conversation left off — an agent that opens by asking their name again has
 * just told them the last five minutes went nowhere. The build session is an
 * ordinary session on the user's own profile, so the seeded runbook is the
 * only carrier: everything Setup learned has to be in it.
 */
const ANSWERS = {
  connectors: ['Notion', 'Slack'],
  context: 'kitchen reno, contractor quotes due Friday',
  name: 'Sam'
} as unknown as OnboardingAnswers

const runbook = () => buildFirstTaskRunbook('Plant tracker', ANSWERS)

describe('the picture Setup hands to the build session', () => {
  it('carries every fact the user gave', () => {
    for (const fact of ['Sam', 'kitchen reno', 'Notion', 'Slack']) {
      expect(runbook(), `the runbook drops "${fact}"`).toContain(fact)
    }
  })

  it('tells the agent not to ask again for what it was handed', () => {
    expect(runbook()).toMatch(/never introduce yourself or ask who they are/i)
    expect(runbook()).toMatch(/without re-asking/i)
  })

  // Naming the tools without this reads as "you have Slack" — and the first
  // build is the one thing that must never bounce the user into an OAuth page.
  it('names their tools as NOT connected', () => {
    expect(runbook()).toMatch(/none are connected yet/i)
  })

  // Setup can be skipped, and every answer is optional on the way through.
  it('says nothing at all about a user who told Setup nothing', () => {
    const bare = buildFirstTaskRunbook('Plant tracker', { connectors: [] } as unknown as OnboardingAnswers)

    expect(bare).not.toMatch(/undefined|\bnull\b/)
    expect(bare).not.toMatch(/user is called\b/i)
  })
})

describe('creating the setup profile', () => {
  it('clones the default profile with shared auth and no shell alias', async () => {
    const request = vi.fn().mockResolvedValue({})

    await ensureSetupProfile(request)

    expect(request).toHaveBeenCalledExactlyOnceWith(
      'profiles.create',
      expect.objectContaining({
        name: SETUP_PROFILE,
        clone_from: 'default',
        share_auth: true,
        no_alias: true
      })
    )
  })

  it('propagates creation failures while allowing an existing setup profile', async () => {
    const failure = new Error('Backend unavailable')

    await expect(ensureSetupProfile(vi.fn().mockRejectedValue(failure))).rejects.toBe(failure)
    await expect(
      ensureSetupProfile(vi.fn().mockRejectedValue(new Error('Profile already exists')))
    ).resolves.toBeUndefined()
  })
})
