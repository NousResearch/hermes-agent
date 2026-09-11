import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'

import { $machine } from '@/store/machine'

import { OnboardingChatDirective } from './directive'
import { $setupHandoff, buildFirstTaskRunbook, parseHandoffPlan, resetSetupHandoffForTests } from './setup-profile'

const ANSWERS = { connectors: [], name: 'BK' } as unknown as Parameters<typeof buildFirstTaskRunbook>[1]

describe('the machine-setup plan', () => {
  beforeEach(() => {
    resetSetupHandoffForTests()
    $machine.set(null)
  })

  it('reads the attr, and anything else is an ordinary build', () => {
    expect(parseHandoffPlan('machine-setup')).toBe('machine-setup')
    expect(parseHandoffPlan(' MACHINE-SETUP ')).toBe('machine-setup')
    expect(parseHandoffPlan('drivers')).toBe('build')
    expect(parseHandoffPlan(undefined)).toBe('build')
  })

  it('sends the machine-setup agent to look before it plans', () => {
    const runbook = buildFirstTaskRunbook('Set up this Mac', ANSWERS, 'machine-setup')

    expect(runbook).toContain('START BY LOOKING, NOT PLANNING')
    expect(runbook).toContain('nvidia-smi')
    expect(runbook).toContain('Want me to run this?')
  })

  it('hands the agent what the app already knows, freshness first', () => {
    $machine.set({
      locale: 'en-US',
      ageDays: 0,
      arch: 'arm64',
      model: '',
      nvidia: true,
      platform: 'win32',
      release: '10.0.26100',
      username: ''
    })

    const runbook = buildFirstTaskRunbook('Set up this Spark', ANSWERS, 'machine-setup')

    expect(runbook).toContain('What the app can already see about it: set up today, an NVIDIA Spark')
  })

  it('keeps a plain build on the build rules, with no machine instructions', () => {
    const runbook = buildFirstTaskRunbook('Plant tracker', ANSWERS)

    expect(runbook).toContain('NO external account')
    expect(runbook).not.toContain('MACHINE SETUP JOB')
  })

  it('never sends either plan to a sign-in to finish the first job', () => {
    for (const plan of ['build', 'machine-setup'] as const) {
      expect(buildFirstTaskRunbook('First job', ANSWERS, plan)).toMatch(/no external account|not need an account/i)
    }
  })

  it('carries the plan from the card through to the handoff', () => {
    render(
      <OnboardingChatDirective
        attrs={{
          brief: 'Get this machine ready for work',
          plan: 'machine-setup',
          step: 'handoff',
          task: 'Set up this Mac'
        }}
        streaming={false}
      />
    )

    // The card performs the handoff on mount — there is no surface to pick.
    expect(screen.queryAllByRole('button')).toHaveLength(0)
    expect($setupHandoff.get()?.plan).toBe('machine-setup')
  })
})
