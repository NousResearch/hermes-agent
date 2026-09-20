import { afterEach, expect, it } from 'vitest'

import { buildFirstTaskRunbook } from '@/components/onboarding-chat/setup-profile'
import { $machine } from '@/store/machine'
import { DEFAULT_ANSWERS } from '@/store/onboarding-answers'
import { buildChatOnboardingPrompt, forkOptions, machineForkOption } from '@/store/onboarding-script'

afterEach(() => $machine.set(null))

it('hands an account task to the connection flow without substituting a no-auth build', () => {
  for (const connectors of [[], ['gmail', 'notion']]) {
    const runbook = buildFirstTaskRunbook('Check my email', { ...DEFAULT_ANSWERS, connectors })

    expect(runbook).toContain('Check my email')
    expect(runbook).toContain('manage_connections action="status"')
    expect(runbook).toContain('only the apps needed for this task')
    expect(runbook).not.toMatch(
      /must be finishable with NO external|build the no-auth core|connectors set to every one/
    )
    expect(runbook).toContain('never fabricate account data')
  }
})

it('offers connector-based work without displacing the existing fresh-machine and Spark fork', () => {
  const prompt = buildChatOnboardingPrompt()
  expect(prompt).toContain('Use my email to find messages that need a reply')
  expect(prompt).toContain('at most one option per app')
  expect(prompt).not.toContain('the first task must be FINISHABLE with no external account')

  for (const profile of [
    { platform: 'darwin', arch: 'arm64', nvidia: false, model: 'Mac', ageDays: 1 },
    { platform: 'win32', arch: 'arm64', nvidia: true, model: '', ageDays: null },
    { platform: 'linux', arch: 'arm64', nvidia: true, model: 'NVIDIA_DGX_Spark', ageDays: null }
  ]) {
    $machine.set({ ...profile, release: '', username: '', locale: 'en-US' })
    expect(forkOptions()).toEqual([machineForkOption(), 'Something else'])

    if (profile.ageDays === null) {
      expect(buildChatOnboardingPrompt()).not.toMatch(/\bNEW Spark\b/)
    }

    const machineRunbook = buildFirstTaskRunbook(
      'Set up this computer',
      {
        ...DEFAULT_ANSWERS,
        connectors: ['gmail']
      },
      'machine-setup'
    )

    expect(machineRunbook).toContain('START BY LOOKING, NOT PLANNING')
    expect(machineRunbook).not.toContain('CONNECT FOR THIS TASK')
  }
})
