import { describe, expect, it } from 'vitest'

import { buildFirstTaskRunbook, pluginsRunbook } from '@/components/onboarding-chat/setup-profile'
import type { DesktopMachineProfile } from '@/global'
import { $machine } from '@/store/machine'
import { DEFAULT_ANSWERS } from '@/store/onboarding-answers'
import { forkOptions } from '@/store/onboarding-script'

describe('plugins in the handoff runbook', () => {
  it('adds nothing when no plugin was picked', () => {
    expect(pluginsRunbook(DEFAULT_ANSWERS)).toBe('')
    expect(buildFirstTaskRunbook('Organize my work', DEFAULT_ANSWERS)).not.toContain('PLUGINS FROM ONBOARDING')
  })
})

it('offers the NVIDIA games pill only where its plugins run, and Blender everywhere', () => {
  const machine: DesktopMachineProfile = {
    ageDays: 400,
    arch: 'x64',
    locale: 'en-US',
    model: 'PC',
    nvidia: true,
    platform: 'win32',
    release: '10',
    username: 'sid'
  }

  $machine.set(machine)
  expect(forkOptions()).toEqual(
    expect.arrayContaining(['Set up my games and streaming', 'Help me make something in Blender'])
  )

  $machine.set({ ...machine, nvidia: false, platform: 'darwin' })
  expect(forkOptions()).toContain('Help me make something in Blender')
  expect(forkOptions()).not.toContain('Set up my games and streaming')
  $machine.set(null)
})
