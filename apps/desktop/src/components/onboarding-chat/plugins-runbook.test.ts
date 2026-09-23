import { describe, expect, it } from 'vitest'

import { buildFirstTaskRunbook, pluginsRunbook } from '@/components/onboarding-chat/setup-profile'
import type { DesktopMachineProfile } from '@/global'
import { $machine } from '@/store/machine'
import { DEFAULT_ANSWERS } from '@/store/onboarding-answers'
import { buildChatOnboardingPrompt, forkOptions } from '@/store/onboarding-script'

describe('plugins in the handoff runbook', () => {
  it('names every settled outcome and never tells the build agent to install', () => {
    const text = pluginsRunbook({
      pluginOutcomes: {
        blender: { detail: '', state: 'installed', tools: ['mcp__blender__a', 'mcp__blender__b'] },
        'nvidia-app': { detail: 'unavailable on darwin', state: 'failed', tools: [] },
        'nvidia-broadcast': { detail: '', state: 'skipped', tools: [] }
      },
      plugins: ['blender', 'nvidia-app', 'nvidia-broadcast', 'extra']
    })

    expect(text).toContain('ready in this chat now: blender (2 tools)')
    expect(text).toMatch(/not installed: nvidia-app \(failed: unavailable on darwin\), nvidia-broadcast \(skipped/)
    expect(text).toContain('not offered for install: extra')
    expect(text).not.toMatch(/manage_catalog|hermes plugins install/)
  })

  it('adds nothing when no plugin was picked', () => {
    expect(pluginsRunbook(DEFAULT_ANSWERS)).toBe('')
    expect(buildFirstTaskRunbook('Organize my work', DEFAULT_ANSWERS)).not.toContain('PLUGINS FROM ONBOARDING')
  })
})

it('places the install beat after the first-task step and before the handoff line', () => {
  const prompt = buildChatOnboardingPrompt('Sid')
  const install = prompt.indexOf('THE INSTALL BEAT')

  expect(install).toBeGreaterThan(prompt.indexOf('step="first"'))
  expect(install).toBeLessThan(prompt.indexOf('7. THE HANDOFF'))
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
  expect(buildChatOnboardingPrompt('Sid')).toContain('run the install beat for nvidia-app and nvidia-broadcast')

  $machine.set({ ...machine, nvidia: false, platform: 'darwin' })
  expect(forkOptions()).toContain('Help me make something in Blender')
  expect(forkOptions()).not.toContain('Set up my games and streaming')
  $machine.set(null)
})
