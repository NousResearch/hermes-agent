import { describe, expect, it } from 'vitest'

import { planPluginOpenSession } from './plugin-open-session-plan'

describe('planPluginOpenSession', () => {
  it('defaults to navigation: dial the worker, do not steal chrome', () => {
    const plan = planPluginOpenSession({
      activeProfile: 'default',
      profile: 'worker'
    })

    expect(plan.switchWorkspace).toBeNull()
    expect(plan.dialWithoutSwitching).toBe('worker')
    expect(plan.showAllProfiles).toBe(true)
    expect(plan.requireActiveProfileForHydration).toBe(false)
  })

  it('does not steal chrome even when keepAllProfilesScope is explicitly true', () => {
    const plan = planPluginOpenSession({
      activeProfile: 'default',
      keepAllProfilesScope: true,
      profile: 'worker'
    })

    expect(plan.switchWorkspace).toBeNull()
    expect(plan.dialWithoutSwitching).toBe('worker')
    expect(plan.showAllProfiles).toBe(true)
    expect(plan.requireActiveProfileForHydration).toBe(false)
  })

  it('does not re-dial when chrome is already on that profile', () => {
    const plan = planPluginOpenSession({
      activeProfile: 'worker',
      keepAllProfilesScope: true,
      profile: 'worker'
    })

    expect(plan.switchWorkspace).toBeNull()
    expect(plan.dialWithoutSwitching).toBeNull()
    // Still restores the unified list: re-opening an already-live bot must not
    // leave Sessions collapsed onto a profile whose forever-chat is hidden.
    expect(plan.showAllProfiles).toBe(true)
  })

  it('treats keepAllProfilesScope:false as an explicit workspace switch', () => {
    const plan = planPluginOpenSession({
      activeProfile: 'default',
      keepAllProfilesScope: false,
      profile: 'worker'
    })

    expect(plan.switchWorkspace).toBe('worker')
    expect(plan.dialWithoutSwitching).toBeNull()
    expect(plan.showAllProfiles).toBe(false)
    expect(plan.requireActiveProfileForHydration).toBe(true)
  })

  it('treats an empty profile as neither a dial nor a workspace switch', () => {
    const plan = planPluginOpenSession({
      activeProfile: 'default',
      keepAllProfilesScope: true,
      profile: '  '
    })

    expect(plan.switchWorkspace).toBeNull()
    expect(plan.dialWithoutSwitching).toBeNull()
    expect(plan.showAllProfiles).toBeNull()
  })
})
