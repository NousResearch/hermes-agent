import { describe, expect, it, vi } from 'vitest'

import { setWorkspaceScope } from '@/components/pane-shell/workspace-scope'
import { $activeGatewayProfile } from '@/store/profile'

import {
  $petActivity,
  $petAtRest,
  $petMotion,
  $petState,
  derivePetState,
  flashPetActivity,
  hasPetSpriteForMeta,
  mergePetInfoMeta,
  type PetInfo,
  petOwner,
  petOwnerChanged,
  petOwnerUsesAmbientGateway,
  petProfile,
  requestPetForOwner,
  setPetActivity
} from './pet'

describe('pet owner routing', () => {
  it('follows the exact Bot workspace owner instead of the ambient default profile', () => {
    $activeGatewayProfile.set('default')
    setWorkspaceScope('bots', 'jaime-vpn::scout', {
      kind: 'route',
      route: {
        connectionId: 'jaime-vpn',
        mode: 'remote',
        profile: 'scout',
        targetProfile: 'scout'
      }
    })

    expect(petProfile()).toBe('scout')
    expect(petOwner()).toEqual({
      connectionId: 'jaime-vpn',
      profile: 'scout',
      targetProfile: 'scout'
    })
    expect(petOwnerUsesAmbientGateway(petOwner())).toBe(false)

    setWorkspaceScope('sessions')
  })

  it('does not treat initial owner hydration as a switch', () => {
    const ambient = { profile: 'default', targetProfile: 'default' }
    const routed = { connectionId: 'jaime-vpn', profile: 'scout', targetProfile: 'scout' }

    expect(petOwnerChanged(undefined, ambient)).toBe(false)
    expect(petOwnerChanged(ambient, { ...ambient })).toBe(false)
    expect(petOwnerChanged(ambient, routed)).toBe(true)
  })

  it('falls back to ambient ownership for a malformed Bot route', () => {
    $activeGatewayProfile.set('default')
    setWorkspaceScope('bots', 'broken::route', {
      kind: 'route',
      route: { connectionId: undefined, profile: undefined } as never
    })

    expect(() => petOwner()).not.toThrow()
    expect(petOwner()).toEqual({ profile: 'default', targetProfile: 'default' })

    setWorkspaceScope('sessions')
  })

  it('routes a Bot pet RPC through the selected bot connection', async () => {
    const ambient = vi.fn()
    const routed = vi.fn(async () => ({ slug: 'cache-capy' }))

    const owner = {
      connectionId: 'jaime-vpn',
      profile: 'scout',
      targetProfile: 'scout'
    }

    await expect(
      requestPetForOwner(owner, 'pet.info', { knownRevision: 'old' }, ambient, routed)
    ).resolves.toEqual({ slug: 'cache-capy' })
    expect(routed).toHaveBeenCalledWith('jaime-vpn', 'scout', 'pet.info', {
      knownRevision: 'old',
      profile: 'scout'
    })
    expect(ambient).not.toHaveBeenCalled()
  })

  it('uses the ambient profile outside an exactly routed Bot workspace', () => {
    $activeGatewayProfile.set('nightwatch')
    setWorkspaceScope('bots', 'group:ops', {
      kind: 'blocked',
      message: 'Group chats have no single owner.'
    })

    expect(petProfile()).toBe('nightwatch')
    expect(petOwner()).toEqual({ profile: 'nightwatch', targetProfile: 'nightwatch' })
    expect(petOwnerUsesAmbientGateway(petOwner())).toBe(true)

    setWorkspaceScope('sessions')
    $activeGatewayProfile.set('default')
  })
})

describe('derivePetState', () => {
  it('rests at idle by default and uses waiting when awaiting input', () => {
    expect(derivePetState({})).toBe('idle')
    expect(derivePetState({ awaitingInput: true })).toBe('waiting')
  })

  it('runs when busy or a tool is executing', () => {
    expect(derivePetState({ busy: true })).toBe('run')
    expect(derivePetState({ toolRunning: true })).toBe('run')
  })

  it('reviews while reasoning (below tool, above bare busy)', () => {
    expect(derivePetState({ reasoning: true })).toBe('review')
    expect(derivePetState({ reasoning: true, busy: true })).toBe('review')
    expect(derivePetState({ reasoning: true, toolRunning: true })).toBe('run')
  })

  it('waits (blocked on the user) above the in-flight signals', () => {
    expect(derivePetState({ awaitingInput: true, toolRunning: true, busy: true })).toBe('waiting')
    // but a finish beat still wins over waiting
    expect(derivePetState({ justCompleted: true, awaitingInput: true })).toBe('wave')
  })

  it('honors the full priority chain: error > celebrate > complete > tool', () => {
    expect(derivePetState({ error: true, celebrate: true, busy: true })).toBe('failed')
    expect(derivePetState({ celebrate: true, justCompleted: true, toolRunning: true })).toBe('jump')
    expect(derivePetState({ justCompleted: true, toolRunning: true })).toBe('wave')
  })
})

describe('roam motion', () => {
  it('only reports at-rest when the agent-driven state is plain idle', () => {
    $petActivity.set({})
    expect($petAtRest.get()).toBe(true)

    $petActivity.set({ busy: true })
    expect($petAtRest.get()).toBe(false)

    $petActivity.set({})
    expect($petAtRest.get()).toBe(true)
  })

  it('shows the roam pose while wandering, but never overrides real activity', () => {
    $petActivity.set({})
    $petMotion.set('run')
    expect($petState.get()).toBe('run')

    // Hops surface the jump pose.
    $petMotion.set('jump')
    expect($petState.get()).toBe('jump')

    // Activity wins over a wander in progress.
    $petActivity.set({ reasoning: true, busy: true })
    expect($petState.get()).toBe('review')

    // Back at rest, the wander resumes its pose; clearing it returns to idle.
    $petActivity.set({})
    expect($petState.get()).toBe('jump')
    $petMotion.set(null)
    expect($petState.get()).toBe('idle')

    $petActivity.set({})
  })
})

describe('pet info metadata cache helpers', () => {
  it('treats matching slug and spritesheet revision as a reusable sprite payload', () => {
    const current = {
      enabled: true,
      slug: 'boba',
      displayName: 'Old Boba',
      scale: 0.33,
      spritesheetBase64: 'large-sprite-payload',
      spritesheetRevision: '100:2048'
    }

    const meta = {
      enabled: true,
      slug: 'boba',
      displayName: 'Boba',
      scale: 0.5,
      spritesheetRevision: '100:2048'
    }

    expect(hasPetSpriteForMeta(current, meta)).toBe(true)
    expect(mergePetInfoMeta(current, meta)).toMatchObject({
      enabled: true,
      slug: 'boba',
      displayName: 'Boba',
      scale: 0.5,
      spritesheetBase64: 'large-sprite-payload',
      spritesheetRevision: '100:2048'
    })
  })

  it('returns the same reference when nothing changed to avoid redundant store updates', () => {
    const current: PetInfo = {
      enabled: true,
      slug: 'boba',
      displayName: 'Boba',
      scale: 0.33,
      spritesheetBase64: 'large-sprite-payload',
      spritesheetRevision: '100:2048'
    }

    const meta = {
      enabled: true,
      slug: 'boba',
      displayName: 'Boba',
      scale: 0.33,
      spritesheetRevision: '100:2048'
    }

    expect(mergePetInfoMeta(current, meta)).toBe(current)
  })
})

describe('flashPetActivity', () => {
  it('clears stale sibling beats so a completion never inherits a prior error', () => {
    // A turn errors (sad), then the next turn finishes cleanly. The celebrate
    // beat must win — error is highest priority, so a merge-only flash would
    // keep the pet on the failed pose.
    setPetActivity({ error: true })
    flashPetActivity({ celebrate: true })

    expect($petActivity.get().error).toBe(false)
    expect($petState.get()).toBe('jump')

    setPetActivity({})
  })
})
