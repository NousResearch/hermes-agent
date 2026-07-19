import { afterEach, beforeEach, describe, expect, expectTypeOf, it, vi } from 'vitest'

import { $petSignals, clearPetSignals } from '@/store/pet-signals'
import { $activeGatewayProfile } from '@/store/profile'

import {
  createPluginContext,
  PLUGIN_PET_ACTIVITY_LIMITS,
  type PluginContext,
  type PluginPetActivityInput
} from './plugin'

let pluginSequence = 0
let originalProfile = 'default'

interface TestContext {
  ctx: PluginContext
  dispose: () => void
}

function testContext(id = `pet-test-${++pluginSequence}`): TestContext {
  const disposers: (() => void)[] = []
  const ctx = createPluginContext(id, dispose => disposers.push(dispose))

  return {
    ctx,
    dispose: () => {
      for (const dispose of disposers) {
        dispose()
      }
    }
  }
}

const activity = (overrides: Partial<PluginPetActivityInput> = {}): PluginPetActivityInput => ({
  id: 'job',
  state: 'working',
  ttlMs: PLUGIN_PET_ACTIVITY_LIMITS.minTtlMs,
  ...overrides
})

beforeEach(() => {
  originalProfile = $activeGatewayProfile.get()
})

afterEach(() => {
  for (const source of new Set($petSignals.get().map(signal => signal.source))) {
    clearPetSignals(source)
  }

  $activeGatewayProfile.set(originalProfile)
  vi.useRealTimers()
})

describe('plugin pet activity', () => {
  it('publishes only through the host-injected plugin source', () => {
    const { ctx, dispose } = testContext('source-owner')

    try {
      expectTypeOf<PluginPetActivityInput>().not.toHaveProperty('source')
      expect(() =>
        ctx.pet.publishActivity({ ...activity(), source: 'plugin:spoofed' } as PluginPetActivityInput)
      ).toThrow(/unsupported field/i)
      expect($petSignals.get()).toEqual([])

      ctx.pet.publishActivity(activity())
      expect($petSignals.get()).toMatchObject([{ id: 'job', source: 'plugin:source-owner' }])
    } finally {
      dispose()
    }
  })

  it('clears only the calling plugin source', () => {
    const first = testContext('first-owner')
    const second = testContext('second-owner')

    try {
      first.ctx.pet.publishActivity(activity())
      second.ctx.pet.publishActivity(activity())

      first.ctx.pet.clearActivity('job')

      expect($petSignals.get()).toMatchObject([{ source: 'plugin:second-owner' }])
    } finally {
      first.dispose()
      second.dispose()
    }
  })

  it('clears every visible signal on plugin disposal and rejects late publication', () => {
    const owned = testContext('dispose-owner')

    owned.ctx.pet.publishActivity(activity({ id: 'one' }))
    owned.ctx.pet.publishActivity(activity({ id: 'two' }))
    owned.dispose()

    expect($petSignals.get()).toEqual([])
    expect(() => owned.ctx.pet.publishActivity(activity({ id: 'late' }))).toThrow(/disposed/i)
  })

  it('injects strictly increasing source generations within one millisecond', () => {
    vi.useFakeTimers()
    vi.setSystemTime(1_000)
    const { ctx, dispose } = testContext('same-millisecond')

    try {
      ctx.pet.publishActivity(activity({ id: 'one' }))
      const first = $petSignals.get()[0].createdAt

      ctx.pet.publishActivity(activity({ id: 'two' }))
      const second = $petSignals.get().find(signal => signal.id === 'two')?.createdAt

      ctx.pet.publishActivity(activity({ id: 'one', state: 'thinking' }))
      const replacement = $petSignals.get().find(signal => signal.id === 'one')?.createdAt

      expect(second).toBeGreaterThan(first)
      expect(replacement).toBeGreaterThan(second ?? first)
    } finally {
      dispose()
    }
  })

  it('publishes a newer generation after hot reload and retained tombstones', () => {
    vi.useFakeTimers()
    vi.setSystemTime(2_000)
    const first = testContext('hot-reload')

    first.ctx.pet.publishActivity(activity())
    const firstGeneration = $petSignals.get()[0].createdAt
    first.dispose()

    const reloaded = testContext('hot-reload')

    try {
      reloaded.ctx.pet.publishActivity(activity({ state: 'thinking' }))
      expect($petSignals.get()[0].createdAt).toBeGreaterThan(firstGeneration)
      expect($petSignals.get()[0].state).toBe('thinking')
    } finally {
      reloaded.dispose()
    }
  })

  it('bounds distinct identities per source without evicting another plugin', () => {
    const bounded = testContext('bounded-owner')
    const other = testContext('other-owner')

    try {
      other.ctx.pet.publishActivity(activity({ id: 'kept' }))

      for (let index = 0; index < PLUGIN_PET_ACTIVITY_LIMITS.maxIdsPerSource; index += 1) {
        bounded.ctx.pet.publishActivity(activity({ id: `job-${index}` }))
      }

      expect(() => bounded.ctx.pet.publishActivity(activity({ id: 'overflow' }))).toThrow(/identity limit/i)
      expect($petSignals.get()).toHaveLength(PLUGIN_PET_ACTIVITY_LIMITS.maxIdsPerSource + 1)
      expect($petSignals.get()).toContainEqual(expect.objectContaining({ id: 'kept', source: 'plugin:other-owner' }))
    } finally {
      bounded.dispose()
      other.dispose()
    }
  })

  it.each([
    ['empty id', { id: '' }],
    ['padded id', { id: ' job ' }],
    ['oversized id', { id: 'x'.repeat(PLUGIN_PET_ACTIVITY_LIMITS.maxIdLength + 1) }],
    ['short TTL', { ttlMs: PLUGIN_PET_ACTIVITY_LIMITS.minTtlMs - 1 }],
    ['long TTL', { ttlMs: PLUGIN_PET_ACTIVITY_LIMITS.maxTtlMs + 1 }],
    ['fractional TTL', { ttlMs: PLUGIN_PET_ACTIVITY_LIMITS.minTtlMs + 0.5 }],
    ['non-finite TTL', { ttlMs: Number.POSITIVE_INFINITY }],
    ['unknown state', { state: 'unknown' }],
    ['idle state', { state: 'idle' }],
    ['negative priority', { priority: PLUGIN_PET_ACTIVITY_LIMITS.minPriority - 1 }],
    ['oversized priority', { priority: PLUGIN_PET_ACTIVITY_LIMITS.maxPriority + 1 }],
    ['fractional priority', { priority: 0.5 }],
    ['non-finite priority', { priority: Number.POSITIVE_INFINITY }],
    ['title metadata', { title: 'not public' }],
    ['action metadata', { actions: [{ id: 'done', label: 'Done' }] }]
  ])('fails closed for %s', (_label, overrides) => {
    const { ctx, dispose } = testContext()

    try {
      expect(() => ctx.pet.publishActivity({ ...activity(), ...overrides } as PluginPetActivityInput)).toThrow()
      expect($petSignals.get()).toEqual([])
    } finally {
      dispose()
    }
  })

  it('clears visible activity on active-profile changes', () => {
    $activeGatewayProfile.set('profile-a')
    const { ctx, dispose } = testContext('profile-owner')

    try {
      ctx.pet.publishActivity(activity())
      expect($petSignals.get()).toHaveLength(1)

      $activeGatewayProfile.set('profile-b')
      expect($petSignals.get()).toEqual([])

      ctx.pet.publishActivity(activity({ state: 'thinking' }))
      expect($petSignals.get()).toMatchObject([{ state: 'thinking' }])
    } finally {
      dispose()
    }
  })

  it('guards a publication disposer against clearing a newer replacement', () => {
    const { ctx, dispose } = testContext('guarded-disposer')

    try {
      const clearFirst = ctx.pet.publishActivity(activity({ state: 'working' }))
      ctx.pet.publishActivity(activity({ state: 'thinking' }))

      clearFirst()
      expect($petSignals.get()).toMatchObject([{ state: 'thinking' }])
    } finally {
      dispose()
    }
  })
})
