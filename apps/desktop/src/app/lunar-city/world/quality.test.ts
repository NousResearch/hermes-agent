import { describe, expect, it, vi } from 'vitest'

import { animationDistanceUnits, applyQualitySettings, createQualityController, qualitySettings } from './quality'

describe('qualitySettings', () => {
  it('maps short, normal, and long animation tiers to monotonically increasing runtime distances', () => {
    expect(animationDistanceUnits('short')).toBeLessThan(animationDistanceUnits('normal'))
    expect(animationDistanceUnits('normal')).toBeLessThan(animationDistanceUnits('long'))
  })

  it.each([
    ['efficient', 0.7, 'none', 'short', 0],
    ['balanced', 0.85, 'near', 'normal', 0],
    ['detailed', 1, 'near', 'long', 0]
  ] as const)('defines the exact %s tier', (tier, renderScale, dynamicShadows, animationDistance, lodAdvance) => {
    expect(qualitySettings(tier)).toMatchObject({
      animationDistance,
      dynamicShadows,
      lodAdvance,
      renderScale
    })
  })

  it('applies render scale as output pixels divided by display pixels', () => {
    const engine = { setHardwareScalingLevel: vi.fn() }

    applyQualitySettings(engine, qualitySettings('balanced'))

    expect(engine.setHardwareScalingLevel).toHaveBeenCalledWith(1 / 0.85)
  })
})

describe('automatic quality degradation', () => {
  it('resets its visual governor when the operator explicitly changes tiers', () => {
    const controller = createQualityController('detailed')

    controller.setTier('efficient')

    expect(controller.settings()).toMatchObject({
      animationDistance: 'short',
      dynamicShadows: 'none',
      lodAdvance: 0,
      renderScale: 0.7
    })
  })

  it('degrades a detailed world in the exact declared order after each 120 over-budget interactive frames', () => {
    const controller = createQualityController('detailed')

    const observeOverBudget = () => {
      for (let frame = 0; frame < 120; frame += 1) {
        controller.noteFrame({ elapsedMs: 34, interactive: true })
      }
    }

    observeOverBudget()
    expect(controller.settings()).toMatchObject({
      renderScale: 0.85,
      dynamicShadows: 'near',
      animationDistance: 'long'
    })
    observeOverBudget()
    expect(controller.settings()).toMatchObject({ renderScale: 0.7, dynamicShadows: 'near', animationDistance: 'long' })
    observeOverBudget()
    expect(controller.settings()).toMatchObject({ renderScale: 0.7, dynamicShadows: 'none', animationDistance: 'long' })
    observeOverBudget()
    expect(controller.settings()).toMatchObject({ animationDistance: 'short', decorations: true })
    observeOverBudget()
    expect(controller.settings()).toMatchObject({ decorations: false, lodAdvance: 0 })
    observeOverBudget()
    expect(controller.settings()).toMatchObject({ decorations: false, lodAdvance: 1 })
  })

  it('recovers only one degradation step after 600 under-budget frames without changing its selected identity', () => {
    const selected = 'session:stable' as never
    const controller = createQualityController('detailed', { selectedEntityKey: selected })

    for (let frame = 0; frame < 120; frame += 1) {
      controller.noteFrame({ elapsedMs: 34, interactive: true })
    }

    for (let frame = 0; frame < 600; frame += 1) {
      controller.noteFrame({ elapsedMs: 1, interactive: false })
    }

    expect(controller.settings().renderScale).toBe(1)
    expect(controller.selectedEntityKey()).toBe(selected)
  })
})
