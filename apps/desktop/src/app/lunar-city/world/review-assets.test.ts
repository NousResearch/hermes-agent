import { describe, expect, it, vi } from 'vitest'

import type { BabylonNodeLike, BabylonSceneLike, LunarCityWorldModules } from '../model'

import { importedClipHasMotion, importReviewLeader, projectWorldPoint, reviewAnimationGroups } from './review-assets'

describe('card model review integration', () => {
  it('recognizes held poses from constant tracks and retains real deformation tracks', () => {
    const group = (values: number[]) => ({
      name: 'wait',
      targetedAnimations: [{ animation: { getKeys: () => values.map(value => ({ value })) } }]
    })

    expect(importedClipHasMotion(group([0, 0, 0]))).toBe(false)
    expect(importedClipHasMotion(group([0, 0.02, 0]))).toBe(true)
    expect(importedClipHasMotion({ name: 'unknown' })).toBeUndefined()

    const weightedTarget = {},
      unusedTarget = {}

    const animated = {
      name: 'wait',
      targetedAnimations: [{ target: unusedTarget, animation: { getKeys: () => [{ value: 0 }, { value: 1 }] } }]
    }

    expect(importedClipHasMotion(animated, new Set([weightedTarget]))).toBe(false)
    expect(importedClipHasMotion(animated, new Set([unusedTarget]))).toBe(true)
  })
  it('exposes only imported playable clips and stops automatic playback until requested', () => {
    const idle = { name: 'leader:cat:idle', start: vi.fn(), stop: vi.fn() }
    const walk = { name: 'walk', start: vi.fn(), stop: vi.fn() }
    const result = reviewAnimationGroups('cat', [idle, walk, { name: 'talking' }])
    expect(result.stateClips).toEqual({ idle: idle.name, walk: walk.name })
    expect(result.groups.get('walk')).toBe(walk)
    expect(idle.stop).toHaveBeenCalledOnce()
    expect(walk.stop).toHaveBeenCalledOnce()
    expect(idle.start).not.toHaveBeenCalled()
  })

  it('retains source scale, hierarchy and materials while making every mesh select the same static leader', async () => {
    const root: BabylonNodeLike = { name: 'conversion-root' }
    const material = { alpha: 1 }
    const body = { name: 'body', parent: root, material }
    const eyes = { name: 'eyes', parent: root, material }
    const position = vi.fn()
    const rotation = vi.fn()

    const modules = {
      ImportMeshAsync: vi.fn(async () => ({ meshes: [body, eyes], transformNodes: [root], animationGroups: [] })),
      TransformNode: class {
        name = 'wrapper'
        position = { set: position }
        rotation = { set: rotation }
      }
    } as unknown as LunarCityWorldModules

    const loaded = await importReviewLeader(
      { id: 'owl', uri: 'models/owl.glb', heightMetres: 1.5, position: { x: 3, y: 2, z: 5 }, rotationY: Math.PI },
      {} as BabylonSceneLike,
      modules,
      uri => `/review/${uri}`
    )

    expect(body.parent).toBe(root)
    expect(root.parent).toBe(loaded.root)
    expect(body.material).toBe(material)
    expect(position).toHaveBeenCalledWith(3, 2, 5)
    expect(rotation).toHaveBeenCalledWith(0, Math.PI, 0)
    expect(loaded.root.scaling).toBeUndefined()
    expect(loaded.stateClips).toEqual({})
    expect((eyes as BabylonNodeLike).metadata?.lunarCity).toMatchObject({
      leaderId: 'owl',
      focusEntityKey: loaded.focusEntityKey
    })
  })

  it('projects badge percentages and hides points behind or outside the camera', () => {
    const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    expect(projectWorldPoint({ x: 0.5, y: 0.5, z: 0.5 }, identity)).toEqual({ x: 75, y: 25, visible: true })
    expect(projectWorldPoint({ x: 2, y: 0, z: 0.5 }, identity)?.visible).toBe(false)
    expect(projectWorldPoint({ x: 0, y: 0, z: -1 }, identity)?.visible).toBe(false)
    expect(projectWorldPoint({ x: 0, y: 0, z: 0.5 }, [...identity.slice(0, 15), -1])?.visible).toBe(false)
  })
})
