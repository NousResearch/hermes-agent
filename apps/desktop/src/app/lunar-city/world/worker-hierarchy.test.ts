import {
  Animation,
  AnimationGroup,
  Bone,
  Matrix,
  MeshBuilder,
  NullEngine,
  Scene,
  Skeleton,
  TransformNode
} from '@babylonjs/core'
import { describe, expect, it } from 'vitest'

import type { BabylonNodeLike } from '../model'

import { cloneWorkerHierarchy, fitHistoricalWorker } from './worker-hierarchy'

describe('worker skeleton isolation', () => {
  it('retargets two real cloned skeletons and animations independently and disposes only their skeletons', () => {
    const engine = new NullEngine()
    const scene = new Scene(engine)
    const root = new TransformNode('worker', scene)
    const joint = new TransformNode('joint', scene)
    joint.parent = root
    const mesh = MeshBuilder.CreateBox('body', {}, scene)
    mesh.parent = root
    const skeleton = new Skeleton('source', 'source', scene)
    const bone = new Bone('joint', skeleton, null, Matrix.Identity())
    bone.linkTransformNode(joint)
    mesh.skeleton = skeleton
    const animation = new Animation('walk', 'position.x', 30, Animation.ANIMATIONTYPE_FLOAT)
    animation.setKeys([
      { frame: 0, value: 0 },
      { frame: 30, value: 1 }
    ])
    const group = new AnimationGroup('walk', scene)
    group.addTargetedAnimation(animation, joint)
    const parent = new TransformNode('entities', scene)
    const groups = new Map([['walk', group]])

    const first = cloneWorkerHierarchy(
      root as unknown as BabylonNodeLike,
      parent as unknown as BabylonNodeLike,
      groups,
      'first'
    )

    const second = cloneWorkerHierarchy(
      root as unknown as BabylonNodeLike,
      parent as unknown as BabylonNodeLike,
      groups,
      'second'
    )

    const firstMesh = first.nodeMap.get(mesh) as unknown as typeof mesh
    const secondMesh = second.nodeMap.get(mesh) as unknown as typeof mesh
    expect(firstMesh.skeleton).not.toBe(skeleton)
    expect(firstMesh.skeleton).not.toBe(secondMesh.skeleton)
    expect(firstMesh.skeleton!.bones[0]!.getTransformNode()).toBe(first.nodeMap.get(joint))
    const firstGroup = first.animations.get('walk') as AnimationGroup
    firstGroup.start(false)
    firstGroup.pause()
    firstGroup.goToFrame(15)
    expect(first.nodeMap.get(joint)!.position!.x).toBeCloseTo(0.5)
    expect(second.nodeMap.get(joint)!.position!.x).toBe(0)
    expect(joint.position.x).toBe(0)
    first.disposeSkeletons()
    expect(scene.skeletons).toContain(skeleton)
    expect(scene.skeletons).toContain(secondMesh.skeleton)
    expect(scene.skeletons).not.toContain(firstMesh.skeleton)
    scene.dispose()
    engine.dispose()
  })
})

it('fits enabled historical geometry to metres at its ground pivot without hidden variants or repeated scale multiplication', () => {
  const engine = new NullEngine(), scene = new Scene(engine)
  const anchor = new TransformNode('anchor', scene); anchor.position.set(12, 7, -3)
  const wrapper = new TransformNode('metric', scene); wrapper.parent = anchor
  const body = MeshBuilder.CreateBox('body', { height: 8 }, scene); body.parent = wrapper; body.position.y = 5
  const hidden = MeshBuilder.CreateBox('hidden-variant', { height: 80 }, scene); hidden.parent = wrapper; hidden.setEnabled(false)
  const fit = fitHistoricalWorker(wrapper as never, [body, hidden] as never)!
  expect(fit.scale).toBeCloseTo(1.2 / 8)
  body.computeWorldMatrix(true)
  expect(body.getBoundingInfo().boundingBox.minimumWorld.y).toBeCloseTo(7)
  expect(body.getBoundingInfo().boundingBox.maximumWorld.y).toBeCloseTo(8.2)
  expect(fitHistoricalWorker(wrapper as never, [body, hidden] as never)).toEqual(fit)
  expect(body.scaling.asArray()).toEqual([1, 1, 1])
  scene.dispose(); engine.dispose()
})

it('fits hardware-instance LODs independently while preserving shared source geometry', () => {
  const engine = new NullEngine(), scene = new Scene(engine)

  for (const height of [7.68, 2.3, 2.25]) {
    const source = MeshBuilder.CreateBox(`source-${height}`, { height }, scene)
    source.position.y = height / 2
    const before = [...source.getVerticesData('position')!]
    const root = new TransformNode('metric', scene)
    const instance = source.createInstance('worker'); instance.parent = root
    fitHistoricalWorker(root as never, [instance] as never)
    instance.computeWorldMatrix(true)
    expect(instance.getBoundingInfo().boundingBox.minimumWorld.y).toBeCloseTo(0)
    expect(instance.getBoundingInfo().boundingBox.maximumWorld.y).toBeCloseTo(1.2)
    expect([...source.getVerticesData('position')!]).toEqual(before)
  }

  scene.dispose(); engine.dispose()
})
