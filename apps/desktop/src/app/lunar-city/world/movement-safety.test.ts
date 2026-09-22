import { describe, expect, it } from 'vitest'

import type { NavigationCollider } from '../model'

import { createMovementSafety } from './movement-safety'

const floor = { positions: new Float32Array([-10,2.2,-10,10,2.2,-10,10,2.2,10,-10,2.2,10]), indices: new Uint32Array([0,1,2,0,2,3]) }
const point = (x: number, z = 0, y = 2.2) => ({ x, y, z })
describe('physical movement safety', () => {
  it('grounds on actual triangles and rejects off-mesh, underground and unsupported crossings', () => {
    const safety = createMovementSafety(floor)
    expect(safety.resolvePosition(point(2,3,2.3))?.y).toBeCloseTo(2.2)
    const border = point(10 + 1e-6)
    expect(safety.resolvePosition(border)).toBeDefined()
    expect(safety.canTraverse(border, border)).toBe(true)
    expect(safety.resolvePosition(point(12))).toBeUndefined()
    expect(safety.resolvePosition(point(0,0,-2))).toBeUndefined()
    expect(safety.canTraverse(point(0),point(20))).toBe(false)

    const gap = createMovementSafety({
      positions: new Float32Array([-2,0,-1,.03,0,-1,.03,0,1,-2,0,1,.04,0,-1,2,0,-1,2,0,1,.04,0,1]),
      indices: new Uint32Array([0,1,2,0,2,3,4,5,6,4,6,7])
    })

    // A 1cm gap between sampling points must still stop the complete segment.
    expect(gap.canTraverse(point(-.1,0,0),point(.1,0,0))).toBe(false)
  })
  it('sweeps actor radius through rotated buildings and cylindrical trunks without false tree-square corners', () => {
    const box: NavigationCollider = {id:'building',kind:'box',center:point(0,0,4),halfExtents:{x:1,y:2,z:2},rotationY:Math.PI/4}
    const tree: NavigationCollider = {id:'tree',kind:'cylinder',center:point(5,0,4),radius:.3,height:4}
    const safety = createMovementSafety(floor,[box,tree])
    expect(safety.canTraverse(point(-4),point(4))).toBe(false)
    expect(safety.resolvePosition(point(0))).toBeUndefined()
    expect(safety.canTraverse(point(4),point(6))).toBe(false)
    expect(safety.canTraverse(point(4,1),point(6,1))).toBe(true)
    expect(safety.resolvePosition(point(5.5,.5))).toBeDefined()
  })
  it('retains worker clearance while taller and wider leader envelopes reject undersized openings', () => {
    const header: NavigationCollider = {id:'header',kind:'box',center:point(0,0,4.2),halfExtents:{x:2,y:.2,z:.2},rotationY:0}
    expect(createMovementSafety(floor,[header]).canTraverse(point(0,-1),point(0,1))).toBe(true)
    expect(createMovementSafety(floor,[header],{radius:.55,height:2.1}).canTraverse(point(0,-1),point(0,1))).toBe(false)
    const walls: NavigationCollider[] = [-1,1].map(x => ({id:String(x),kind:'box',center:point(x,0,3.2),halfExtents:{x:.5,y:1,z:.2},rotationY:0}))
    expect(createMovementSafety(floor,walls).canTraverse(point(0,-1),point(0,1))).toBe(true)
    expect(createMovementSafety(floor,walls,{radius:.55,height:2.1}).canTraverse(point(0,-1),point(0,1))).toBe(false)
  })

})
