import { describe, expect, it } from 'vitest'

import plans from '../../../../public/lunar-city/interior-plans-v1/plans.json'

import { createInteriorNavigation, interiorWorldPoint } from './interior-navigation'

const empty={positions:new Float32Array(),indices:new Uint32Array()}
describe('authored interior physical routes',()=>{
  it('grounds every actual plan route and routes assigned pickup through door openings while blocking walls',()=>{
    for(const plan of plans.plans){
      const query=createInteriorNavigation({computePath:()=>undefined},empty,[],[plan])

      for(const route of plan.routes){
        const points=route.map(point=>interiorWorldPoint(plan,point))

        for(const point of points){expect(query.resolvePosition?.(point),`${plan.id} ${JSON.stringify(point)}`).toBeDefined()}

        for(let i=1;i<points.length;i++){expect(query.canTraverse?.(points[i-1]!,points[i]!),plan.id).toBe(true)}
      }

      const pickup=plan.worldAnchors.jobPickup!, target={x:pickup[0]!,y:pickup[1]!,z:pickup[2]!}
      expect(query.computePath(interiorWorldPoint(plan,plan.portals[0]!.approach),target),`${plan.id} pickup`).toBeDefined()
      const wall=plan.solids.find(solid=>solid.kind==='wall')

      if(wall){expect(query.resolvePosition?.(interiorWorldPoint(plan,[wall.center[0]!,wall.center[1]!-wall.size[1]!/2,wall.center[2]!]))).toBeUndefined()}
    }
  })
  it('fits each measured leader envelope through its own doors to the role workspace',()=>{
    for(const plan of plans.plans){
      if(!plan.leaderEnvelope){continue}
      const envelope={radius:plan.leaderEnvelope.radiusMetres,height:plan.leaderEnvelope.size[1]!}
      const query=createInteriorNavigation({computePath:()=>undefined},empty,[],[plan],envelope)
      const source=plan.worldAnchors.leaderWork!
      const target={x:source[0]!,y:source[1]!,z:source[2]!}
      expect(query.resolvePosition?.(target),plan.id+' measured workspace').toBeDefined()
      expect(query.computePath(interiorWorldPoint(plan,plan.portals[0]!.approach),target),plan.id+' measured doorway').toBeDefined()
    }
  })
  it('walks up the real waterworks ramp and across its elevated catwalk without creating a river floor',()=>{
    const plan=plans.plans.find(plan=>plan.id==='engineering-workshop')!
    const query=createInteriorNavigation({computePath:()=>undefined},empty,[],[plan])
    const ramp=plan.ramps[0]!

    for(let step=0;step<=20;step++){
      const t=step/20, point=ramp.start.map((value,i)=>value+(ramp.end[i]!-value)*t)
      const world=interiorWorldPoint(plan,point)
      expect(query.resolvePosition?.(world)?.y).toBeCloseTo(world.y,5)
    }

    expect(query.resolvePosition?.(interiorWorldPoint(plan,[0,0,0]))).toBeUndefined()
    expect(query.resolvePosition?.(interiorWorldPoint(plan,[0,.8,ramp.end[2]!]))).toBeDefined()
  })
})
