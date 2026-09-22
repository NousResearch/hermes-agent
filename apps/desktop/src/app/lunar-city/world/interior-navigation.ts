import type { NavigationCollider, Vec3 } from '../model'

import { createMovementSafety } from './movement-safety'
import type { NavigationQuery } from './navigation'

export interface NavigableInteriorPlan {
  id: string
  exteriorTransform: { position: readonly number[]; rotation: readonly number[] }
  solids: readonly { id: string; kind: string; center: readonly number[]; size: readonly number[] }[]
  routes: readonly (readonly (readonly number[])[])[]
  portals: readonly { entry: readonly number[]; approach: readonly number[] }[]
  runtimePortalStatus: string
  worldAnchors: Readonly<Record<string, readonly number[] | undefined>>
  ramps?: readonly {start: readonly number[]; end: readonly number[]; width: number}[]
}
interface Geometry { positions: Float32Array; indices: Uint32Array }

export function interiorWorldPoint(plan: NavigableInteriorPlan, point: readonly number[]): Vec3 {
  const yaw=plan.exteriorTransform.rotation[1]!, position=plan.exteriorTransform.position

  return {x:position[0]!+point[0]!*Math.cos(yaw)+point[2]!*Math.sin(yaw),y:position[1]!+point[1]!,z:position[2]!-point[0]!*Math.sin(yaw)+point[2]!*Math.cos(yaw)}
}

/** Adds real authored floors and wall solids; the river-bank plan stays closed
 * until its separate approaches can join the verified outdoor floor. */
export function createInteriorNavigation(base: NavigationQuery, geometry: Geometry, colliders: readonly NavigationCollider[], plans: readonly NavigableInteriorPlan[], envelope?: {radius:number;height:number}): NavigationQuery {
  const eligible=plans.filter(plan=>plan.runtimePortalStatus==='authored_approach')

  if (!eligible.length) {return base}
  const ids=new Set(eligible.map(plan=>`building:${plan.id}`))
  const solids=colliders.filter(collider=>!ids.has(collider.id))
  const positions=Array.from(geometry.positions), indices=Array.from(geometry.indices)
  const nodes: Vec3[]=[], portals: number[]=[]

  const addNode=(point:Vec3) => {const existing=nodes.findIndex(node=>Math.hypot(node.x-point.x,node.y-point.y,node.z-point.z)<1e-5);

if(existing>=0){return existing}nodes.push(point);

return nodes.length-1}

  for (const plan of eligible) {
    for(const solid of plan.solids) {
      const center=interiorWorldPoint(plan,solid.center), size=solid.size

      if(solid.kind==='floor') {
        const offset=positions.length/3

        for(const [x,z] of [[-1,-1],[1,-1],[1,1],[-1,1]]) {
          const point=interiorWorldPoint(plan,[solid.center[0]!+x!*size[0]!/2,solid.center[1]!+size[1]!/2,solid.center[2]!+z!*size[2]!/2])
          positions.push(point.x,point.y,point.z)
        }

        indices.push(offset,offset+1,offset+2,offset,offset+2,offset+3)
      } else if(solid.kind!=='ceiling') {
        solids.push({id:`interior:${plan.id}:${solid.id}`,kind:'box',center,halfExtents:{x:size[0]!/2,y:size[1]!/2,z:size[2]!/2},rotationY:plan.exteriorTransform.rotation[1]!})
      }
    }

    for(const ramp of plan.ramps??[]) {
      const dx=ramp.end[0]!-ramp.start[0]!, dz=ramp.end[2]!-ramp.start[2]!, length=Math.hypot(dx,dz), offset=positions.length/3

      for(const [point,sign] of [[ramp.start,-1],[ramp.start,1],[ramp.end,1],[ramp.end,-1]] as const) {
        const transformed=interiorWorldPoint(plan,[point[0]! - sign*dz/length*ramp.width/2,point[1]!,point[2]!+sign*dx/length*ramp.width/2])
        positions.push(transformed.x,transformed.y,transformed.z)
      }

      indices.push(offset,offset+1,offset+2,offset,offset+2,offset+3)
    }

    for(const route of plan.routes) {for(const point of route){addNode(interiorWorldPoint(plan,point))}}

    for(const portal of plan.portals) {addNode(interiorWorldPoint(plan,portal.entry));portals.push(addNode(interiorWorldPoint(plan,portal.approach)))}
  }

  const safety=createMovementSafety({positions:new Float64Array(positions),indices:new Uint32Array(indices)},solids,envelope)
  const edges=new Map<number,Array<{to:number;path:readonly Vec3[];cost:number}>>()
  const length=(path:readonly Vec3[])=>path.slice(1).reduce((sum,p,i)=>sum+Math.hypot(p.x-path[i]!.x,p.y-path[i]!.y,p.z-path[i]!.z),0)

  const verified=(from:Vec3,to:Vec3,path:readonly Vec3[]|undefined): readonly Vec3[]|undefined=>{
    if(!path?.length){return undefined}
    const all=[from,...path,to]

    return all.every((p,i)=>!!safety.resolvePosition(p)&&(i===0||safety.canTraverse(all[i-1]!,p)))?all:undefined
  }

  const connect=(from:number,to:number,path:readonly Vec3[])=>{const values=edges.get(from)??[];values.push({to,path,cost:length(path)});edges.set(from,values)}

  for(let a=0;a<nodes.length;a++){for(let b=a+1;b<nodes.length;b++){
    let path:readonly Vec3[]|undefined

    if(safety.canTraverse(nodes[a]!,nodes[b]!)){path=[nodes[a]!,nodes[b]!]}
    else if(portals.includes(a)&&portals.includes(b)){path=verified(nodes[a]!,nodes[b]!,base.computePath(nodes[a]!,nodes[b]!))}

    if(path){connect(a,b,path);connect(b,a,[...path].reverse())}
  }}

  return {...safety,dispose:()=>base.dispose?.(),computePath(from,to){
    if(!safety.resolvePosition(from)||!safety.resolvePosition(to)){return undefined}

    if(safety.canTraverse(from,to)){return [from,to]}
    const direct=verified(from,to,base.computePath(from,to));

if(direct){return direct}
    const starts=new Map<number,readonly Vec3[]>(),ends=new Map<number,readonly Vec3[]>()
    nodes.forEach((node,index)=>{
      const start=safety.canTraverse(from,node)?[from,node]:portals.includes(index)?verified(from,node,base.computePath(from,node)):undefined
      const end=safety.canTraverse(node,to)?[node,to]:portals.includes(index)?verified(node,to,base.computePath(node,to)):undefined

      if(start){starts.set(index,start)}

if(end){ends.set(index,end)}
    })
    const costs=new Map<number,number>(),paths=new Map<number,readonly Vec3[]>(),pending=new Set<number>()

    for(const [index,path] of starts){costs.set(index,length(path));paths.set(index,path);pending.add(index)}
    const visited=new Set<number>();let best:readonly Vec3[]|undefined,bestCost=Infinity

    while(pending.size){
      const current=[...pending].sort((a,b)=>costs.get(a)!-costs.get(b)!)[0]!
      pending.delete(current);visited.add(current)
      const tail=ends.get(current),cost=costs.get(current)!,path=paths.get(current)!

      if(tail&&cost+length(tail)<bestCost){best=[...path,...tail.slice(1)];bestCost=cost+length(tail)}

      for(const edge of edges.get(current)??[]){
        const next=cost+edge.cost

        if(!visited.has(edge.to)&&next<(costs.get(edge.to)??Infinity)){costs.set(edge.to,next);paths.set(edge.to,[...path,...edge.path.slice(1)]);pending.add(edge.to)}
      }
    }

    return best
  }}
}
