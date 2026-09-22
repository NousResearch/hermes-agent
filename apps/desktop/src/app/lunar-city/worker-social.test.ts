import { expect, it } from 'vitest'

import { entityKey } from './identity'
import type { LunarCitySnapshot, LunarEntity } from './model'
import { createWorkerSocialController, workerRelationships } from './worker-social'

const worker=(profile:string,x:number):LunarEntity=>{const identity={kind:'session' as const,connectionId:'one',profile,sessionId:'same'};

return {identity,key:entityKey(identity),authority:'authoritative',observedAt:1,destination:'garden',animation:'rest',sourceState:'idle',position:{x,y:0,z:0}}}

const snapshot=(rows:LunarEntity[]):LunarCitySnapshot=>({revision:1,observedAt:1,entities:new Map(rows.map(e=>[e.key,e])),sources:[]})
it('pairs gestures without mutating workers and cancels on work, stale authority and motion suppression',()=>{
 const a=worker('a',0),b=worker('b',2),input=snapshot([a,b]),c=createWorkerSocialController();c.update(input)
 const env={enabled:true,position:(key:typeof a.key)=>input.entities.get(key)?.position,available:()=>true,supports:()=>true}
 expect(c.requestGreeting(a.key,env)).toBe(true)
 const poses=new Map(c.tick(100,env));expect(poses.get(a.key)).toMatchObject({animation:'talk',facing:Math.PI/2});expect(poses.get(b.key)?.animation).toBe('listen')
 expect(new Map(c.tick(4500,env)).get(a.key)?.animation).toBe('listen');expect(a.animation).toBe('rest');expect(a.position?.x).toBe(0)
 c.update(snapshot([a,{...b,sourceState:'running',animation:'work'}]));expect(c.tick(10,env).size).toBe(0)
 c.update(snapshot([{...a,authority:'stale'},b]));expect(c.requestGesture(a.key,'talk',env)).toBe(false)
 c.update(input);expect(c.requestGesture(a.key,'talk',env)).toBe(true);expect(c.tick(10,{...env,enabled:false}).size).toBe(0);expect(c.active()).toBe(false)
 c.update(input);expect(c.requestGesture(a.key,'think',env)).toBe(true);c.update(snapshot([b]));expect(c.tick(1,env).has(a.key)).toBe(false);expect(c.active()).toBe(false)
})
it('caps pairs and matches delegation only within exact parent ownership',()=>{
 const rows=Array.from({length:12},(_,i)=>worker(String(i),i*.8)),input=snapshot(rows),c=createWorkerSocialController();c.update(input)
 c.tick(1,{enabled:true,position:key=>input.entities.get(key)?.position,available:()=>true,supports:()=>true});expect(c.encounters()).toHaveLength(2)
 const parent=rows[0],identity={connectionId:parent.identity.connectionId,profile:parent.identity.profile,sessionId:'same',kind:'subagent' as const,subagentId:'child'},child={...parent,identity,key:entityKey(identity)}
 const foreign={...parent,identity:{...parent.identity,connectionId:'two'}};foreign.key=entityKey(foreign.identity)
 const related=snapshot([parent,child,foreign]);expect(workerRelationships(child,related).map(e=>e.key)).toEqual([parent.key]);expect(workerRelationships(foreign,related)).toEqual([])
})
