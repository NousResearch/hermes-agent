import { expect, it } from 'vitest'

import type { EntityKey, LunarCitySnapshot, LunarEntity } from '../model'

import { observedLeaderLife } from './leader-observed-life'

function profile(connectionId:string):LunarEntity{return {key:`profile:${connectionId}` as EntityKey,identity:{kind:'profile',connectionId,profile:'worker'},authority:'authoritative',observedAt:1,destination:'garden',animation:'rest'}}

function session(connectionId:string,animation='work'):LunarEntity{return {key:`session:${connectionId}` as EntityKey,identity:{kind:'session',connectionId,profile:'worker',sessionId:'one'},authority:'authoritative',observedAt:1,destination:'project',animation}}

function snapshot(...entities:LunarEntity[]):LunarCitySnapshot{return {entities:new Map(entities.map(entity=>[entity.key,entity])),observedAt:1,revision:1,sources:[]}}

it('projects only exact authoritative owner activity and releases work when observed idle',()=>{
 expect(observedLeaderLife(snapshot(profile('a'),session('b')),['cat'])[0]).toMatchObject({mode:'home',reason:'no-session-observation'})
 expect(observedLeaderLife(snapshot(profile('a'),session('a')),['cat'])[0]).toMatchObject({mode:'work',reason:'observed-work'})
 expect(observedLeaderLife(snapshot(profile('a'),session('a','idle')),['cat'])[0]).toMatchObject({mode:'idle',reason:'observed-idle'})
 expect(observedLeaderLife(snapshot(profile('a'),{...session('a'),authority:'stale'}),['cat'])[0]).toMatchObject({mode:'unavailable',reason:'stale-owner'})
})
it('holds a shared model instead of attributing one of multiple owners work',()=>{
 const projected=observedLeaderLife(snapshot(profile('a'),profile('b'),session('a')),['cat'])
 expect(projected[0]).toMatchObject({mode:'unavailable',reason:'ambiguous-model'})
 expect(projected[0]!.ownerKeys).toHaveLength(2)
})
