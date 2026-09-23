import '../lunar-city.css'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { createRoot } from 'react-dom/client'

import { AlertsPanel } from '../components/alerts-panel'
import { DelegationLinks } from '../components/delegation-links'
import { ProjectCompoundsPanel } from '../components/project-compounds-panel'
import { WorkerLife } from '../components/worker-life'
import { entityKey } from '../identity'
import type { EntityKey, LunarCitySnapshot, LunarCityWorldHandle, LunarEntity, WorldManifestV2 } from '../model'
import { mapObservedState } from '../state-map'
import { createProjectCompoundController } from '../world/project-compounds'

const STATES = ['idle', 'running', 'blocked', 'failed', 'review', 'orchestration', 'resource_wait', 'completed', 'disconnected'] as const

export function demoSnapshot(state: string, revision: number, manifest: WorldManifestV2, delegated = false): LunarCitySnapshot {
  const anchor = manifest.destinations.garden

  const rows: LunarEntity[] = ['Archivist', 'Engineer'].map((profile,index) => {
    const identity = { kind:'session' as const, connectionId:'local-city-demo', profile, sessionId:`demo-${index}` }
    const sourceState = index ? 'idle' : state

    return { key:entityKey(identity), identity, observedAt:revision, sourceState, projectId: `/demo/${profile.toLowerCase()}`,
      ...mapObservedState({source:'session',status:sourceState,fresh:state!=='disconnected'}),
      position:{x:anchor.x+index*1.7,y:anchor.y,z:anchor.z},
      presentation:{configuredTitle:index?'Engineering partner':'Archive specialist',groups:[],metadata:{source:'local-demo',state:'fresh',observedAt:revision},placement:{lodHint:0,overflow:false,slot:index}} }
  })

  if (delegated) {
    const parent = rows[0], identity = { kind:'subagent' as const, connectionId:'local-city-demo', profile:'Archivist', sessionId:'demo-0', subagentId:'demo-child' }
    rows.push({...parent, key:entityKey(identity), identity, position:{...parent.position!, x:parent.position!.x+1.7}, presentation:undefined})
  }

  return {revision,observedAt:revision,entities:new Map(rows.map(e=>[e.key,e])),sources:[{source:'Local demonstration',authority:state==='disconnected'?'stale':'authoritative',observedAt:revision}]}
}

function SystemsDemo({ world, manifest }: { world:LunarCityWorldHandle; manifest:WorldManifestV2 }) {
  const [enabled,setEnabled]=useState(false),[state,setState]=useState('idle'),[revision,setRevision]=useState(1),[delegated,setDelegated]=useState(false)
  const [reduced,setReduced]=useState(window.matchMedia('(prefers-reduced-motion: reduce)').matches),[quality,setQuality]=useState('balanced')
  useEffect(()=>{const motion=(e:Event)=>setReduced((e as CustomEvent<boolean>).detail), qualityChange=(e:Event)=>setQuality((e as CustomEvent<string>).detail);window.addEventListener('lunar-review-motion',motion);window.addEventListener('lunar-review-quality',qualityChange);

return()=>{window.removeEventListener('lunar-review-motion',motion);window.removeEventListener('lunar-review-quality',qualityChange)}},[])
  const [visible,setVisible]=useState(!document.hidden)
  useEffect(()=>{const changed=()=>setVisible(!document.hidden);document.addEventListener('visibilitychange',changed);

return()=>document.removeEventListener('visibilitychange',changed)},[])
  const [selected,setSelected]=useState<EntityKey>()
  const worldRef=useRef(world)
  const snapshot=demoSnapshot(state,revision,manifest,delegated)
  useEffect(()=>{
    if (!enabled) {return}
    const next=demoSnapshot(state,revision,manifest,delegated);world.applySnapshot(next)
    const entity=[...next.entities.values()][0];setSelected(entity.key)
    world.dispatchCamera({kind:'focus',entityKey:entity.key,follow:true})
  },[enabled,state,revision,world,manifest,delegated])
  const compounds=useMemo(()=>createProjectCompoundController(manifest.projectSlots),[manifest])
  const report=compounds.update(snapshot)
  const presentation=useCallback((key:EntityKey)=>world.getWorkerPresentation?.(key),[world])

  const project=useCallback((point:{x:number;y:number;z:number})=>{const result=world.projectWorldPoint?.(point);

return result?.visible?result:undefined},[world])

  const select=(entity:LunarEntity)=>{setSelected(entity.key);world.dispatchCamera({kind:'focus',entityKey:entity.key,follow:true})}

  return <aside aria-label="Local systems demonstration" className="lunar-city-systems-demo">
    <button onClick={()=>{if(enabled){world.applySnapshot({revision:revision+1,observedAt:revision+1,entities:new Map(),sources:[]});}setEnabled(!enabled)}}>{enabled?'Close systems demo':'Local systems demo'}</button>
    {enabled?<><h2>Local systems demo</h2><p>Simulated observations for review only. No gateway calls, assignments or messages.</p>
      <label>First worker observation <select aria-label="Demo observation" onChange={event=>{setState(event.target.value);setRevision(n=>n+1)}} value={state}>{STATES.map(s=><option key={s}>{s}</option>)}</select></label>
      <label><input checked={delegated} onChange={event=>{setDelegated(event.target.checked);setRevision(n=>n+1)}} type="checkbox"/> Include an observed child worker</label>
      <ProjectCompoundsPanel onSelectEntity={select} report={report} snapshot={snapshot}/>
      {createPortal(<DelegationLinks enabled={quality!=='efficient'} onSelectEntity={select} presentation={presentation} project={project} reducedMotion={reduced} snapshot={snapshot} visible={visible}/>,document.body)}
      <AlertsPanel onSelectEntity={select} snapshot={snapshot}/>
      <WorkerLife demo onSelect={select} selectedEntityKey={selected} snapshot={snapshot} worldRef={worldRef}/>
    </>:null}
  </aside>
}

export function mountSystemsDemo(world:LunarCityWorldHandle,manifest:WorldManifestV2) {
  const host=document.createElement('div');document.body.append(host);const root=createRoot(host)
  root.render(<SystemsDemo manifest={manifest} world={world}/>)

  return ()=>{root.unmount();host.remove()}
}
