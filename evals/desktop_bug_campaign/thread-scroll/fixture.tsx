import './styles.css'
import React, {useState, startTransition} from 'react'
import {AssistantRuntimeProvider,useExternalStoreRuntime} from '@assistant-ui/react'
import {Thread} from './components/assistant-ui/thread'
import {setActiveProfile} from './store/profile'
import {rescopeConnectionScopedStores} from './lib/connection-scoped'
import {saveThreadScrollPosition} from './store/thread-scroll'
import {requestScrollToBottom} from './store/thread-scroll'
import {createRoot} from 'react-dom/client'
import {MemoryRouter} from 'react-router'
import {I18nProvider} from './i18n'
import {RootTooltipProvider} from './components/ui/tooltip'
const histories=Object.fromEntries(['a','b'].map(key=>[key,Array.from({length:key==='a'?200:12},(_,i)=>({id:`${key}-${i}`,role:i%2?'assistant' as const:'user' as const,content:[{type:'text' as const,text:i%2?`## Response ${i}\n\n${'A paragraph of persisted transcript fixture content. '.repeat(15)}\n\n\`\`\`python\nprint(\"example\")\n\`\`\``:`Question ${i}`}]}))]))
function TranscriptProbe({id='first'}) {
 const [key,setKey]=useState('a'); const [loaded,setLoaded]=useState(true)
 const [epoch,setEpoch]=useState(0); const [partial,setPartial]=useState(false); const [hydrating,setHydrating]=useState(false)
 const [running,setRunning]=useState(false); const [growth,setGrowth]=useState(false)
 Object.assign(window,{scrollProbe:{run:()=>setRunning(true),grow:()=>setGrowth(true),scope:(profile: string,remote: string | null)=>{setActiveProfile(profile);rescopeConnectionScopedStores(remote?{mode:'remote',baseUrl:remote,profile}:null);setEpoch(e=>e+1)},remount:()=>setEpoch(e=>e+1),partial:()=>{saveThreadScrollPosition('hydrating',{kind:'offset',fromBottom:9000});setHydrating(true);setPartial(true);setEpoch(e=>e+1)},release:()=>startTransition(()=>setPartial(false))}})
 const messages=loaded?((partial || (!hydrating && location.search.includes('ownership')))?histories[key].slice(-12):histories[key]):[]
 const runtime=useExternalStoreRuntime({messages:growth?messages.map((m,i)=>i===messages.length-1?{...m,content:[{type:'text' as const,text:m.content[0].text+'\n\n'+('Additional streamed paragraph.\n\n'.repeat(30))}]}:m):messages,isRunning:running,onNew:async()=>{},convertMessage:m=>m})
 return <section data-probe={id}><button id={`switch-${id}`} onClick={()=>setKey(k=>k==='a'?'b':'a')}>Switch {key}</button><button id={`reload-${id}`} onClick={()=>{setLoaded(false);setTimeout(()=>setLoaded(true),500)}}>Reload</button><button id={`jump-${id}`} onClick={()=>requestScrollToBottom(id)}>Jump</button><div style={{height:700,width:850}}><AssistantRuntimeProvider runtime={runtime}><Thread key={epoch} sessionKey={hydrating?'hydrating':key} sessionId={id}/></AssistantRuntimeProvider></div></section>
}
createRoot(document.getElementById('root')!).render(<I18nProvider><RootTooltipProvider><MemoryRouter><div style={{display:'flex'}}><TranscriptProbe/>{location.search.includes('twins')&&<TranscriptProbe id="second"/>}</div></MemoryRouter></RootTooltipProvider></I18nProvider>)
