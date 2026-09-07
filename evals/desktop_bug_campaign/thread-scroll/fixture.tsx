import './styles.css'
import React, {useState} from 'react'
import {AssistantRuntimeProvider,useExternalStoreRuntime} from '@assistant-ui/react'
import {Thread} from './components/assistant-ui/thread'
import {requestScrollToBottom} from './store/thread-scroll'
import {createRoot} from 'react-dom/client'
import {MemoryRouter} from 'react-router'
import {I18nProvider} from './i18n'
import {RootTooltipProvider} from './components/ui/tooltip'
const histories=Object.fromEntries(['a','b'].map(key=>[key,Array.from({length:key==='a'?200:12},(_,i)=>({id:`${key}-${i}`,role:i%2?'assistant' as const:'user' as const,content:[{type:'text' as const,text:i%2?`## Response ${i}\n\n${'A paragraph of persisted transcript fixture content. '.repeat(15)}\n\n\`\`\`python\nprint(\"example\")\n\`\`\``:`Question ${i}`}]}))]))
function TranscriptProbe({id='first'}) {
 const [key,setKey]=useState('a'); const [loaded,setLoaded]=useState(true)
 const runtime=useExternalStoreRuntime({messages:loaded?histories[key]:[],isRunning:false,onNew:async()=>{},convertMessage:m=>m})
 return <section data-probe={id}><button id={`switch-${id}`} onClick={()=>setKey(k=>k==='a'?'b':'a')}>Switch {key}</button><button id={`reload-${id}`} onClick={()=>{setLoaded(false);setTimeout(()=>setLoaded(true),500)}}>Reload</button><button id={`jump-${id}`} onClick={()=>requestScrollToBottom(id)}>Jump</button><div style={{height:700,width:850}}><AssistantRuntimeProvider runtime={runtime}><Thread sessionKey={key} sessionId={id}/></AssistantRuntimeProvider></div></section>
}
createRoot(document.getElementById('root')!).render(<I18nProvider><RootTooltipProvider><MemoryRouter><div style={{display:'flex'}}><TranscriptProbe/>{location.search.includes('twins')&&<TranscriptProbe id="second"/>}</div></MemoryRouter></RootTooltipProvider></I18nProvider>)
