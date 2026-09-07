import { createRoot } from 'react-dom/client'
import { AssistantRuntimeProvider, ThreadPrimitive, useExternalStoreRuntime, type ThreadMessage } from '@assistant-ui/react'
import type { ReactNode } from 'react'
import type { SessionMessage } from '@/types/hermes'
import { QueryClientProvider } from '@tanstack/react-query'
import { queryClient } from '@/lib/query-client'
import { I18nProvider } from '@/i18n'
import { ThemeProvider } from '@/themes/context'
import { RootTooltipProvider } from '@/components/ui/tooltip'
import { MarkdownTextContent } from '@/components/assistant-ui/markdown-text'
import { SystemMessage } from '@/components/assistant-ui/thread/system-message'
import { preprocessMarkdown } from '@/lib/markdown-preprocess'
import { toChatMessages } from '@/lib/chat-messages/hydration'
import { toRuntimeMessage } from '@/lib/chat-runtime'
import { assistantTextPart, renderMediaTags } from '@/lib/chat-messages/parts'
import { stripGeneratedImageEchoes } from '@/lib/generated-images'
import { extractEmbeddedImages } from '@/lib/embedded-images'
import { stripPreviewTargets } from '@/lib/preview-targets'
const hard = 'First line  \nSecond line'
const soft = 'Soft first\nSoft second'
const code = '```python\nvalue = 1  \nvalue = 2 \n```'
const corpus = hard + '\n\n' + code
const image = 'data:image/png;base64,' + 'A'.repeat(64)
const assistant = assistantTextPart(corpus)
const ingress = {
  assistant: assistant.type === 'text' ? assistant.text : '',
  media: renderMediaTags(corpus + '\n\nMEDIA: /tmp/image.png'),
  preview: stripPreviewTargets(corpus + '\n\n[Preview: x](#preview:test)'),
  embedded: extractEmbeddedImages(corpus + '\n\n' + image).cleanedText,
  generated: stripGeneratedImageEchoes(corpus + '\n\n![result](/tmp/image.png)', ['/tmp/image.png'])
}
Object.assign(window, {probeInputs:{hard,soft,code}, probeOutputs:{hard:preprocessMarkdown(hard),soft:preprocessMarkdown(soft),code:preprocessMarkdown(code)}, ingress})
function Providers({children}: {children: ReactNode}) {
  return <QueryClientProvider client={queryClient}><I18nProvider><ThemeProvider><RootTooltipProvider>{children}</RootTooltipProvider></ThemeProvider></I18nProvider></QueryClientProvider>
}
function Delivery({messages}: {messages: ThreadMessage[]}) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({messages, isRunning:false, onNew:async()=>{}})
  return <AssistantRuntimeProvider runtime={runtime}><ThreadPrimitive.Root><ThreadPrimitive.Messages components={{Message: SystemMessage}}/></ThreadPrimitive.Root></AssistantRuntimeProvider>
}
Object.assign(window, {renderProducer(rows: SessionMessage[]) {
  const hydrated = toChatMessages(rows)
  const runtime = hydrated.map(toRuntimeMessage)
  const mount = document.createElement('section'); mount.id = 'producer'; document.body.append(mount)
  createRoot(mount).render(<Providers><Delivery messages={runtime}/></Providers>)
  return {hydrated,runtime}
}})
createRoot(document.getElementById('root')!).render(<Providers><main style={{padding:40}}><h1>Production Markdown renderer probe</h1>{Object.entries({hard,soft,code,...ingress}).map(([id,text])=><section id={id} key={id}><h2>{id}</h2><MarkdownTextContent text={text} isRunning={false}/></section>)}</main></Providers>)
