import '@/styles.css'
import { createRoot } from 'react-dom/client'
import { AssistantRuntimeProvider, ThreadPrimitive, useExternalStoreRuntime, type ThreadMessage } from '@assistant-ui/react'
import type { ReactNode } from 'react'
import type { SessionMessage } from '@/types/hermes'
import { QueryClientProvider } from '@tanstack/react-query'
import { queryClient } from '@/lib/query-client'
import { I18nProvider } from '@/i18n'
import { ThemeProvider } from '@/themes/context'
import { RootTooltipProvider } from '@/components/ui/tooltip'
import { SystemMessage } from '@/components/assistant-ui/thread/system-message'
import { toChatMessages } from '@/lib/chat-messages/hydration'
import { toChatMessages as baselineHydrate } from '@baseline-hydration'
import { toRuntimeMessage } from '@/lib/chat-runtime'
function Providers({children}: {children: ReactNode}) {
  return <QueryClientProvider client={queryClient}><I18nProvider><ThemeProvider><RootTooltipProvider>{children}</RootTooltipProvider></ThemeProvider></I18nProvider></QueryClientProvider>
}
function Delivery({messages}: {messages: ThreadMessage[]}) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({messages, isRunning:false, onNew:async()=>{}})
  return <AssistantRuntimeProvider runtime={runtime}><ThreadPrimitive.Root><ThreadPrimitive.Messages components={{Message: SystemMessage}}/></ThreadPrimitive.Root></AssistantRuntimeProvider>
}
Object.assign(window, {renderProducer(rows: SessionMessage[]) {
  const hydrated = (location.search.includes('baseline') ? baselineHydrate : toChatMessages)(rows)
  const runtime = hydrated.map(toRuntimeMessage)
  const mount = document.createElement('section'); mount.id = 'producer'; mount.style.cssText='padding:24px; max-width:960px; margin:auto'; document.body.append(mount)
  createRoot(mount).render(<Providers><Delivery messages={runtime}/></Providers>)
  return {hydrated,runtime}
}})
