import '@/styles.css'

import {
  AssistantRuntimeProvider,
  MessagePrimitive,
  type ThreadMessage,
  useExternalStoreRuntime
} from '@assistant-ui/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useState } from 'react'
import { flushSync } from 'react-dom'
import { createRoot } from 'react-dom/client'
import { MemoryRouter } from 'react-router'

import { Thread } from '@/components/assistant-ui/thread'
import { ThreadMessageList } from '@/components/assistant-ui/thread/list'
import { StickyHumanMessageContainer, USER_BUBBLE_BASE_CLASS } from '@/components/assistant-ui/thread/user-message'
import { PaneTab, PaneTabLabel, PaneTabStrip } from '@/components/ui/pane-tab'
import { RootTooltipProvider } from '@/components/ui/tooltip'
import { I18nProvider } from '@/i18n'

// Real primitives and stylesheet; only the sample labels and glass inputs are fixtures.
const root = document.documentElement
root.classList.add('dark')
root.setAttribute('data-hermes-glass', '')
root.setAttribute('data-hermes-glass-scope', 'window')
root.style.setProperty('--translucency-glass-keep', '40%')

const messages = Array.from({ length: 6 }, (_, turn) => [
  {
    id: `user-${turn}`,
    role: 'user',
    attachments: [],
    createdAt: new Date(0),
    content: [{ type: 'text', text: `Prompt ${turn}` }],
    metadata: { custom: {} }
  },
  {
    id: `assistant-${turn}`,
    role: 'assistant',
    createdAt: new Date(0),
    content: [{ type: 'text', text: `Reply ${turn}` }],
    status: { type: 'complete', reason: 'stop' },
    metadata: { custom: {}, unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [] }
  }
]).flat() as ThreadMessage[]

function User() {
  const [expanded, setExpanded] = useState(false)

  return (
    <StickyHumanMessageContainer
      attachments={
        <div data-testid="attachment" style={{ height: 40 }}>
          Attachment
        </div>
      }
    >
      <button
        className={USER_BUBBLE_BASE_CLASS}
        onClick={() => setExpanded(value => !value)}
        style={{ height: expanded ? 140 : 48 }}
      >
        A user prompt — click to change its height
      </button>
    </StickyHumanMessageContainer>
  )
}

function Assistant() {
  return (
    <MessagePrimitive.Root data-slot="aui_assistant-message-root">
      <div data-testid="reply-marker" style={{ height: 900, background: '#ff0000' }}>
        Scrolling reply
      </div>
    </MessagePrimitive.Root>
  )
}

const components = { UserMessage: User, AssistantMessage: Assistant }

function Transcript({ id }: { id: string }) {
  const [currentMessages, setMessages] = useState(() =>
    new URLSearchParams(location.search).has('preceding')
      ? ([{ ...messages[1], id: 'standalone' }, ...messages] as ThreadMessage[])
      : messages
  )

  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: currentMessages,
    isRunning: false,
    onNew: async () => {},
    onEdit: async () => {}
  })

  const realMessages = new URLSearchParams(location.search).has('real')

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div>
        <button
          data-testid={`${id}-append`}
          onClick={() =>
            flushSync(() =>
              setMessages(previous => [
                ...previous,
                { ...messages[1], id: `appended-${previous.length}` } as ThreadMessage
              ])
            )
          }
        >
          Append reply
        </button>
        <div data-testid={id} style={{ height: 430, width: 480 }}>
          {realMessages ? <Thread sessionKey={id} /> : <ThreadMessageList components={components} sessionKey={id} />}
        </div>
      </div>
    </AssistantRuntimeProvider>
  )
}

createRoot(document.getElementById('root')!).render(
  <QueryClientProvider client={new QueryClient()}>
    <I18nProvider>
      <RootTooltipProvider>
        <MemoryRouter>
          <div style={{ padding: 40 }}>
            <PaneTabStrip>
              <PaneTab
                active
                data-testid="active-tab"
                onClose={() => document.body.setAttribute('data-closed', 'true')}
              >
                <PaneTabLabel as="button" onClick={() => document.body.setAttribute('data-activated', 'true')}>
                  A long session title that reaches underneath the close button
                </PaneTabLabel>
              </PaneTab>
              <PaneTab data-testid="idle-tab" dirty onClose={() => {}}>
                <PaneTabLabel>Another long session title with unsaved changes</PaneTabLabel>
              </PaneTab>
              <PaneTab data-testid="fixed-tab">
                <PaneTabLabel>Sessions</PaneTabLabel>
              </PaneTab>
              <PaneTab data-testid="short-tab" onClose={() => {}}>
                <PaneTabLabel>X</PaneTabLabel>
              </PaneTab>
              <PaneTab data-testid="selected-tab" onClose={() => {}} selected>
                <PaneTabLabel>Selected</PaneTabLabel>
              </PaneTab>
            </PaneTabStrip>
            <div style={{ display: 'flex', gap: 30, marginTop: 40 }}>
              <Transcript id="first-transcript" />
              <Transcript id="second-transcript" />
            </div>
          </div>
        </MemoryRouter>
      </RootTooltipProvider>
    </I18nProvider>
  </QueryClientProvider>
)
