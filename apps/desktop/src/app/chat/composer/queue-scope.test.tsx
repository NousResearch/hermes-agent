import { AssistantRuntimeProvider, type ThreadMessageLike, useExternalStoreRuntime } from '@assistant-ui/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen } from '@testing-library/react'
import { useState } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { I18nProvider } from '@/i18n/context'
import { en } from '@/i18n/en'
import { mainComposerScope } from '@/store/composer'
import { $queuedPromptsBySession, enqueueQueuedPrompt } from '@/store/composer-queue'
import { $sessionStates } from '@/store/session-states'

import type { ChatBarState } from './types'

import { ChatBar } from './index'

afterEach(() => {
  cleanup()
  $queuedPromptsBySession.set({})
  $sessionStates.set({})
  mainComposerScope.clear()
  localStorage.clear()
})

const chatBarState: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { enabled: false, active: false }
}

function Harness({ queueSessionKey }: { queueSessionKey: null | string }) {
  const runtime = useExternalStoreRuntime({
    convertMessage: (message: ThreadMessageLike) => message,
    isRunning: true,
    messages: [] as ThreadMessageLike[],
    onNew: async () => {}
  })

  const [client] = useState(() => new QueryClient({ defaultOptions: { queries: { retry: false } } }))

  return (
    <QueryClientProvider client={client}>
      <AssistantRuntimeProvider runtime={runtime}>
        <MemoryRouter>
          <I18nProvider configClient={null} initialLocale="en">
            <ChatBar
              busy
              disabled={false}
              freshDraftKey="__new__:fresh"
              gateway={null}
              onCancel={vi.fn()}
              onSubmit={vi.fn(async () => true)}
              queueSessionKey={queueSessionKey}
              // The runtime of the chat just left is still active for a frame after the switch.
              sessionId="session-a"
              state={chatBarState}
            />
          </I18nProvider>
        </MemoryRouter>
      </AssistantRuntimeProvider>
    </QueryClientProvider>
  )
}

it('a new chat does not show the queue of the session just left (#139190)', async () => {
  $sessionStates.set({ 'session-a': { storedSessionId: 'session-a' } as ClientSessionState })
  enqueueQueuedPrompt('session-a', { attachments: [], text: 'for A' })

  const view = render(<Harness queueSessionKey="session-a" />)
  expect(await screen.findByText(en.composer.queued(1))).toBeTruthy()

  view.rerender(<Harness queueSessionKey={null} />)
  expect(screen.queryByText(en.composer.queued(1))).toBeNull()
})
