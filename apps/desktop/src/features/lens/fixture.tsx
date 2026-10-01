import '@/styles.css'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'

import { ackComposerInsert, onComposerInsertRequest } from '@/app/chat/composer/focus'
import { PreviewPane } from '@/app/chat/right-rail/preview-pane'
import { I18nProvider } from '@/i18n/context'
import { $activeSessionId, $selectedStoredSessionId, setSessionOwnerHint } from '@/store/session'

const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })

function selectConnection(connectionId: string) {
  const id = 'lens-' + connectionId
  setSessionOwnerHint(id, { connectionId, profile: 'research' })
  $selectedStoredSessionId.set(id)
  $activeSessionId.set(id)
}

selectConnection('fixture-a')

function Fixture() {
  const [draft, setDraft] = useState('')
  useEffect(
    () =>
      onComposerInsertRequest(detail => {
        setDraft(detail.text)
        ackComposerInsert(detail.token, true)
      }),
    []
  )

  return (
    <QueryClientProvider client={queryClient}>
      <I18nProvider configClient={null} initialLocale="en">
        <div style={{ height: '100vh', display: 'grid', gridTemplateColumns: '320px 1fr' }}>
          <div className="flex flex-col gap-2">
            <button onClick={() => selectConnection('fixture-a')}>Connection A</button>
            <button onClick={() => selectConnection('fixture-b')}>Connection B</button>
            <textarea aria-label="Chat draft" className="flex-1" readOnly value={draft} />
          </div>
          <PreviewPane
            embedded
            tabId="lens-test"
            target={{
              kind: 'url',
              label: 'Field Notes',
              source: 'http://127.0.0.1:5176/e2e/lens/source.html',
              url: 'http://127.0.0.1:5176/e2e/lens/source.html'
            }}
          />
        </div>
      </I18nProvider>
    </QueryClientProvider>
  )
}

createRoot(document.getElementById('root')!).render(<Fixture />)
