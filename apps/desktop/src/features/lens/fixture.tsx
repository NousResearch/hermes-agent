import '@/styles.css'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'

import { ackComposerInsert, onComposerInsertRequest } from '@/app/chat/composer/focus'
import { PreviewPane } from '@/app/chat/right-rail/preview-pane'
import { setLensScope } from '@/features/lens/store'
import { I18nProvider } from '@/i18n/context'

const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
setLensScope('lens-integration')

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
          <textarea aria-label="Chat draft" readOnly value={draft} />
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
