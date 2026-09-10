import '@/styles.css'

import { createRoot } from 'react-dom/client'
import { MemoryRouter } from 'react-router'

import { ComposerStatusStack } from '@/app/chat/composer/status-stack'
import { RootTooltipProvider } from '@/components/ui/tooltip'
import { I18nProvider } from '@/i18n'
import { $todosBySession } from '@/store/todos'

// Input records only: the status grouping, rows, collapse and CSS are production code.
const count = Number(new URLSearchParams(location.search).get('tasks') ?? 20)
$todosBySession.set({
  'task-scroll-fixture': Array.from({ length: count }, (_, i) => ({
    id: `task-${i + 1}`,
    content: `Phase ${i + 1}: renderer geometry task ${'long description '.repeat(i % 3)}`,
    status: 'completed' as const
  }))
})

createRoot(document.getElementById('root')!).render(
  <I18nProvider>
    <RootTooltipProvider>
      <MemoryRouter>
        <div style={{ position: 'absolute', bottom: 30, left: 200, width: 700 }}>
          <ComposerStatusStack queue={null} sessionId="task-scroll-fixture" />
          <div data-slot="fixture-composer" style={{ position: 'relative', height: 90, background: '#ddd' }}>
            Composer boundary — component fixture, no backend
          </div>
        </div>
      </MemoryRouter>
    </RootTooltipProvider>
  </I18nProvider>
)
