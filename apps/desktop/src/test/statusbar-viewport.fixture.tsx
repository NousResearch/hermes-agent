import '@/styles.css'

import { createRoot } from 'react-dom/client'
import { MemoryRouter } from 'react-router'

import { ContextUsagePanel } from '@/app/shell/context-usage-panel'
import { StatusbarControls, type StatusbarItem } from '@/app/shell/statusbar-controls'
import { createBrowserZoom } from '@/lib/browser-zoom'
import { $statusbarHiddenIds } from '@/store/statusbar-prefs'

const categories = ['system', 'tools', 'memory', 'skills', 'conversation', 'thinking', 'images', 'other'].map(
  (id, index) => ({ color: 'teal', id, label: id, tokens: 1000 * (index + 1) })
)

const usage = {
  calls: 1,
  context_max: 272_000,
  context_percent: 47,
  context_used: 128_200,
  input: 0,
  output: 0,
  total: 0
}

const breakdown = { ...usage, categories, estimated_total: usage.context_used, model: 'test-model' }

const items: StatusbarItem[] = Array.from({ length: 18 }, (_, index) => ({
  id: `item-${index}`,
  label: `Item ${index + 1}`,
  toggleLabel: `Item ${index + 1}`,
  variant: 'action'
}))

// Keep many customizable items without overflowing the bar's trigger itself.
$statusbarHiddenIds.set(items.map(item => item.id))

items.push({
  id: 'context-usage',
  label: 'Context meter',
  lockedVisible: true,
  menuClassName: 'w-auto border-(--ui-stroke-secondary) p-0',
  menuContent: (
    <ContextUsagePanel
      breakdown={breakdown}
      compact={{
        busy: false,
        onRequest: () => {
          document.body.dataset.compacted = 'true'
        },
        pending: false,
        unavailable: false
      }}
      loading={false}
      usage={usage}
    />
  ),
  toggleLabel: 'Context meter',
  variant: 'menu'
})

const zoom = createBrowserZoom('/statusbar-viewport-test')
Object.assign(window, { setTestZoom: zoom.setPercent })

createRoot(document.getElementById('root')!).render(
  <MemoryRouter>
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ flex: 1 }} />
      <StatusbarControls items={items} />
    </div>
  </MemoryRouter>
)
