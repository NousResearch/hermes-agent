import '@/styles.css'

import { useStore } from '@nanostores/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createRoot } from 'react-dom/client'

import { watchPreviewTiles } from '@/app/chat/preview-tile'
import { actOnActivePreview } from '@/app/chat/right-rail/preview-act'
import { readActivePreview } from '@/app/chat/right-rail/preview-reader'
import { activePreviewScriptRunner } from '@/app/chat/right-rail/preview-script-runner'
import { queryVisible } from '@/components/pane-shell/pane-visibility'
import { TreeNode } from '@/components/pane-shell/tree/renderer/tree-node'
import { $layoutTree, activateTreePane, closeTabPane, setTreeGroupMinimized } from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'
import { $previewTabs, openPreview } from '@/store/preview'
import { $realProfilePromptMuted } from '@/store/real-profile-consent'

$realProfilePromptMuted.set(true)
watchPreviewTiles()
const url = new URL(location.href).searchParams.get('page')!
openPreview({ kind: 'url', label: 'Fixture', source: url, url })
const tabId = $previewTabs.get()[0].id
const paneId = `preview-tile:${tabId}`
registry.register({
  area: 'panes',
  id: 'workspace',
  title: 'Chat',
  data: { uncloseable: true },
  render: () => <input aria-label="Composer" />
})

for (const id of ['file-a', 'file-b', 'file-c']) {
  registry.register({ area: 'panes', id, title: id, render: () => <div>{id}</div> })
}

$layoutTree.set({
  type: 'split',
  id: 'root',
  orientation: 'row',
  weights: [1, 2],
  children: [
    { type: 'group', id: 'chat', panes: ['workspace'], active: 'workspace' },
    { type: 'group', id: 'browser', panes: [paneId, 'file-a', 'file-b', 'file-c'], active: paneId, tabStrip: 'always' }
  ]
})
const client = new QueryClient({ defaultOptions: { queries: { enabled: false, retry: false } } })

function Fixture() {
  const tree = useStore($layoutTree)

  return (
    <QueryClientProvider client={client}>
      <div style={{ display: 'flex', height: '100vh' }}>{tree && <TreeNode node={tree} root />}</div>
    </QueryClientProvider>
  )
}

createRoot(document.getElementById('root')!).render(<Fixture />)
Object.assign(window, {
  fixture: {
    run: (code: string) => activePreviewScriptRunner()?.(code),
    read: readActivePreview,
    drive: actOnActivePreview,
    hide: () => setTreeGroupMinimized('browser', true),
    restore: () => setTreeGroupMinimized('browser', false),
    close: () => closeTabPane(paneId),
    activate: activateTreePane,
    paneId,
    visibleGuest: () => Boolean(queryVisible('webview')),
    runnerPresent: () => Boolean(activePreviewScriptRunner())
  }
})
