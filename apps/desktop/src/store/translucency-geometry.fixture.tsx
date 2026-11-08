import { createRoot } from 'react-dom/client'

import { group, split } from '@/components/pane-shell/tree/model'
import { LayoutTreeRoot } from '@/components/pane-shell/tree/renderer'
import { $hiddenTreePanes, $layoutTree, $narrowViewport, mirrorLayoutTree } from '@/components/pane-shell/tree/store'
import { Sidebar, SidebarProvider } from '@/components/ui/sidebar'
import { registry } from '@/contrib/registry'
import { setPaneWidthOverride } from '@/store/panes'
import { $translucency, $translucencyBook, setTranslucencyMode, setTranslucencyScope } from '@/store/translucency'

registry.register({
  area: 'panes',
  id: 'sessions',
  title: 'Sessions',
  data: { placement: 'left', width: '160px', lifecycleKeepAlive: true },
  render: () => (
    <Sidebar className="bg-(--ui-sidebar-surface-background)" collapsible="none" style={{ width: '100%' }}>
      Sessions
    </Sidebar>
  )
})
registry.register({
  area: 'panes',
  id: 'chat',
  title: 'Chat',
  data: { placement: 'main' },
  render: () => <div data-chat-surface>Chat</div>
})
registry.register({
  area: 'panes',
  id: 'files',
  title: 'Files',
  data: { placement: 'right', width: '160px' },
  render: () => <div>Files</div>
})

const tree = () => split('row', [group(['sessions'], { id: 'sessions-zone' }), group(['chat']), group(['files'])])
$layoutTree.set(null)
$narrowViewport.set(false)
$translucencyBook.set({
  mode: 'glass',
  base: { intensity: 29, fade: 0, material: 'titlebar', scope: 'sidebar' },
  light: {},
  dark: {}
})
createRoot(document.getElementById('fixture')!).render(
  <SidebarProvider>
    <LayoutTreeRoot />
  </SidebarProvider>
)

const api = {
  mount: () => $layoutTree.set(tree()),
  mirror: mirrorLayoutTree,
  reorder: () => $layoutTree.set(split('row', [group(['files']), group(['sessions']), group(['chat'])])),
  hide: (hidden: boolean) => $hiddenTreePanes.set(new Set(hidden ? ['sessions'] : [])),
  width: (width: number) => setPaneWidthOverride('sessions', width),
  scope: setTranslucencyScope,
  mode: setTranslucencyMode,
  state: () => $translucency.get(),
  book: () => $translucencyBook.get(),
  settle: async () => {
    for (let i = 0; i < 4; i += 1) {
      await new Promise(requestAnimationFrame)
    }
  }
}

Object.assign(window, { sidebarGlass: api })
