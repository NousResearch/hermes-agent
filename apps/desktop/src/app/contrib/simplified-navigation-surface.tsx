import { useStore } from '@nanostores/react'
import { type ComponentProps, type CSSProperties, useLayoutEffect, useState } from 'react'
import { useLocation } from 'react-router'

import { $hiddenStripTabs, $hiddenTreePanes, setTreePaneHidden } from '@/components/pane-shell/tree/store'
import { setPaneRouteCollapsed } from '@/components/pane-shell/tree/store-route-collapse'
import { $focusedSessionIsTile } from '@/store/session-focus'
import { usePortalShell } from '@/store/shell-variant'

import { SimplifiedGlobalNav } from '../chat/sidebar/simplified-global-nav'
import { appViewForPath, isOverlayView } from '../routes'
import { useTitlebarHeight } from '../shell/use-touch-titlebar'

type Props = ComponentProps<typeof SimplifiedGlobalNav>

export function SimplifiedNavigationSurface(props: Props) {
  const enabled = usePortalShell()

  return enabled ? <NavigationWithChatSidebar {...props} /> : null
}

function NavigationWithChatSidebar(props: Props) {
  const { pathname } = useLocation()
  const focusedChat = useStore($focusedSessionIsTile)
  const hiddenTabs = useStore($hiddenStripTabs)
  const hiddenPanes = useStore($hiddenTreePanes)
  const titlebarHeight = useTitlebarHeight()
  const view = appViewForPath(pathname)
  // Chats show the sidebar; workspace pages take the whole stage. A route overlay covers whichever
  // layout it opened over without changing it (route-cover.tsx), so closing it reveals that layout
  // instead of remounting it, and moving between overlays and pages never mounts a covered sidebar.
  const pageLayout = !focusedChat && view !== 'chat'
  const covering = isOverlayView(view)
  const [coveredPageLayout, setCoveredPageLayout] = useState(pageLayout)

  if (!covering && coveredPageLayout !== pageLayout) {
    setCoveredPageLayout(pageLayout)
  }

  const userHidden = hiddenTabs.has('sessions')

  // Route visibility is transient: it collapses the sidebar without unmounting it (so returning to
  // a chat is a flip, not a rebuild) and never touches the user's own hide preference.
  useLayoutEffect(() => {
    setPaneRouteCollapsed('sessions', covering ? coveredPageLayout : pageLayout)
  }, [covering, coveredPageLayout, pageLayout])
  useLayoutEffect(() => {
    setTreePaneHidden('sessions', userHidden)
  }, [userHidden, hiddenPanes])
  useLayoutEffect(() => () => setPaneRouteCollapsed('sessions', false), [])

  return (
    <div className="flex shrink-0" style={{ '--titlebar-height': `${titlebarHeight}px` } as CSSProperties}>
      <SimplifiedGlobalNav {...props} />
    </div>
  )
}
