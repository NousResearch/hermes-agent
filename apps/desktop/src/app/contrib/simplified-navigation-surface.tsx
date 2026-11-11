import { useStore } from '@nanostores/react'
import { type ComponentProps, type CSSProperties, useLayoutEffect } from 'react'
import { useLocation } from 'react-router'

import { $hiddenStripTabs, $hiddenTreePanes, setTreePaneHidden } from '@/components/pane-shell/tree/store'
import { $focusedSessionIsTile } from '@/store/session-focus'
import { usePortalShell } from '@/store/shell-variant'

import { SimplifiedGlobalNav } from '../chat/sidebar/simplified-global-nav'
import { appViewForPath } from '../routes'
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
  const chat = focusedChat || appViewForPath(pathname) === 'chat'
  const hidden = !chat || hiddenTabs.has('sessions')

  // Route visibility is transient; never overwrite the user's sidebar or layout preference.
  useLayoutEffect(() => {
    setTreePaneHidden('sessions', hidden)
  }, [hidden, hiddenPanes])
  useLayoutEffect(() => () => setTreePaneHidden('sessions', $hiddenStripTabs.get().has('sessions')), [])

  return (
    <div className="flex shrink-0" style={{ '--titlebar-height': `${titlebarHeight}px` } as CSSProperties}>
      <SimplifiedGlobalNav {...props} />
    </div>
  )
}
