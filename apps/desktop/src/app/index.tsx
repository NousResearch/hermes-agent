import { useLocation, useNavigate } from 'react-router'

import { isAuxiliaryWindow } from '@/store/windows'

import { ContribController } from './contrib'
import type { JarvisShellView } from './jarvis/i18n'
import { JarvisShell } from './jarvis/shell'
import {
  CRON_ROUTE,
  NEW_CHAT_ROUTE,
  PROFILES_ROUTE,
  routePathname,
  SETTINGS_ROUTE,
  SKILLS_ROUTE
} from './routes'

const JARVIS_VIEW_TARGETS: Record<JarvisShellView, string> = {
  jarvis: NEW_CHAT_ROUTE,
  tasks: CRON_ROUTE,
  memory: `${SETTINGS_ROUTE}?tab=config:memory`,
  tools: `${SKILLS_ROUTE}?tab=toolsets`,
  settings: SETTINGS_ROUTE,
  profile: PROFILES_ROUTE
}

function jarvisViewForLocation(pathname: string, search: string): JarvisShellView {
  const path = routePathname(pathname)
  const params = new URLSearchParams(search)

  if (path === CRON_ROUTE) {
    return 'tasks'
  }

  if (path === PROFILES_ROUTE) {
    return 'profile'
  }

  if (path === SKILLS_ROUTE && params.get('tab') === 'toolsets') {
    return 'tools'
  }

  if (path === SETTINGS_ROUTE) {
    return params.get('tab') === 'config:memory' ? 'memory' : 'settings'
  }

  return 'jarvis'
}

export type AppCompositionMode = 'product-shell' | 'special-window'

export interface AppWindowModeFlags {
  auxiliary: boolean
}

function appCompositionMode({ auxiliary }: AppWindowModeFlags): AppCompositionMode {
  return auxiliary ? 'special-window' : 'product-shell'
}

// The app root remains the contribution-driven runtime: panes, titlebar/statusbar
// items, keybinds, palette commands, routes, themes, gateway boot, sessions, and
// streams all stay inside ContribController. JarvisShell is only the product
// chrome that navigates to those existing surfaces.
function AppRoot() {
  const location = useLocation()
  const navigate = useNavigate()
  const compositionMode = appCompositionMode({ auxiliary: isAuxiliaryWindow() })

  if (compositionMode === 'special-window') {
    return <ContribController />
  }

  return (
    <JarvisShell
      activeView={jarvisViewForLocation(location.pathname, location.search)}
      onViewChange={view => navigate(JARVIS_VIEW_TARGETS[view])}
    >
      <ContribController layoutMode="embedded" />
    </JarvisShell>
  )
}

export default AppRoot
export { JarvisShell } from './jarvis/shell'
export { appCompositionMode, JARVIS_VIEW_TARGETS, jarvisViewForLocation }
