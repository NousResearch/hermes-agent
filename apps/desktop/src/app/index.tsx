import { useStore } from '@nanostores/react'
import { useCallback, useMemo } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { $activeConnectionId } from '@/store/connections'
import { requestGatewayForAgent } from '@/store/gateway'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { isAuxiliaryWindow } from '@/store/windows'

import { ContribController } from './contrib'
import type { JarvisShellView } from './jarvis/i18n'
import { JarvisOnboarding } from './jarvis/onboarding'
import {
  $jarvisOnboardingCompletedAt,
  jarvisOnboardingScopeKey,
  readJarvisOnboardingState,
  shouldShowJarvisOnboarding
} from './jarvis/onboarding-state'
import { JarvisShell } from './jarvis/shell'
import {
  AGENTS_ROUTE,
  ARTIFACTS_ROUTE,
  PROMPTS_ROUTE,
  COMMAND_CENTER_ROUTE,
  CONNECTIONS_ROUTE,
  CRON_ROUTE,
  MESSAGING_ROUTE,
  NEW_CHAT_ROUTE,
  PROFILES_ROUTE,
  routePathname,
  SETTINGS_ROUTE,
  SKILLS_ROUTE,
  STARMAP_ROUTE,
  WEBHOOKS_ROUTE
} from './routes'

const JARVIS_VIEW_TARGETS: Record<JarvisShellView, string> = {
  prompts: PROMPTS_ROUTE,
  jarvis: NEW_CHAT_ROUTE,
  tasks: CRON_ROUTE,
  agents: AGENTS_ROUTE,
  messaging: MESSAGING_ROUTE,
  webhooks: WEBHOOKS_ROUTE,
  artifacts: ARTIFACTS_ROUTE,
  memory: `${STARMAP_ROUTE}?view=list`,
  starmap: STARMAP_ROUTE,
  tools: SKILLS_ROUTE,
  connections: CONNECTIONS_ROUTE,
  insights: COMMAND_CENTER_ROUTE,
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

  if (path === STARMAP_ROUTE && params.get('view') === 'list') {
    return 'memory'
  }

  // One entry per page: every Capabilities tab (skills, toolsets, MCP) is "tools".
  const byPath: Partial<Record<string, JarvisShellView>> = {
    [AGENTS_ROUTE]: 'agents',
    [PROMPTS_ROUTE]: 'prompts',
    [ARTIFACTS_ROUTE]: 'artifacts',
    [COMMAND_CENTER_ROUTE]: 'insights',
    [CONNECTIONS_ROUTE]: 'connections',
    [MESSAGING_ROUTE]: 'messaging',
    [SKILLS_ROUTE]: 'tools',
    [STARMAP_ROUTE]: 'starmap',
    [WEBHOOKS_ROUTE]: 'webhooks'
  }

  const direct = byPath[path]

  if (direct) {
    return direct
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
  const activeConnectionId = useStore($activeConnectionId)
  const activeProfile = useStore($activeGatewayProfile)

  const onboardingScope = {
    connectionId: activeConnectionId ?? 'local',
    profile: normalizeProfileKey(activeProfile)
  }

  const onboardingScopeKey = jarvisOnboardingScopeKey(onboardingScope)
  const compositionMode = appCompositionMode({ auxiliary: isAuxiliaryWindow() })

  const isOnboardingScopeCurrent = useCallback((expected: { connectionId?: null | string; profile?: null | string }) => {
    const currentConnectionId = $activeConnectionId.get() ?? 'local'
    const currentProfile = normalizeProfileKey($activeGatewayProfile.get())

    return currentConnectionId === (expected.connectionId ?? 'local') && currentProfile === normalizeProfileKey(expected.profile)
  }, [])

  // Storage does not notify, so the wizard re-reads its scoped state whenever
  // it reports a change (finished or closed). Without this, closing the wizard
  // would leave it on screen until something else happened to re-render.

  const onboardingChangedAt = useStore($jarvisOnboardingCompletedAt)

  const onboardingState = useMemo(
    () =>
      readJarvisOnboardingState(undefined, {
        connectionId: $activeConnectionId.get() ?? 'local',
        profile: normalizeProfileKey($activeGatewayProfile.get())
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- onboardingChangedAt is the storage-change signal and onboardingScopeKey covers connection + profile; the scope object itself changes identity on every render
    [onboardingChangedAt, onboardingScopeKey]
  )

  const showOnboarding = compositionMode === 'product-shell' && shouldShowJarvisOnboarding(onboardingState)

  if (compositionMode === 'special-window') {
    return <ContribController />
  }

  return (
    <>
      <JarvisShell
        activeView={jarvisViewForLocation(location.pathname, location.search)}
        onViewChange={view => navigate(JARVIS_VIEW_TARGETS[view])}
      >
        <ContribController layoutMode="embedded" />
      </JarvisShell>
      {showOnboarding ? (
        <JarvisOnboarding
          isScopeCurrent={isOnboardingScopeCurrent}
          key={onboardingScopeKey}
          // Finishing setup is the person's own click: picking up the
          // connections they chose right there is a continuation, not a hijack.
          onComplete={() => {
            if (readJarvisOnboardingState(undefined, onboardingScope)?.selections?.connections?.length) {
              navigate(CONNECTIONS_ROUTE)
            }
          }}
          // Closing setup is a choice, not a dead end: the wizard is gone for
          // this profile, and the keys/models live in Settings from here on.
          onDismiss={() => navigate(`${SETTINGS_ROUTE}?tab=providers&pview=keys`)}
          requestGateway={(method, params) =>
            requestGatewayForAgent(onboardingScope.connectionId, onboardingScope.profile, method, params)
          }
          scope={onboardingScope}
        />
      ) : null}
    </>
  )
}

export default AppRoot
export { JarvisShell } from './jarvis/shell'
export { appCompositionMode, JARVIS_VIEW_TARGETS, jarvisViewForLocation }
