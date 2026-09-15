import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useRef } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { type CommandCenterSection } from '@/app/command-center'
import {
  AGENTS_ROUTE,
  type AppView,
  appViewForPath,
  COMMAND_CENTER_ROUTE,
  isOverlayView,
  NEW_CHAT_ROUTE,
  routePathname,
  STARMAP_ROUTE
} from '@/app/routes'
import { $activeConnectionId } from '@/store/connections'
import { $activeGatewayProfile } from '@/store/profile'

const SECTIONS = ['sessions', 'system', 'usage'] as const

export function useOverlayRouting() {
  const location = useLocation()
  const navigate = useNavigate()
  const connectionId = useStore($activeConnectionId)
  const profile = useStore($activeGatewayProfile)
  const scope = `${connectionId ?? ''}\0${profile}`

  const currentView = appViewForPath(location.pathname)
  const settingsOpen = currentView === 'settings'
  const commandCenterOpen = currentView === 'command-center'
  const agentsOpen = currentView === 'agents'
  const starmapOpen = currentView === 'starmap'
  const cronOpen = currentView === 'cron'
  const profilesOpen = currentView === 'profiles'
  const webhooksOpen = currentView === 'webhooks'
  const chatOpen = currentView === 'chat'
  const overlayOpen = isOverlayView(currentView)

  // Keep the underlying route while an overlay is open. Contributed pages
  // also retain their caller, so closing one doesn't switch chats or bots.
  const returns = useRef({ scope, paths: [] as Array<{ path: string; view: AppView }> })

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    if (returns.current.scope !== scope) {
      returns.current = { scope, paths: [] }
    }

    if (overlayOpen) {
      return
    }

    const entry = { path: `${location.pathname}${location.search}${location.hash}`, view: currentView }
    const paths = returns.current.paths
    const index = paths.findIndex(item => routePathname(item.path) === location.pathname)

    if (currentView !== 'extension') {
      returns.current.paths = [entry]
    } else if (index >= 0) {
      // Returning from another page, or changing only a query: don't create
      // a loop back to the page we just closed.
      paths.splice(index, paths.length - index, entry)
    } else {
      paths.push(entry)
    }
  }, [currentView, location.hash, location.pathname, location.search, overlayOpen, scope])

  const commandCenterInitialSection = useMemo<CommandCenterSection | undefined>(
    () => SECTIONS.find(value => value === new URLSearchParams(location.search).get('section')),
    [location.search]
  )

  const openCommandCenterSection = useCallback(
    (section: CommandCenterSection) => navigate(`${COMMAND_CENTER_ROUTE}?section=${section}`),
    [navigate]
  )

  const resetOverlayReturnRoute = useCallback(() => {
    returns.current.paths = []
  }, [])

  const closeToReturnRoute = useCallback(
    (closePage: boolean) => {
      const currentScope = `${$activeConnectionId.get() ?? ''}\0${$activeGatewayProfile.get()}`
      const paths = returns.current.scope === currentScope ? returns.current.paths : []

      if (closePage) {
        paths.pop()
      }

      // A plugin may have unloaded since it supplied the return route. Don't
      // reinterpret its old path as a session id.
      while (paths.length && appViewForPath(paths.at(-1)!.path) !== paths.at(-1)!.view) {
        paths.pop()
      }

      void navigate(paths.at(-1)?.path ?? NEW_CHAT_ROUTE, { replace: true })
    },
    [navigate]
  )

  const closeOverlayToPreviousRoute = useCallback(() => closeToReturnRoute(false), [closeToReturnRoute])
  const closeContributedRoute = useCallback(() => closeToReturnRoute(true), [closeToReturnRoute])

  const toggleCommandCenter = useCallback(() => {
    if (commandCenterOpen) {
      closeOverlayToPreviousRoute()
    } else {
      navigate(COMMAND_CENTER_ROUTE)
    }
  }, [closeOverlayToPreviousRoute, commandCenterOpen, navigate])

  const openAgents = useCallback(() => navigate(AGENTS_ROUTE), [navigate])
  const openStarmap = useCallback(() => navigate(STARMAP_ROUTE), [navigate])

  return {
    agentsOpen,
    chatOpen,
    closeContributedRoute,
    closeOverlayToPreviousRoute,
    commandCenterInitialSection,
    commandCenterOpen,
    cronOpen,
    currentView,
    openAgents,
    openCommandCenterSection,
    openStarmap,
    profilesOpen,
    resetOverlayReturnRoute,
    settingsOpen,
    starmapOpen,
    toggleCommandCenter,
    webhooksOpen
  }
}
