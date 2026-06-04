import { useCallback, useEffect, useMemo, useRef } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'

import { type CommandCenterSection } from '@/app/command-center'
import {
  AGENTS_ROUTE,
  appViewForPath,
  COMMAND_CENTER_ROUTE,
  isOverlayView,
  NEW_CHAT_ROUTE,
  SETTINGS_ROUTE
} from '@/app/routes'

const SECTIONS = ['sessions', 'system', 'usage'] as const

export function useOverlayRouting() {
  const location = useLocation()
  const navigate = useNavigate()

  const currentView = appViewForPath(location.pathname)
  const settingsOpen = currentView === 'settings'
  const commandCenterOpen = currentView === 'command-center'
  const agentsOpen = currentView === 'agents'
  const cronOpen = currentView === 'cron'
  const profilesOpen = currentView === 'profiles'
  const chatOpen = currentView === 'chat'
  const overlayOpen = isOverlayView(currentView)

  // Overlay routes (settings/command-center/agents) stash the underlying path
  // so closing them returns there instead of bouncing to /.
  const returnPathRef = useRef(NEW_CHAT_ROUTE)

  useEffect(() => {
    if (!overlayOpen) {
      returnPathRef.current = `${location.pathname}${location.search}${location.hash}`
    }
  }, [location.hash, location.pathname, location.search, overlayOpen])

  const commandCenterInitialSection = useMemo<CommandCenterSection | undefined>(
    () => SECTIONS.find(value => value === new URLSearchParams(location.search).get('section')),
    [location.search]
  )

  const openOverlayRoute = useCallback(
    (path: string) => {
      if (!overlayOpen) {
        returnPathRef.current = `${location.pathname}${location.search}${location.hash}`
      }

      navigate(path)
    },
    [location.hash, location.pathname, location.search, navigate, overlayOpen]
  )

  const openCommandCenterSection = useCallback(
    (section: CommandCenterSection) => openOverlayRoute(`${COMMAND_CENTER_ROUTE}?section=${section}`),
    [openOverlayRoute]
  )

  const closeOverlayToPreviousRoute = useCallback(
    () => navigate(returnPathRef.current || NEW_CHAT_ROUTE, { replace: true }),
    [navigate]
  )

  const toggleCommandCenter = useCallback(() => {
    if (commandCenterOpen) {
      closeOverlayToPreviousRoute()
    } else {
      openOverlayRoute(COMMAND_CENTER_ROUTE)
    }
  }, [closeOverlayToPreviousRoute, commandCenterOpen, openOverlayRoute])

  const openAgents = useCallback(() => openOverlayRoute(AGENTS_ROUTE), [openOverlayRoute])
  const openSettings = useCallback(() => openOverlayRoute(SETTINGS_ROUTE), [openOverlayRoute])

  const openSettingsTab = useCallback(
    (tab: string) => openOverlayRoute(`${SETTINGS_ROUTE}?tab=${tab}`),
    [openOverlayRoute]
  )

  return {
    agentsOpen,
    chatOpen,
    closeOverlayToPreviousRoute,
    commandCenterInitialSection,
    commandCenterOpen,
    cronOpen,
    currentView,
    openAgents,
    openCommandCenterSection,
    openSettings,
    openSettingsTab,
    profilesOpen,
    settingsOpen,
    toggleCommandCenter
  }
}
