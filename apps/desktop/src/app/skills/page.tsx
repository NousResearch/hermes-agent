import type { ComponentProps } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { normalizeHermesOpenString } from '@/lib/hermes-open-target'
import { $selectedStoredSessionId } from '@/store/session'

import { appViewForPath, isOverlayView, NEW_CHAT_ROUTE, routePathname, sessionRoute } from '../routes'

import { SkillsView } from './index'

interface SkillsPageProps extends Pick<ComponentProps<typeof SkillsView>, 'setStatusbarItemGroup'> {}

function returnTarget(state: unknown): string | null {
  if (!state || typeof state !== 'object' || !('returnTo' in state) || typeof state.returnTo !== 'string') {
    return null
  }

  const pathname = routePathname(state.returnTo)

  if (!pathname.startsWith('/') || normalizeHermesOpenString(pathname) !== pathname) {
    return null
  }

  try {
    const view = appViewForPath(pathname)

    return view === 'skills' || isOverlayView(view) ? null : state.returnTo
  } catch {
    // A malformed percent-encoded session id is not a usable return route.
    return null
  }
}

/** The workspace page owns navigation; embedded capability views do not. */
export function SkillsPage(props: SkillsPageProps) {
  const navigate = useNavigate()
  const { state } = useLocation()

  const close = () => {
    const sessionId = $selectedStoredSessionId.get()
    navigate(returnTarget(state) ?? (sessionId ? sessionRoute(sessionId) : NEW_CHAT_ROUTE), { replace: true })
  }

  return <SkillsView {...props} onClose={close} />
}
