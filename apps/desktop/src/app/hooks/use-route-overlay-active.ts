import { useLocation } from 'react-router-dom'

import { appViewForPath, isOverlayView } from '@/app/routes'

/**
 * True while a full-screen route overlay (settings, agents, command-center, …)
 * is showing.
 *
 * A portaled Radix modal sits above the app shell, so it would cover such a
 * route. Any modal that sends the user to one (e.g. "set up image generation" →
 * `/settings`) can `if (useRouteOverlayActive()) return null` to *yield* the
 * screen — its open state lives in a store, so it stays open — and reappear,
 * re-running its mount effects (a free refresh), when the route overlay closes.
 */
export function useRouteOverlayActive(): boolean {
  const { pathname } = useLocation()

  return isOverlayView(appViewForPath(pathname))
}
