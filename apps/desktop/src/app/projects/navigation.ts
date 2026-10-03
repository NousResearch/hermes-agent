import { syncWorkspaceRoute } from '../routes'

import { projectOverviewRoute } from './model'

/**
 * Open the Projects cockpit on `projectId` from a surface with no router handle
 * (the sidebar's project menus). Same hash-route move as the SDK's
 * `host.navigate`: set the hash, then front the workspace pane so a re-open
 * while a session tile is focused still brings the page forward.
 */
export function openProjectOverview(projectId: string): void {
  const to = projectOverviewRoute(projectId)

  window.location.hash = `#${to}`
  syncWorkspaceRoute(to)
}
