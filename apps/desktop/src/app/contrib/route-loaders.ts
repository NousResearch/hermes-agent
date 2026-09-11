export const loadArtifactsView = () => import('../artifacts')
export const loadCronView = () => import('../cron')
export const loadMessagingView = () => import('../messaging')
export const loadSettingsView = () => import('../settings')
export const loadSkillsView = () => import('../skills')

/** Common first-click routes, ordered from lighter/more frequent to heavier. */
export const COMMON_ROUTE_WARMUP_LOADERS = [
  loadMessagingView,
  loadArtifactsView,
  loadCronView,
  loadSettingsView,
  loadSkillsView
] as const

export function shouldWarmCommonRoutes(gatewayOpen: boolean, auxiliaryWindow: boolean): boolean {
  return gatewayOpen && !auxiliaryWindow
}
