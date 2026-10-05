import { $profiles } from '@/store/profile'

export interface PluginProfileRoute {
  connectionId: string
  mode: 'local' | 'remote'
  /** Electron's authoritative registry primary. Absent on older shells. */
  primary?: true
  /** Desktop profile used to select the connection route. */
  profile: string
  /** Backend Hermes profile served by that route. */
  targetProfile: string
}

export async function pluginRouteStillRegistered(route: PluginProfileRoute): Promise<boolean> {
  const getProfileRoutes = window.hermesDesktop?.getProfileRoutes

  if (!getProfileRoutes) {
    return false
  }

  try {
    const routes = await getProfileRoutes($profiles.get().map(profile => profile.name))

    return routes.some(
      candidate =>
        candidate.connectionId === route.connectionId &&
        candidate.profile === route.profile &&
        candidate.targetProfile === route.targetProfile
    )
  } catch {
    return false
  }
}
