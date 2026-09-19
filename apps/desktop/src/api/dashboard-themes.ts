import type { DashboardThemesResponse } from '@/types/hermes'

import { hermesApi } from './client'

// Deliberately NOT profileScoped(): `dashboard.theme` lives in the dashboard
// server's own home, and `/api/dashboard/*` is not a profile-scoped route — a
// `?profile=` would route the request to that profile's pooled backend and
// write `~/.hermes/profiles/<p>/config.yaml`, a file the web Dashboard never
// reads. hermesApi() still adds the connection scope, which is the routing
// this key does honour.

/** Available dashboard themes + the active one (`dashboard.theme` in
 *  config.yaml) — the same active-theme key the web Dashboard reads, so a
 *  name shared between the two surfaces can be synced across them. */
export function getDashboardThemes(): Promise<DashboardThemesResponse> {
  return hermesApi<DashboardThemesResponse>({
    path: '/api/dashboard/themes'
  })
}

/** Set the active dashboard theme (persists to config.yaml -> dashboard.theme). */
export function setDashboardTheme(name: string): Promise<{ ok: boolean; theme: string }> {
  return hermesApi<{ ok: boolean; theme: string }>({
    path: '/api/dashboard/theme',
    method: 'PUT',
    body: { name }
  })
}
