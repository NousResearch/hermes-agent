/**
 * Local Models — bundled desktop plugin.
 * 
 * Provides the /local-models page in the desktop app, reusing the
 * plugins/local-models/dashboard/plugin_api.py REST router.
 * 
 * Ships ON by default (bundled, like Kanban's core UI).
 */

import type {
  HermesPlugin,
  RouteContribution,
  SidebarNavContribution,
  ROUTES_AREA,
  SIDEBAR_NAV_AREA,
} from '@hermes/plugin-sdk'

import { LocalModelsSettings } from './local-models-settings'

export default {
  id: 'local-models',
  name: 'Local Models',
  register(ctx) {
    // Register the /local-models route
    ctx.register<RouteContribution>({
      id: 'local-models-page',
      area: 'routes' as typeof ROUTES_AREA,
      data: {
        path: '/local-models',
        element: <LocalModelsSettings />,
      },
    })

    // Register sidebar nav item under Providers
    ctx.register<SidebarNavContribution>({
      id: 'local-models-nav',
      area: 'sidebar-nav' as typeof SIDEBAR_NAV_AREA,
      data: {
        path: '/local-models',
        label: 'Local Models',
        icon: 'cpu',
        section: 'Providers',
        order: 10,
      },
    })
  },
} satisfies HermesPlugin
