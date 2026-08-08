import {
  type HermesPlugin,
  type RouteContribution,
  ROUTES_AREA,
  SIDEBAR_NAV_AREA,
  type SidebarNavContribution
} from '@hermes/plugin-sdk'

import { UsagePage } from './page'

const plugin: HermesPlugin = {
  id: 'usage',
  name: 'Usage',
  register(ctx) {
    ctx.registerMany([
      {
        id: 'page',
        area: ROUTES_AREA,
        data: { path: '/usage' } satisfies RouteContribution,
        render: () => <UsagePage />
      },
      {
        id: 'nav',
        area: SIDEBAR_NAV_AREA,
        order: 60,
        data: { codicon: 'graph-line', label: 'Usage', path: '/usage' } satisfies SidebarNavContribution
      }
    ])
  }
}

export default plugin
