import type {
  HermesPlugin,
  PluginContext,
  RouteContribution,
  SidebarNavContribution
} from '@hermes/plugin-sdk'
import { ROUTES_AREA, SIDEBAR_NAV_AREA } from '@hermes/plugin-sdk'

import { bindApi } from './api'
import { VaultPage } from './vault-page'

const plugin: HermesPlugin = {
  id: 'vault',
  name: 'Hermes Vault',
  description: 'Local-first Markdown knowledge base (Obsidian-compatible) with bidirectional links and graph view',
  defaultEnabled: true,
  register(ctx: PluginContext) {
    bindApi(ctx)

    ctx.registerMany([
      {
        id: 'page',
        area: ROUTES_AREA,
        data: { path: '/vault' } satisfies RouteContribution,
        render: () => <VaultPage />
      },
      {
        id: 'nav',
        area: SIDEBAR_NAV_AREA,
        order: 48,
        data: { codicon: 'book', label: 'Vault', path: '/vault' } satisfies SidebarNavContribution
      }
    ])
  }
}

export default plugin
