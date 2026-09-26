/**
 * Native Desktop surface for the canonical hermes-achievements dashboard
 * backend. The plugin owns presentation only; every achievement definition,
 * scan, evaluation, unlock, and category comes from plugin_api.py via ctx.rest.
 */

import {
  type HermesPlugin,
  host,
  PALETTE_AREA,
  type PaletteContribution,
  type RouteContribution,
  ROUTES_AREA,
  SIDEBAR_NAV_AREA,
  type SidebarNavContribution
} from '@hermes/plugin-sdk'

import { bindAchievementsApi } from './api'
import { ACHIEVEMENTS_LOCALES } from './i18n'
import { AchievementsPage } from './page'

export const ACHIEVEMENTS_ROUTE = '/achievements'

const plugin: HermesPlugin = {
  id: 'hermes-achievements',
  name: 'Achievements',
  description: 'Progress, unlocks, and category milestones from your Hermes session history.',
  defaultEnabled: true,
  register(ctx) {
    ctx.i18n.register(ACHIEVEMENTS_LOCALES)
    ctx.onDispose(bindAchievementsApi(ctx.rest))

    const open = () => host.navigate(ACHIEVEMENTS_ROUTE)

    ctx.registerMany([
      {
        id: 'page',
        area: ROUTES_AREA,
        title: ctx.i18n.t('title'),
        data: { path: ACHIEVEMENTS_ROUTE } satisfies RouteContribution,
        render: () => <AchievementsPage />
      },
      {
        id: 'nav',
        area: SIDEBAR_NAV_AREA,
        order: 60,
        data: {
          codicon: 'star-full',
          label: ctx.i18n.t('nav'),
          path: ACHIEVEMENTS_ROUTE
        } satisfies SidebarNavContribution
      },
      {
        id: 'open',
        area: PALETTE_AREA,
        data: {
          id: 'hermes-achievements.open',
          keywords: ['achievements', 'badges', 'progress', 'unlocks', 'trophies'],
          label: ctx.i18n.t('open'),
          run: open
        } satisfies PaletteContribution
      }
    ])
  }
}

export default plugin
