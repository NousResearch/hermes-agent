/**
 * Achievements — a first-class `/achievements` page + sidebar nav row + a live
 * statusbar score chip + a ⌘K command, reusing the existing
 * `plugins/hermes-achievements/dashboard/plugin_api.py` REST router through
 * `ctx.rest` (namespace-scoped to `/api/plugins/hermes-achievements`). No new
 * backend, no core edits.
 *
 * Ships OFF by default (`defaultEnabled: false`): it inventories in
 * Settings ▸ Plugins and registers nothing until the user flips the switch.
 */

import {
  cn,
  Codicon,
  type HermesPlugin,
  host,
  PALETTE_AREA,
  type PaletteContribution,
  type RouteContribution,
  ROUTES_AREA,
  SIDEBAR_NAV_AREA,
  type SidebarNavContribution,
  STATUSBAR_AREAS,
  Tip,
  useQuery
} from '@hermes/plugin-sdk'

import { achievementsKey, bindApi, fetchAchievements } from './api'
import { ACHIEVEMENTS_LOCALES, useAchievementsI18n } from './i18n'
import { AchievementsPage } from './page'

// Live "36/60" pill — one glance at the score from anywhere, clicks through
// to the page. Shares the page's query (one cache, one poll).
function ScoreChip(): React.JSX.Element | null {
  const k = useAchievementsI18n()

  const { data } = useQuery({
    queryKey: achievementsKey(),
    queryFn: fetchAchievements,
    refetchInterval: 120_000
  })

  if (!data?.unlocked_count) {
    return null
  }

  return (
    <Tip label={k.scoreTip(data.unlocked_count, data.total_count)}>
      <button
        className={cn(
          'inline-flex h-full items-center gap-1 rounded-none px-1.5 text-[0.6875rem] tabular-nums transition-colors',
          'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
        )}
        onClick={() => host.navigate('/achievements')}
        type="button"
      >
        <Codicon name="milestone" size="0.7rem" />
        <span>
          {data.unlocked_count}/{data.total_count}
        </span>
      </button>
    </Tip>
  )
}

const plugin: HermesPlugin = {
  id: 'hermes-achievements',
  name: 'Achievements',
  defaultEnabled: false,
  register(ctx) {
    ctx.i18n.register(ACHIEVEMENTS_LOCALES)
    ctx.onDispose(bindApi(ctx.rest))

    ctx.registerMany([
      {
        id: 'page',
        area: ROUTES_AREA,
        data: { path: '/achievements' } satisfies RouteContribution,
        title: 'Achievements',
        render: () => <AchievementsPage />
      },
      {
        id: 'nav',
        area: SIDEBAR_NAV_AREA,
        order: 55,
        data: { codicon: 'milestone', label: 'Achievements', path: '/achievements' } satisfies SidebarNavContribution
      },
      {
        id: 'score',
        area: STATUSBAR_AREAS.right,
        order: 90,
        render: () => <ScoreChip />
      },
      {
        id: 'open',
        area: PALETTE_AREA,
        data: {
          id: 'hermes-achievements.open',
          label: 'Achievements: Open',
          keywords: ['achievements', 'badges', 'tiers', 'trophy'],
          run: () => host.navigate('/achievements')
        } satisfies PaletteContribution
      }
    ])
  }
}

export default plugin
