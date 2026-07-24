/**
 * Kanban — the founding plugin use case, now pure SDK-consumer work: a
 * first-class `/kanban` board page + sidebar nav row + a live statusbar count,
 * all reusing the existing `plugins/kanban/dashboard/plugin_api.py` REST router
 * through `ctx.rest` (namespace-scoped to `/api/plugins/kanban`). No new
 * backend, no core edits.
 *
 * Ships OFF by default (`defaultEnabled: false`): it inventories in
 * Settings ▸ Plugins and registers nothing until the user flips the switch.
 */

import './kanban.css'

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
  useQuery,
  useValue
} from '@hermes/plugin-sdk'

import { $boardSlug, bindApi, boardKey, fetchBoard } from './api'
import { KanbanBoardPage } from './board'
import { KANBAN_LOCALES } from './i18n'
import { useKanban } from './ui'

// Live "N running / ready" pill — one glance at fleet activity from anywhere,
// clicks through to the board. Shares the board query (one cache, one poll with
// the page); hidden when nothing is in flight (or unloaded).
function KanbanCount() {
  const k = useKanban()
  const slug = useValue($boardSlug)

  // Socket-invalidated like the page (same cache); slow socketless heartbeat.
  const { data: board } = useQuery({
    queryFn: () => fetchBoard(false),
    queryKey: boardKey(slug, false),
    refetchInterval: 60_000
  })

  if (!board) {
    return null
  }

  const count = (name: string) => board.columns.find(col => col.name === name)?.tasks.length ?? 0
  const active = count('running') + count('ready')

  if (active === 0) {
    return null
  }

  return (
    <Tip label={k.countTip(count('running'), count('ready'))}>
      <button
        className={cn(
          'inline-flex h-full items-center gap-1 rounded-none px-1.5 text-[0.6875rem] tabular-nums transition-colors',
          'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
        )}
        onClick={() => host.navigate('/kanban')}
        type="button"
      >
        <Codicon name="project" size="0.7rem" />
        <span>{active}</span>
      </button>
    </Tip>
  )
}

const plugin: HermesPlugin = {
  id: 'kanban',
  name: 'Kanban',
  defaultEnabled: false,
  register(ctx) {
    ctx.i18n.register(KANBAN_LOCALES)
    ctx.onDispose(bindApi(ctx.rest, ctx.storage, ctx.socket))

    ctx.registerMany([
      {
        id: 'page',
        area: ROUTES_AREA,
        data: { path: '/kanban' } satisfies RouteContribution,
        render: () => <KanbanBoardPage />
      },
      {
        id: 'nav',
        area: SIDEBAR_NAV_AREA,
        order: 50,
        data: { codicon: 'project', label: 'Kanban', path: '/kanban' } satisfies SidebarNavContribution
      },
      {
        id: 'count',
        area: STATUSBAR_AREAS.right,
        order: 80,
        render: () => <KanbanCount />
      },
      {
        id: 'open',
        area: PALETTE_AREA,
        data: {
          id: 'kanban.open',
          label: 'Kanban: Open board',
          keywords: ['kanban', 'board', 'tasks', 'agents'],
          run: () => host.navigate('/kanban')
        } satisfies PaletteContribution
      }
    ])
  }
}

export default plugin
