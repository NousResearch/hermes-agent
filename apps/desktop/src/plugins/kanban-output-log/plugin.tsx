/**
 * Kanban output log — a live, read-only view of a card worker's transcript.
 *
 * A companion plugin to the bundled Kanban board. The board's card drawer
 * exposes the `kanban.card.drawer` contribution area (see
 * `@/contrib/kanban-card-drawer`); this plugin registers its "Output log"
 * section there. The section reads its data from THIS plugin's own backend
 * router (`plugins/kanban-output-log/dashboard/plugin_api.py`, namespace
 * `/api/plugins/kanban-output-log`), reached through `ctx.rest`.
 *
 * Enabled by default (no `defaultEnabled: false`): the section only appears
 * inside a card drawer on the Kanban board, so it costs nothing until a user
 * opens a card.
 */

import { type HermesPlugin, KANBAN_CARD_DRAWER_AREAS, type KanbanCardDrawerSectionContribution } from '@hermes/plugin-sdk'

import { bindOutputLogRest, OutputLogSection } from './output-log'

const plugin: HermesPlugin = {
  id: 'kanban-output-log',
  name: 'Kanban output log',
  description:
    'Live, read-only transcript of a Kanban card’s worker (progress, tool calls and thinking) in the Desktop board’s card drawer.',
  register(ctx) {
    bindOutputLogRest(ctx.rest)
    ctx.register({
      id: 'output-log',
      order: 10,
      area: KANBAN_CARD_DRAWER_AREAS.sections,
      data: { Section: OutputLogSection } satisfies KanbanCardDrawerSectionContribution
    })
  }
}

export default plugin
