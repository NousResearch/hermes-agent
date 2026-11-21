/**
 * AgentTrace — hierarchical monitoring of delegated subagents and tool calls.
 *
 * A first-class `/agent-trace` page + sidebar nav row, a right-docked live
 * pane, and a statusbar chip — all fed by the gateway's existing `tool.*` /
 * `subagent.*` event stream and snapshot RPCs (`delegation.status`,
 * `subagent.list`, `session.history`). No new backend, no core edits.
 *
 * Ships OFF by default (`defaultEnabled: false`): it inventories in
 * Capabilities ▸ Plugins and registers nothing until the user flips the
 * switch (the kanban pattern).
 */

import './agent-trace.css'

import {
  Codicon,
  type HermesPlugin,
  host,
  type KeybindContribution,
  KEYBINDS_AREA,
  PALETTE_AREA,
  type PaletteContribution,
  PANES_AREA,
  type RouteContribution,
  ROUTES_AREA,
  SIDEBAR_NAV_AREA,
  type SidebarNavContribution,
  STATUSBAR_AREAS,
  Tip,
  useValue
} from '@hermes/plugin-sdk'

import { AGENT_TRACE_LOCALES, useAgentTrace } from './i18n'
import { TracePage } from './page'
import { TracePane } from './pane'
import { $trace, activeSubagentCount, bindTrace, runningToolCount } from './trace-store'

/** Live "N active steps" chip — one glance at delegation/tool activity from
 *  anywhere; clicks through to the full trace. Hidden when nothing is in
 *  flight (the kanban count pattern). */
function TraceChip() {
  const k = useAgentTrace()
  const state = useValue($trace)
  const agents = activeSubagentCount(state)
  const tools = runningToolCount(state)
  const total = agents + tools

  if (total === 0) {
    return null
  }

  return (
    <Tip label={k.countTip(agents, tools)}>
      <button
        className="inline-flex h-full items-center gap-1 rounded-none px-1.5 text-[0.6875rem] tabular-nums text-(--ui-text-tertiary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground"
        onClick={() => host.navigate('/agent-trace')}
        type="button"
      >
        <Codicon name="pulse" size="0.7rem" />
        <span>{total}</span>
      </button>
    </Tip>
  )
}

const plugin: HermesPlugin = {
  id: 'agent-trace',
  name: 'AgentTrace',
  description:
    'Hierarchical execution monitoring — a live subagent/tool-call tree per session, a right-docked pane, and a statusbar activity chip.',
  defaultEnabled: false,
  register(ctx) {
    ctx.i18n.register(AGENT_TRACE_LOCALES)
    bindTrace(ctx)

    const openTrace = () => host.navigate('/agent-trace')

    ctx.registerMany([
      {
        id: 'page',
        area: ROUTES_AREA,
        data: { path: '/agent-trace' } satisfies RouteContribution,
        render: () => <TracePage />
      },
      {
        id: 'pane',
        area: PANES_AREA,
        title: 'AgentTrace',
        data: { placement: 'right', width: '300px' },
        render: () => <TracePane />
      },
      {
        id: 'chip',
        area: STATUSBAR_AREAS.right,
        order: 85,
        render: () => <TraceChip />
      }
    ])

    // Labels re-register on locale switch so nav/palette/keybind rows follow
    // the active language (the kanban pattern).
    const registerLabels = () =>
      ctx.registerMany([
        {
          id: 'nav',
          area: SIDEBAR_NAV_AREA,
          order: 55,
          data: { codicon: 'pulse', label: ctx.i18n.t('nav'), path: '/agent-trace' } satisfies SidebarNavContribution
        },
        {
          id: 'open',
          area: PALETTE_AREA,
          data: {
            id: 'agentTrace.open',
            action: 'agentTrace.open',
            label: ctx.i18n.t('openCommand'),
            keywords: ['agenttrace', 'trace', 'subagents', 'delegation', 'tools', 'monitoring'],
            run: openTrace
          } satisfies PaletteContribution
        },
        {
          id: 'open',
          area: KEYBINDS_AREA,
          data: {
            id: 'agentTrace.open',
            category: 'view',
            defaults: ['mod+alt+a'],
            label: ctx.i18n.t('openCommand'),
            run: openTrace
          } satisfies KeybindContribution
        }
      ])

    let disposeLabels = registerLabels()
    ctx.i18n.onLocaleChange(() => {
      disposeLabels()
      disposeLabels = registerLabels()
    })
  }
}

export default plugin
