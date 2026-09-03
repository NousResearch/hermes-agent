/**
 * Owner Command Center — Desktop plugin entry point.
 *
 * Contributes a full-page workspace route at /owner-command-center and a
 * sidebar navigation entry labeled "Command Center". Does NOT touch the
 * existing /command-center technical admin overlay — that route and its
 * CommandCenterView remain completely unchanged.
 *
 * Route: /owner-command-center (contributed full-page, not an overlay)
 * Sidebar label: Command Center
 * Palette: "Command Center: Open"
 *
 * Data source (M1, per Architect's Path B correction, 2026-08-31): a fully
 * in-process, deterministic synthetic provider (./provider.ts), read via
 * ./api.ts. There is no backend adapter, no server route, and no network
 * call anywhere in this plugin — `ctx.rest` is never bound or invoked.
 * `/api/plugins/command-center/v1/*` is a PROPOSED future integration
 * namespace only; it is not mounted or implemented in M1.
 * All data is SIMULATED. No real Hermes/Syntex/EKV integration in M1.
 */

import './enterprise.css'

import type {
  HermesPlugin,
  KeybindContribution,
  PaletteContribution,
  RouteContribution,
  SidebarNavContribution,
} from '@hermes/plugin-sdk'
import { host, KEYBINDS_AREA, PALETTE_AREA, ROUTES_AREA, SIDEBAR_NAV_AREA } from '@hermes/plugin-sdk'

import { OwnerCommandCenterPage } from './page'

const ROUTE = '/owner-command-center'

const plugin: HermesPlugin = {
  id: 'enterprise',
  name: 'Owner Command Center',
  description: 'Owner-facing enterprise Command Center — visualization, awareness, and synthetic control surface (M1).',
  defaultEnabled: true,

  register(ctx) {
    // M1: no data-layer binding step. ./api.ts reads directly from the
    // in-process synthetic provider (./provider.ts) — there is no REST
    // channel to bind and no ctx.rest call anywhere in this plugin.

    ctx.registerMany([
      // Full-page route — NOT an overlay. Workspace/main surface.
      {
        id: 'page',
        area: ROUTES_AREA,
        data: { path: ROUTE } satisfies RouteContribution,
        render: () => <OwnerCommandCenterPage />,
      },

      // Sidebar navigation entry
      {
        id: 'nav',
        area: SIDEBAR_NAV_AREA,
        order: 60,
        data: {
          codicon: 'organization',
          label: 'Command Center',
          path: ROUTE,
        } satisfies SidebarNavContribution,
      },

      // Command palette entry
      {
        id: 'open',
        area: PALETTE_AREA,
        data: {
          id: 'enterprise.open',
          label: 'Command Center: Open',
          keywords: ['command', 'center', 'owner', 'enterprise', 'attention', 'decisions', 'agents'],
          run: () => host.navigate(ROUTE),
        } satisfies PaletteContribution,
      },

      // Keyboard shortcut — ⌘⌥E (mod+alt+e)
      // mod+alt+<letter> is the safe plugin namespace (core never uses it with letters)
      {
        id: 'open-keybind',
        area: KEYBINDS_AREA,
        data: {
          id: 'enterprise.open',
          category: 'view',
          defaults: ['mod+alt+e'],
          label: 'Command Center: Open',
          run: () => host.navigate(ROUTE),
        } satisfies KeybindContribution,
      },
    ])
  },
}

export default plugin
