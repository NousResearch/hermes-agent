/**
 * Welcome — the first Unitalk Desktop Plugin.
 * Replaces the empty new-session screen with a warm, professional welcome
 * dashboard for non-technical users. Pure SDK consumer — no core edits.
 */

import type { HermesPlugin } from '@hermes/plugin-sdk'
import { ROUTES_AREA, SIDEBAR_NAV_AREA } from '@hermes/plugin-sdk'
import { WelcomePage } from './page'

const plugin: HermesPlugin = {
  id: 'welcome',
  name: 'Accueil',
  description:
    "Tableau de bord d'accueil pour votre Collaborateur IA.",
  defaultEnabled: true,
  register(ctx) {
    ctx.registerMany([
      {
        id: 'page',
        area: ROUTES_AREA,
        data: { path: '/' },
        render: () => <WelcomePage />
      },
      {
        id: 'nav',
        area: SIDEBAR_NAV_AREA,
        order: 0,
        data: { codicon: 'home', label: 'Accueil', path: '/' }
      }
    ])
  }
}

export default plugin