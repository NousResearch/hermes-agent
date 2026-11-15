/**
 * Backup & Move: back up this agent to a file, or carry it into a Hermes Cloud agent (merging, or
 * replacing the parts the user ticks).
 *
 * Reached from ⌘K, from `hermes://open/move-to-cloud` (the Nous Portal's empty agent page links
 * there), and from its own route. All network work goes through `move` in the SDK, which can only
 * call the `/api/move/*` routes; the bundle is built by this computer's backend and merged by the
 * cloud agent's, so nothing here touches files directly.
 */

import {
  type HermesPlugin,
  host,
  move,
  PALETTE_AREA,
  type PaletteContribution,
  type RouteContribution,
  ROUTES_AREA,
  SIDEBAR_NAV_AREA,
  type SidebarNavContribution
} from '@hermes/plugin-sdk'

import { bindCopy, MOVE_LOCALES } from './i18n'
import { MoveToCloudPage } from './page'

export const MOVE_ROUTE = '/move-to-cloud'

const plugin: HermesPlugin = {
  id: 'move-to-cloud',
  name: 'Move to Hermes Cloud',
  description: 'Copy this agent’s skills, memory, persona, schedules and conversations to a Hermes Cloud agent.',
  register(ctx) {
    ctx.i18n.register(MOVE_LOCALES)
    ctx.register({
      id: 'page',
      area: ROUTES_AREA,
      data: { path: MOVE_ROUTE } satisfies RouteContribution,
      render: () => <MoveToCloudPage openExternal={url => ctx.os.openExternal(url)} />
    })

    // On a Webapp Hermes Cloud serves, this page is backup only (`move.transferAvailable`).
    const registerLabels = () => {
      const copy = bindCopy(ctx.i18n.t)
      const transfer = move.transferAvailable()

      return ctx.registerMany([
        {
          id: 'nav',
          area: SIDEBAR_NAV_AREA,
          order: 90,
          data: {
            codicon: transfer ? 'cloud-upload' : 'archive',
            label: transfer ? copy.nav : copy.backupOnlyTitle,
            path: MOVE_ROUTE
          } satisfies SidebarNavContribution
        },
        {
          id: 'backup',
          area: PALETTE_AREA,
          data: {
            id: 'move-to-cloud.backup',
            label: copy.backupCommand,
            keywords: ['backup', 'export', 'archive', 'save', 'download'],
            run: () => host.navigate(MOVE_ROUTE)
          } satisfies PaletteContribution
        },
        ...(transfer
          ? [
              {
                id: 'open',
                area: PALETTE_AREA,
                data: {
                  id: 'move-to-cloud.open',
                  label: copy.command,
                  keywords: ['move', 'migrate', 'cloud', 'hermes cloud', 'upload', 'transfer', 'replace'],
                  run: () => host.navigate(MOVE_ROUTE)
                } satisfies PaletteContribution
              }
            ]
          : [])
      ])
    }

    let disposeLabels = registerLabels()
    ctx.i18n.onLocaleChange(() => {
      disposeLabels()
      disposeLabels = registerLabels()
    })
  }
}

export default plugin
