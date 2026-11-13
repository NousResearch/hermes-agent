/**
 * Move to Hermes Cloud: one page that carries this computer's agent into a Hermes Cloud agent.
 *
 * Reached from ⌘K, from `hermes://open/move-to-cloud` (the Nous Portal's empty agent page links
 * there), and from its own route. All network work goes through `move` in the SDK, which can only
 * call the `/api/move/*` routes; the bundle is built by this computer's backend and merged by the
 * cloud agent's, so nothing here touches files directly.
 */

import {
  type HermesPlugin,
  host,
  PALETTE_AREA,
  type PaletteContribution,
  type RouteContribution,
  ROUTES_AREA
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

    const registerLabels = () =>
      ctx.register({
        id: 'open',
        area: PALETTE_AREA,
        data: {
          id: 'move-to-cloud.open',
          label: bindCopy(ctx.i18n.t).command,
          keywords: ['move', 'migrate', 'cloud', 'hermes cloud', 'upload', 'transfer', 'export'],
          run: () => host.navigate(MOVE_ROUTE)
        } satisfies PaletteContribution
      })

    let disposeLabels = registerLabels()
    ctx.i18n.onLocaleChange(() => {
      disposeLabels()
      disposeLabels = registerLabels()
    })
  }
}

export default plugin
