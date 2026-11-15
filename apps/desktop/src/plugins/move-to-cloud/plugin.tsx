/**
 * Backup & Move: back up this agent to a file, or carry it into a Hermes Cloud agent (merging, or
 * replacing the parts the user ticks).
 *
 * Lives in Settings ▸ Plugins ▸ Backup & Move. Reached from ⌘K, and from `hermes://open/move-to-cloud`
 * (the Nous Portal's empty agent page links there): that route only forwards to Settings. All
 * network work goes through `move` in the SDK, which can only call the `/api/move/*` routes; the
 * bundle is built by this computer's backend and merged by the cloud agent's, so nothing here
 * touches files directly.
 */

import {
  type HermesPlugin,
  host,
  move,
  PALETTE_AREA,
  type PaletteContribution,
  type RouteContribution,
  ROUTES_AREA
} from '@hermes/plugin-sdk'
import { useEffect } from 'react'

import { bindCopy, MOVE_LOCALES } from './i18n'
import { MOVE_ROUTE, moveSettingsHref, settingsHrefForRoute } from './location'
import { MoveToCloudPage } from './page'

/** The old route: replace it in history with the Settings location, keeping the named agent. */
function ForwardToSettings() {
  useEffect(() => window.location.replace(`#${settingsHrefForRoute(window.location.hash)}`), [])

  return null
}

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
      render: () => <ForwardToSettings />
    })

    // On a Webapp Hermes Cloud serves, this page is backup only (`move.transferAvailable`).
    const transfer = move.transferAvailable()
    const open = () => host.navigate(moveSettingsHref())

    // Registered once: re-registering on a locale change would remount the page and abort a move
    // in flight.
    const title = bindCopy(ctx.i18n.t)
    ctx.registerSettingsPage({
      icon: transfer ? 'cloud-upload' : 'archive',
      id: 'page',
      render: () => <MoveToCloudPage openExternal={url => ctx.os.openExternal(url)} />,
      title: transfer ? title.title : title.backupOnlyTitle
    })

    const registerLabels = () => {
      const copy = bindCopy(ctx.i18n.t)

      return ctx.registerMany([
        {
          id: 'backup',
          area: PALETTE_AREA,
          data: {
            id: 'move-to-cloud.backup',
            label: copy.backupCommand,
            keywords: ['backup', 'export', 'archive', 'save', 'download', 'settings'],
            run: open
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
                  keywords: ['move', 'migrate', 'cloud', 'hermes cloud', 'upload', 'transfer', 'replace', 'settings'],
                  run: open
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
