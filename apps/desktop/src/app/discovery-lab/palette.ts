import { PALETTE_AREA, type PaletteContribution } from '@/app/command-palette/contrib'
import { registry } from '@/contrib/registry'
import { requestRoute } from '@/store/recovery-requests'
import { $portalShell } from '@/store/shell-variant'

import { DISCOVERY_LAB_ROUTE, syncWorkspaceRoute } from '../routes'

import { LAB_TITLE } from './lab-copy'

const KEYWORDS = ['discovery lab', 'onboarding', 'splash', 'sign up', 'testing', 'welcome', 'tray']

function openDiscoveryLab(): void {
  requestRoute(DISCOVERY_LAB_ROUTE)
  // Re-running it from the page with a chat tile in front changes no route, so front the page here.
  syncWorkspaceRoute(DISCOVERY_LAB_ROUTE)
}

/**
 * ⌘K "Discovery lab" (TESTING ONLY): the way in besides the URL; it is a test tool, never a rail
 * destination. Offered in the redesign only, whose onboarding it drives. Returns a disposer.
 */
export function installDiscoveryLabPaletteRow(): () => void {
  let dispose = () => {}

  const stop = $portalShell.subscribe(portal => {
    dispose()

    dispose = portal
      ? registry.register({
          id: 'discovery-lab.open',
          area: PALETTE_AREA,
          data: {
            id: 'discovery-lab.open',
            keywords: KEYWORDS,
            label: LAB_TITLE,
            run: openDiscoveryLab
          } satisfies PaletteContribution
        })
      : () => {}
  })

  return () => {
    stop()
    dispose()
  }
}
