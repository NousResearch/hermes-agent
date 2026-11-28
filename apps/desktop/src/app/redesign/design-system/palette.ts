import { PALETTE_AREA, type PaletteContribution } from '@/app/command-palette/contrib'
import { registry } from '@/contrib/registry'
import { getRuntimeI18nLocale, subscribeRuntimeI18nLocale } from '@/i18n/runtime'
import { requestRoute } from '@/store/recovery-requests'
import { $portalShell } from '@/store/shell-variant'

import { DESIGN_SYSTEM_ROUTE, syncWorkspaceRoute } from '../../routes'

import { designSystemCopy } from './design-system-copy'

const KEYWORDS = ['design system', 'tokens', 'components', 'primitives', 'style guide', 'colours', 'colors', 'type']

function openDesignSystem(): void {
  requestRoute(DESIGN_SYSTEM_ROUTE)
  // Re-running it from the page with a chat tile in front changes no route, so front the page here.
  syncWorkspaceRoute(DESIGN_SYSTEM_ROUTE)
}

/**
 * ⌘K "Design system": the only way in besides the URL (it is a reference, not a rail destination).
 * Offered in the redesign only, whose tokens the page documents, and labelled in the app's language.
 * Returns a disposer.
 */
export function installDesignSystemPaletteRow(): () => void {
  let dispose = () => {}

  const sync = () => {
    dispose()

    dispose = $portalShell.get()
      ? registry.register({
          id: 'design-system.open',
          area: PALETTE_AREA,
          data: {
            id: 'design-system.open',
            keywords: KEYWORDS,
            label: designSystemCopy(getRuntimeI18nLocale()).title,
            run: openDesignSystem
          } satisfies PaletteContribution
        })
      : () => {}
  }

  const stopShell = $portalShell.subscribe(sync)
  const stopLocale = subscribeRuntimeI18nLocale(sync)

  return () => {
    stopShell()
    stopLocale()
    dispose()
  }
}
