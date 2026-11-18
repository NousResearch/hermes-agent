import { useStore } from '@nanostores/react'

import { Codicon } from '@/components/ui/codicon'
import { useContributions } from '@/contrib/react/use-contributions'
import { pluginSettingsRouteHref, SETTINGS_PLUGINS_AREA } from '@/contrib/settings-pages'
import { $interfaceMode, shownInMode } from '@/store/interface-mode'
import { $activeGatewayProfile } from '@/store/profile'

import { SIDEBAR_NAV_AREA, type SidebarNavContribution } from '../../routes'

/** The rail's plugin destinations, in registry order. The rail applies `sidebarNav.prefs` over them
 *  together with its own movable destinations, as classic does over its nav rows. */
export function useSimplifiedPluginPages() {
  const contributions = useContributions(SIDEBAR_NAV_AREA)
  const settings = useContributions(SETTINGS_PLUGINS_AREA)
  const profile = useStore($activeGatewayProfile)
  const mode = useStore($interfaceMode)

  return contributions.flatMap(contribution => {
    const data = contribution.data as Partial<SidebarNavContribution> | undefined

    if (!data?.path?.startsWith('/') || !data.label || !shownInMode(mode)(data)) {
      return []
    }

    return [
      {
        id: contribution.id,
        label: data.label,
        route: data.path,
        settingsRoute:
          contribution.source?.startsWith('plugin:') && settings.some(page => page.source === contribution.source)
            ? pluginSettingsRouteHref({ plugin: contribution.source.slice('plugin:'.length) }, profile)
            : undefined,
        icon: (props: { className?: string }) => <Codicon name={data.codicon || 'plug'} {...props} />
      }
    ]
  })
}
