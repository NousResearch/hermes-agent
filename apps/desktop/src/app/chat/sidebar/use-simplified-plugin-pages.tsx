import { useStore } from '@nanostores/react'

import { Codicon } from '@/components/ui/codicon'
import { useContributions } from '@/contrib/react/use-contributions'
import { $interfaceMode, shownInMode } from '@/store/interface-mode'
import { applySidebarNavPrefs, SIDEBAR_NAV_PREFS_AREA } from '@/store/sidebar-nav'

import { SIDEBAR_NAV_AREA, type SidebarNavContribution } from '../../routes'

export function useSimplifiedPluginPages() {
  const contributions = useContributions(SIDEBAR_NAV_AREA)
  const preferences = useContributions(SIDEBAR_NAV_PREFS_AREA)
  const mode = useStore($interfaceMode)

  return applySidebarNavPrefs(
    contributions.flatMap(contribution => {
      const data = contribution.data as Partial<SidebarNavContribution> | undefined

      if (!data?.path?.startsWith('/') || !data.label || !shownInMode(mode)(data)) {
        return []
      }

      return [
        {
          id: contribution.id,
          label: data.label,
          route: data.path,
          icon: (props: { className?: string }) => <Codicon name={data.codicon || 'plug'} {...props} />
        }
      ]
    }),
    preferences
  )
}
