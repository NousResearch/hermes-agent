import { useStore } from '@nanostores/react'

import { SegmentedControl, type SegmentedControlOption } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'
import { Clock, FolderOpen, Users } from '@/lib/icons'
import { $sidebarGrouping, setSidebarGrouping, type SidebarGrouping } from '@/store/layout'
import { $showAllProfiles, setShowAllProfiles } from '@/store/profile'
import { $projectScope, ALL_PROJECTS, exitProjectScope } from '@/store/projects'

/**
 * The sidebar's top-level views. Sessions is the chronological flat list
 * (Today / Yesterday / Last week dividers); Projects is the project → repo →
 * lane tree; Profiles fans every profile's sessions into grouped sections.
 *
 * The three views map onto existing sidebar state — the flat grouping atom,
 * the workspace-grouped tree flag, and the all-profiles scope — so this bar is
 * a visible, labeled front door for controls that previously lived behind the
 * Filters menu (and, for Projects, a hover-only icon toggle that users kept
 * missing). See #66894 / #82730 / #79205.
 */
export type SidebarView = 'sessions' | 'projects' | 'profiles'

/**
 * Which of the three views the current store state is showing. `status` is a
 * flat-list grouping, so it still belongs to the Sessions view.
 */
export function resolveSidebarView(grouping: SidebarGrouping, showAllProfiles: boolean): SidebarView {
  // `profile` is the all-profiles scope's own grouping (group by owner). The
  // flat scope can't hold it, so seeing it is the Profiles view no matter what
  // showAllProfiles says.
  if (grouping === 'profile' || showAllProfiles) {
    return 'profiles'
  }

  return grouping === 'project' ? 'projects' : 'sessions'
}

interface SidebarViewSwitcherProps {
  /** The Profiles segment only exists when there is more than one profile. */
  multiProfile: boolean
}

export function SidebarViewSwitcher({ multiProfile }: SidebarViewSwitcherProps) {
  const { t } = useI18n()
  const s = t.sidebar
  const grouping = useStore($sidebarGrouping)
  const showAllProfiles = useStore($showAllProfiles)
  const projectScope = useStore($projectScope)

  const view = resolveSidebarView(grouping, showAllProfiles)

  const onViewChange = (next: SidebarView) => {
    if (next === 'profiles') {
      // Grouping by owner is the store's native "every profile, grouped by
      // profile" mode: it flips the all-profiles scope on and records the
      // grouping so the rail and the Filters menu agree (see
      // setSidebarGrouping).
      setSidebarGrouping('profile')

      return
    }

    // Leaving the all-profiles browse view comes back to the gateway profile;
    // $profileScope follows it automatically.
    setShowAllProfiles(false)

    // The bar is the top-level context switch, so it also leaves any project
    // drill-in — the drill-in keeps its own back row for a step at a time.
    if (projectScope !== ALL_PROJECTS) {
      exitProjectScope()
    }

    setSidebarGrouping(next === 'projects' ? 'project' : 'date')
  }

  const options: SegmentedControlOption<SidebarView>[] = [
    { id: 'sessions', label: s.viewSessions, icon: Clock },
    { id: 'projects', label: s.viewProjects, icon: FolderOpen }
  ]

  if (multiProfile) {
    options.push({ id: 'profiles', label: s.viewProfiles, icon: Users })
  }

  return (
    <div aria-label={s.viewAria} className="shrink-0 px-2 pb-1 pt-1" role="group">
      <SegmentedControl
        className="w-full"
        onChange={onViewChange}
        options={options}
        value={view}
      />
    </div>
  )
}
