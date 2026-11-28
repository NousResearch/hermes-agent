import { useStore } from '@nanostores/react'

import { FigmaAgentIcon } from '@/components/ui/figma-agent-icon'
import { useMediaQuery } from '@/hooks/use-media-query'
import { useI18n } from '@/i18n'
import { portalAgentSwatch } from '@/lib/portal-profile-color'
import { resolveProfileColor } from '@/lib/profile-color'
import { $sidebarOpen } from '@/store/layout'
import { $profileColors } from '@/store/profile'

import { PAGE_SIDEBAR_NARROW, usePageSidebarOpen } from '../overlays/page-sidebar'

import { useAgentSettingsCopy } from './copy'
import type { AgentCapabilityCounts } from './settings-counts'
import { AgentSettingsNavigation, useAgentSettingsLabels, useOpenAgentSettingsSection } from './settings-editor'
import type { AgentSettingsSection } from './settings-routes'

interface AgentSettingsPanelProps {
  counts?: AgentCapabilityCounts['nav']
  isDefault: boolean
  label: string
  onBack: () => void
  profile: string
  section: AgentSettingsSection | null
}

/** The agent's Settings tab (1036:7189): the page sidebar panel trades the agents list for this
 *  agent's settings menu, so the page has one vertical menu. "Back to agents" returns the panel to
 *  the list (the agent's Overview), as a history step the band's Back also undoes. */
export function AgentSettingsPanel({ counts, isDefault, label, onBack, profile, section }: AgentSettingsPanelProps) {
  const { t } = useI18n()
  const c = useAgentSettingsCopy()
  const labels = useAgentSettingsLabels()
  const openSection = useOpenAgentSettingsSection()
  const colors = useStore($profileColors)
  const open = usePageSidebarOpen()

  if (!open) {
    return null
  }

  return (
    <aside aria-label={t.commandCenter.settings} className="page-sidebar-panel agent-settings-panel">
      <button className="agent-settings-panel-back" onClick={onBack} type="button">
        <span aria-hidden>←</span>
        {c.backToAgents}
      </button>
      <div className="agent-settings-panel-agent">
        <FigmaAgentIcon color={portalAgentSwatch(resolveProfileColor(profile, colors)).hue} variant="subtle" />
        <span className="min-w-0 flex-1 truncate">{label}</span>
        {isDefault && <span className="agent-management-default">{t.profiles.defaultBadge}</span>}
      </div>
      <div className="page-sidebar-scroll">
        <AgentSettingsNavigation
          ariaLabel={t.commandCenter.settings}
          counts={counts}
          labels={labels}
          onOpen={openSection}
          section={section}
        />
      </div>
    </aside>
  )
}

/** Whether the settings menu must sit in the stage: the panel that carries it is folded away
 *  (⌘B) or the window is too narrow for panels (the menu then leads the page, as Settings does). */
export function useInlineAgentSettingsMenu(): boolean {
  const narrow = useMediaQuery(PAGE_SIDEBAR_NARROW)
  const panelOpen = useStore($sidebarOpen)

  return narrow || !panelOpen
}
