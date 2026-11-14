import { useMemo } from 'react'
import { Navigate, useLocation } from 'react-router'

import { useI18n } from '@/i18n'

import { CapabilitiesView } from '../capabilities'
import { MessagingView } from '../messaging'
import { Panel, PanelHeader } from '../overlays/panel'
import { SettingsView } from '../settings'
import { SettingsOwnerContext } from '../settings/profile-scope'
import type { SettingsPageProps } from '../settings/types'

import { useAgentSettingsCopy } from './copy'
import type { AgentSettingsOwner } from './data'
import { AgentGeneralSettings, AgentInstructionsSettings } from './settings-general'
import { classicAgentSettingsSearch, resolveAgentSettingsSection } from './settings-routes'
import { AgentSchedulesSettings } from './settings-schedules'

interface ClassicAgentSettingsProps extends SettingsPageProps {
  owner: AgentSettingsOwner
  onProfileRenamed: (name: string) => Promise<void>
}

export function ClassicAgentSettings(props: ClassicAgentSettingsProps) {
  const owner = useMemo(
    () => ({ connectionId: props.owner.connectionId ?? 'local', profile: props.owner.profile }),
    [props.owner.connectionId, props.owner.profile]
  )

  return (
    <SettingsOwnerContext.Provider value={owner}>
      <ClassicAgentSettingsContent {...props} />
    </SettingsOwnerContext.Provider>
  )
}

function ClassicAgentSettingsContent(props: ClassicAgentSettingsProps) {
  const location = useLocation()
  const { t } = useI18n()
  const c = useAgentSettingsCopy()
  const search = classicAgentSettingsSearch(location.search)
  const params = new URLSearchParams(search)
  const section = resolveAgentSettingsSection(params)

  if (search !== location.search) {
    return <Navigate replace to={{ ...location, search }} />
  }

  if (params.get('panel') === 'capabilities') {
    return (
      <Panel onClose={props.onClose}>
        <CapabilitiesView fixedConnection={props.owner.connectionId} fixedProfile={props.owner.profile} />
      </Panel>
    )
  }

  if (params.get('panel') === 'messaging' || section === 'messaging') {
    return (
      <Panel onClose={props.onClose}>
        <MessagingView />
      </Panel>
    )
  }

  if (params.get('panel') === 'settings' || params.get('tab')?.startsWith('agent:')) {
    if (section === 'general' || section === 'instructions' || section === 'schedules') {
      return (
        <Panel onClose={props.onClose}>
          <PanelHeader
            subtitle={props.owner.profile}
            title={
              section === 'instructions' ? c.instructions : section === 'schedules' ? t.cron.title : t.profiles.title
            }
          />
          {section === 'general' && (
            <AgentGeneralSettings onProfileRenamed={props.onProfileRenamed} owner={props.owner} />
          )}
          {section === 'instructions' && <AgentInstructionsSettings owner={props.owner} />}
          {section === 'schedules' && <AgentSchedulesSettings owner={props.owner} />}
        </Panel>
      )
    }
  }

  return <SettingsView {...props} />
}
