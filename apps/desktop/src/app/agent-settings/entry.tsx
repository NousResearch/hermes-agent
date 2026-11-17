import { Suspense, useRef, useState } from 'react'
import { useLocation } from 'react-router'

import { lazyView } from '@/lib/lazy-view'
import { usePortalShell } from '@/store/shell-variant'

import { PageFrameFallback } from '../overlays/page-frame-fallback'
import { SettingsView as ClassicSettingsView } from '../settings'
import { type ConfigSettingsDraft, ConfigSettingsDraftContext } from '../settings/config-draft'
import type { SettingsPageProps } from '../settings/types'

import { agentRouteOwner } from './overview-queries'
import { AgentInstructionsDraftContext, EMPTY_AGENT_INSTRUCTIONS_DRAFT } from './settings-general'

const AgentSettingsView = lazyView(async () => (await import('./index')).AgentSettingsView)

/** The rail warms this chunk too, so an agent page paints without a second Suspense wait. */
export const preloadAgentSettingsView = AgentSettingsView.preload

/** An explicit agent route opens the redesign's owner-pinned agent page. Classic has no agent page,
 *  so it keeps Settings exactly as it always was, whatever the route names. */
export function SettingsView(props: SettingsPageProps) {
  const redesigned = usePortalShell()
  const { search } = useLocation()
  const params = new URLSearchParams(search)
  const profile = params.get('agentProfile')
  const connectionId = params.get('connection') || undefined

  if (!profile || !redesigned) {
    return <ClassicSettingsView {...props} />
  }

  return (
    <AgentSettingsRoute
      {...props}
      connectionId={connectionId}
      key={JSON.stringify([connectionId, profile])}
      profile={profile}
    />
  )
}

interface AgentSettingsRouteProps extends SettingsPageProps {
  connectionId?: string
  profile: string
}

function AgentSettingsRoute({ connectionId, profile, ...props }: AgentSettingsRouteProps) {
  // Pin the ambient owner on entry, not again when a gateway or presentation changes.
  const [owner] = useState(() => agentRouteOwner(connectionId, profile))

  const configDraft = useRef<ConfigSettingsDraft | null>(null)
  const instructionsDraft = useState(EMPTY_AGENT_INSTRUCTIONS_DRAFT)

  return (
    <ConfigSettingsDraftContext.Provider value={configDraft}>
      <AgentInstructionsDraftContext.Provider value={instructionsDraft}>
        <Suspense fallback={<PageFrameFallback frame="agent" onClose={props.onClose} />}>
          <AgentSettingsView {...props} owner={owner} />
        </Suspense>
      </AgentInstructionsDraftContext.Provider>
    </ConfigSettingsDraftContext.Provider>
  )
}
