import { useStore } from '@nanostores/react'
import { lazy, Suspense, useRef, useState } from 'react'
import { useLocation } from 'react-router'

import { resolveOwnerNow } from '@/api/client'
import { PageLoader } from '@/components/page-loader'
import { $activeConnectionId } from '@/store/connections'
import { $simplifiedAgentMode } from '@/store/simplified-agent-mode'

import { SettingsView as ClassicSettingsView } from '../settings'
import { type ConfigSettingsDraft, ConfigSettingsDraftContext } from '../settings/config-draft'
import type { SettingsPageProps } from '../settings/types'

import { AgentInstructionsDraftContext, EMPTY_AGENT_INSTRUCTIONS_DRAFT } from './settings-general'

const AgentSettingsView = lazy(async () => ({ default: (await import('./index')).AgentSettingsView }))

/** An explicit agent route always opens the owner-pinned settings surface. */
export function SettingsView(props: SettingsPageProps) {
  const redesigned = useStore($simplifiedAgentMode)
  const { search } = useLocation()
  const params = new URLSearchParams(search)
  const profile = params.get('agentProfile')
  const connectionId = params.get('connection') || undefined

  if (!profile) {
    return <ClassicSettingsView {...props} />
  }

  return (
    <AgentSettingsRoute
      {...props}
      classic={!redesigned}
      connectionId={connectionId}
      key={JSON.stringify([connectionId, profile])}
      profile={profile}
    />
  )
}

interface AgentSettingsRouteProps extends SettingsPageProps {
  classic: boolean
  connectionId?: string
  profile: string
}

function AgentSettingsRoute({ connectionId, profile, ...props }: AgentSettingsRouteProps) {
  // Pin the ambient owner on entry, not again when a gateway or presentation changes.
  const [owner] = useState(() => ({
    connectionId: resolveOwnerNow({ connectionId, profile }).connectionId ?? $activeConnectionId.get() ?? undefined,
    profile
  }))
  const configDraft = useRef<ConfigSettingsDraft | null>(null)
  const instructionsDraft = useState(EMPTY_AGENT_INSTRUCTIONS_DRAFT)

  return (
    <ConfigSettingsDraftContext.Provider value={configDraft}>
      <AgentInstructionsDraftContext.Provider value={instructionsDraft}>
        <Suspense fallback={<PageLoader />}>
          <AgentSettingsView {...props} owner={owner} />
        </Suspense>
      </AgentInstructionsDraftContext.Provider>
    </ConfigSettingsDraftContext.Provider>
  )
}
