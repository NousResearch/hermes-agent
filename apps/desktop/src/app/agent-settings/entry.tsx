import { useStore } from '@nanostores/react'
import { lazy, Suspense, useState } from 'react'
import { useLocation } from 'react-router'

import { resolveOwnerNow } from '@/api/client'
import { PageLoader } from '@/components/page-loader'
import { $activeConnectionId } from '@/store/connections'
import { $simplifiedAgentMode } from '@/store/simplified-agent-mode'

import { SettingsView as ClassicSettingsView } from '../settings'
import type { SettingsPageProps } from '../settings/types'

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

  return (
    <Suspense fallback={<PageLoader />}>
      <AgentSettingsView {...props} owner={owner} />
    </Suspense>
  )
}
