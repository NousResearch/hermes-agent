import { useStore } from '@nanostores/react'
import { lazy, Suspense } from 'react'
import { useLocation } from 'react-router'

import { PageLoader } from '@/components/page-loader'
import { $simplifiedAgentMode } from '@/store/simplified-agent-mode'

import { SettingsView as ClassicSettingsView } from '../settings'
import type { SettingsPageProps } from '../settings/types'

const AgentSettingsView = lazy(async () => ({ default: (await import('./index')).AgentSettingsView }))

/** An explicit agent route always opens the owner-pinned settings surface. */
export function SettingsView(props: SettingsPageProps) {
  const redesigned = useStore($simplifiedAgentMode)
  const { search } = useLocation()
  const params = new URLSearchParams(search)
  const profile = params.get('agent')
  const connectionId = params.get('connection') || undefined

  if (!profile) {
    return <ClassicSettingsView {...props} />
  }

  return (
    <Suspense fallback={<PageLoader />}>
      <AgentSettingsView
        classic={!redesigned}
        {...props}
        key={JSON.stringify([connectionId, profile])}
        owner={{ connectionId, profile }}
      />
    </Suspense>
  )
}
