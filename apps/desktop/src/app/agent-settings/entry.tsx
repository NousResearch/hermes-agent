import { useStore } from '@nanostores/react'
import { lazy, Suspense } from 'react'
import { useLocation } from 'react-router'

import { $simplifiedAgentMode } from '@/store/simplified-agent-mode'

import { SettingsView as ClassicSettingsView } from '../settings'
import type { SettingsPageProps } from '../settings/types'

const AgentSettingsView = lazy(async () => ({ default: (await import('./index')).AgentSettingsView }))

/** The existing settings route stays unchanged unless v2 explicitly names an agent. */
export function SettingsView(props: SettingsPageProps) {
  const enabled = useStore($simplifiedAgentMode)
  const { search } = useLocation()
  const params = new URLSearchParams(search)
  const profile = params.get('agent')
  const connectionId = params.get('connection') || undefined

  if (!enabled || !profile) {return <ClassicSettingsView {...props} />}

  return (
    <Suspense fallback={null}>
      <AgentSettingsView {...props} key={JSON.stringify([connectionId, profile])} owner={{ connectionId, profile }} />
    </Suspense>
  )
}
