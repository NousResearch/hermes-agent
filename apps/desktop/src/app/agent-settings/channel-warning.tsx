import type { MessagingPlatformsResponse } from '@/hermes'
import { AlertTriangle } from '@/lib/icons'

import { useOverviewActionsCopy } from './overview-actions-copy'

export function AgentChannelWarnings({ messaging }: { messaging?: MessagingPlatformsResponse }) {
  const c = useOverviewActionsCopy()

  const disconnected =
    messaging?.platforms.filter(
      platform =>
        platform.configured && platform.enabled && ['disconnected', 'error', 'failed'].includes(platform.state ?? '')
    ) ?? []

  return disconnected.map(platform => (
    <div className="agent-channel-warning" key={platform.id} role="alert">
      <AlertTriangle aria-hidden />
      <div>
        <p>{c.disconnectedTitle.replace('{name}', platform.name)}</p>
        <p>{c.disconnectedHelp}</p>
      </div>
    </div>
  ))
}
