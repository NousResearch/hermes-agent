import type { MessagingPlatformsResponse } from '@/hermes'
import { AlertTriangle } from '@/lib/icons'
import { platformStatusTone } from '@/lib/platform-status'

import { useOverviewActionsCopy } from './overview-actions-copy'

export function AgentChannelWarnings({ messaging }: { messaging?: MessagingPlatformsResponse }) {
  const c = useOverviewActionsCopy()

  const disconnected =
    messaging?.platforms.filter(
      platform => platform.configured && platform.enabled && ['bad', 'warn'].includes(platformStatusTone(platform))
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
