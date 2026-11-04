import { Button } from '@/components/ui/button'
import type { MessagingPlatformInfo, TeamGatewayInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { openExternalLink } from '@/lib/external-link'
import { ExternalLink } from '@/lib/icons'

/**
 * Shown in place of the messaging platforms when Team Gateway owns this agent's messaging. The
 * gateway switches every direct messaging bot off at each start there, so their setup would
 * configure nothing: the agent's channels are configured in Nous Portal. The surfaces the rule
 * leaves running (API server, webhooks) stay listed below.
 */
export function TeamGatewayMessaging({
  platforms,
  teamGateway
}: {
  platforms: MessagingPlatformInfo[]
  teamGateway: TeamGatewayInfo
}) {
  const { t } = useI18n()
  const g = t.messaging.teamGateway
  const names = new Map(platforms.map(platform => [platform.id, platform.name]))
  const carried = teamGateway.platforms.map(id => names.get(id) ?? id).join(', ')

  return (
    <section
      className="flex flex-wrap items-start justify-between gap-3 border-b border-(--ui-stroke-secondary) px-4 py-4"
      data-testid="team-gateway-messaging"
    >
      <div className="min-w-0 max-w-2xl">
        <h3 className="text-[0.9375rem] font-semibold tracking-tight">{g.title}</h3>
        <p className="mt-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
          {g.body}
        </p>
        {carried && <p className="mt-1 text-xs text-muted-foreground">{g.carried(carried)}</p>}
      </div>
      <Button onClick={() => openExternalLink(teamGateway.portal_url)} size="sm">
        <ExternalLink />
        {g.openPortal}
      </Button>
    </section>
  )
}
