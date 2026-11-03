import { PanelEmpty } from '@/app/overlays/panel'
import { Button } from '@/components/ui/button'
import type { TeamGatewayInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { openExternalLink } from '@/lib/external-link'
import { ExternalLink } from '@/lib/icons'

const PLATFORM_NAMES: Record<string, string> = {
  discord: 'Discord',
  slack: 'Slack',
  telegram: 'Telegram',
  whatsapp: 'WhatsApp'
}

/**
 * The whole Messaging page when Team Gateway owns this agent's messaging. The gateway switches every
 * direct bot off at each start there, so per-platform setup would configure nothing: channels are
 * configured on the agent's page in Nous Portal.
 */
export function TeamGatewayMessaging({ teamGateway }: { teamGateway: TeamGatewayInfo }) {
  const { t } = useI18n()
  const g = t.messaging.teamGateway
  const carried = teamGateway.platforms.map(id => PLATFORM_NAMES[id] ?? id).join(', ')

  return (
    <div className="flex h-full min-h-0 flex-col" data-testid="team-gateway-messaging">
      <PanelEmpty
        action={
          <div className="flex flex-col items-center gap-3">
            {carried && <p className="text-xs text-muted-foreground">{g.carried(carried)}</p>}
            <Button onClick={() => openExternalLink(teamGateway.portal_url)} size="sm">
              <ExternalLink />
              {g.openPortal}
            </Button>
          </div>
        }
        description={g.body}
        icon="broadcast"
        title={g.title}
      />
    </div>
  )
}
