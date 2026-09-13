import { useState } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { Switch } from '@/components/ui/switch'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import {
  type AgentPluginRow,
  type GatewayRequest,
  type PluginSetupReview,
  toggleAgentPlugin
} from '@/store/agent-plugins'

interface AgentPluginToggleProps {
  row: AgentPluginRow
  profile: string | null
  label: string
  busy: boolean
}

/** Setup consent belongs to this scoped row, never the recovering live socket. */
export function AgentPluginToggle({ row, profile, label, busy }: AgentPluginToggleProps) {
  const { t } = useI18n()
  const p = t.settings.plugins.agent
  const { requestGateway, gateway } = useGatewayRequest()

  const [setup, setSetup] = useState<{
    review: PluginSetupReview
    request: GatewayRequest
    profile: string | null
  } | null>(null)

  const key = row.key
  const failMessage = t.skills.plugins.toggleFailed(row.name)

  // Capture before requesting the proposal. Consent must never reconnect onto
  // a different backend, even if the active socket changes while reviewing.
  const request: GatewayRequest = gateway ? gateway.request.bind(gateway) : requestGateway

  return (
    <>
      <Switch
        aria-label={label}
        checked={row.status === 'enabled'}
        disabled={!key || busy || setup !== null}
        onCheckedChange={enable => {
          if (!key) {
            return
          }

          triggerHaptic('selection')
          void toggleAgentPlugin(request, key, enable, failMessage, profile, {
            onSetupRequired: review => setSetup({ review, request, profile })
          })
        }}
      />
      {setup && key && (
        <ConfirmDialog
          busyLabel={p.setupBusy}
          confirmLabel={p.setupConfirm}
          description={
            <span className="flex flex-col gap-2 break-words">
              <span>{setup.review.setup.summary}</span>
              <span>
                {setup.review.consent.key} · {setup.review.consent.hermes_home}
              </span>
              <span>{setup.review.setup.revision}</span>
              {setup.review.setup.details.map((detail, index) => (
                <span key={index}>{detail}</span>
              ))}
              <span>{p.setupTrust}</span>
            </span>
          }
          onClose={() => setSetup(null)}
          onConfirm={async () => {
            const ok = await toggleAgentPlugin(setup.request, key, true, failMessage, setup.profile, {
              setupConsent: setup.review.consent,
              onSetupRequired: review => setSetup({ ...setup, review }),
              throwOnError: true
            })

            if (!ok) {
              throw new Error(failMessage)
            }
          }}
          open
          title={p.setupTitle}
        />
      )}
    </>
  )
}
