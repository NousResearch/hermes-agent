import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { requestComposerSubmit } from '@/app/chat/composer/focus'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { connectorTitle } from '@/lib/connector-tools'
import {
  $firstBuildConnections,
  type FirstBuildConnectorPart,
  openFirstBuildLinks,
  watchFirstBuildWait
} from '@/store/first-build-connectors'
import { requestGatewayForAgent } from '@/store/gateway'

interface FirstBuildConnectorOfferProps {
  part: FirstBuildConnectorPart
  storedId: string
  runtimeId: string
  connectionId: string | null
  profile: string
  target: string
}

export function FirstBuildConnectorOffer({
  part,
  storedId,
  runtimeId,
  connectionId,
  profile,
  target
}: FirstBuildConnectorOfferProps) {
  const connections = useStore($firstBuildConnections, { keys: [storedId] })
  const { t } = useI18n()
  const { toolCallId, toolName, args, result } = part

  useEffect(() => {
    void openFirstBuildLinks(
      storedId,
      { toolCallId, toolName, args, result },
      {
        open: window.hermesDesktop?.openExternal ? url => window.hermesDesktop.openExternal(url) : undefined,
        submit: text => void requestComposerSubmit(text, { displayKind: 'hidden', target })
      }
    )
  }, [storedId, toolCallId, toolName, args, result, target])

  useEffect(
    () =>
      watchFirstBuildWait(storedId, runtimeId, { toolCallId, toolName, args, result }, (method, params) =>
        requestGatewayForAgent(connectionId, profile, method, params, 45000)
      ),
    [storedId, runtimeId, toolCallId, toolName, args, result, connectionId, profile]
  )

  return (
    <div className="my-2 grid min-w-0 max-w-lg gap-3" data-connector-offer>
      {connections[storedId]?.rows.map(row => (
        <div className="flex items-center justify-between gap-3 text-sm" key={row.connector}>
          <span>{connectorTitle(row.connector)}</span>
          <span aria-live="polite" className="text-xs text-muted-foreground">
            {row.phase === 'connected' ? (
              <>
                <span aria-hidden>✓ </span>
                {t.connectors.connected}
              </>
            ) : row.phase === 'timeout' ? (
              t.connectors.notConnected
            ) : row.phase === 'error' ? (
              row.error === 'unavailable' ? (
                t.connectors.notAvailable
              ) : (
                t.connectors.statusError
              )
            ) : (
              t.connectors.waitingSignIn
            )}
          </span>
          {row.connectUrl && row.phase !== 'connected' && !window.hermesDesktop?.openExternal ? (
            <Button asChild size="xs" variant="link">
              <a href={row.connectUrl} rel="noreferrer" target="_blank">
                {t.connectors.connect}
              </a>
            </Button>
          ) : null}
        </div>
      ))}
    </div>
  )
}
