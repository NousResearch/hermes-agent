import { useStore } from '@nanostores/react'

import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { useI18n } from '@/i18n'
import { connectLinkedGateway } from '@/lib/connect-linked-gateway'
import { $activeConnectionId, selectConnection, setConnectionsRegistry } from '@/store/connections'
import { $gatewayConnectRequest, closeGatewayConnect } from '@/store/gateway-connect-request'

export function GatewayConnectDialog() {
  const request = useStore($gatewayConnectRequest)
  const { t } = useI18n()

  if (!request) {
    return null
  }

  return (
    <ConfirmDialog
      busyLabel={t.common.connecting}
      confirmLabel={t.common.connect}
      description={
        <>
          <span className="block">{request.name}</span>
          <span className="break-all">{request.url}</span>
        </>
      }
      doneLabel={t.connectors.connected}
      onClose={closeGatewayConnect}
      onConfirm={async () => {
        if (!window.hermesDesktop?.connections || !window.hermesDesktop.oauthLoginConnectionConfig) {
          throw new Error(t.boot.errors.ipcBridgeUnavailable)
        }

        await connectLinkedGateway(
          request,
          window.hermesDesktop,
          async (registry, id) => {
            setConnectionsRegistry(registry)
            await selectConnection(id)

            if ($activeConnectionId.get() !== id) {
              throw new Error(t.connectors.failed)
            }
          },
          t.boot.failure.signInIncompleteMessage
        )
      }}
      open
      title={t.settings.gateway.remoteTitle}
    />
  )
}
