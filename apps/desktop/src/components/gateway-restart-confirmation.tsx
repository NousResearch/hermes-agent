import { useStore } from '@nanostores/react'

import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { useI18n } from '@/i18n'
import {
  $gatewayRestartConfirmationOpen,
  closeGatewayRestartConfirmation,
  runConfirmedGatewayRestart
} from '@/store/system-actions'

export function GatewayRestartConfirmation() {
  const open = useStore($gatewayRestartConfirmationOpen)
  const { t } = useI18n()

  return (
    <ConfirmDialog
      description="Type RESTART to restart the Hermes gateway. Connected channels and active sessions will reconnect afterward."
      dismissOnConfirm
      onClose={closeGatewayRestartConfirmation}
      onConfirm={async () => {
        await runConfirmedGatewayRestart()
        closeGatewayRestartConfirmation(true)
      }}
      open={open}
      title={t.commandCenter.restartGateway}
      typedConfirmation="RESTART"
    />
  )
}
