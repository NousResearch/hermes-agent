import { LOCAL_CONNECTION_ID } from '@hermes/shared'
import { useRef, useState } from 'react'

import { FirstRunRemoteSetup } from '@/components/remote-setup/first-run'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { useI18n } from '@/i18n'
import { notifyError } from '@/store/notifications'

/**
 * Resolve whether This device would install, without starting the installer.
 * Missing bridge fails closed: a click must not reach runBootstrap unseen.
 */
async function probeLocalBackend(): Promise<boolean> {
  const probe = window.hermesDesktop?.probeLocalBackend

  if (!probe) {
    return true
  }

  return (await probe()).bootstrapNeeded === true
}

export interface LocalDeviceTarget {
  connectionId: string
  label: string
}

/**
 * Gate picking This device on Hermes actually being installed here. Only a
 * click that would start the installer confirms, and it offers Connect to
 * existing inline so the click cannot fall through into runBootstrap. An
 * installed runtime is just another connected gateway: no prompt.
 */
export function useLocalDeviceSwitch() {
  const { t } = useI18n()
  const [prompt, setPrompt] = useState<LocalDeviceTarget | null>(null)
  const [connectOpen, setConnectOpen] = useState(false)
  const waiter = useRef<((accepted: boolean) => void) | null>(null)

  const settle = (accepted: boolean) => {
    const resolve = waiter.current
    waiter.current = null
    setPrompt(null)
    resolve?.(accepted)
  }

  const request = async (target: LocalDeviceTarget): Promise<boolean> => {
    if (target.connectionId !== LOCAL_CONNECTION_ID) {
      return true
    }

    if (waiter.current) {
      return false
    }

    let bootstrapNeeded: boolean

    try {
      bootstrapNeeded = await probeLocalBackend()
    } catch (error) {
      notifyError(error, t.profiles.switchConnectionFailed(target.label))

      return false
    }

    if (!bootstrapNeeded) {
      return true
    }

    return new Promise(resolve => {
      waiter.current = resolve
      setPrompt(target)
    })
  }

  const fleet = t.profiles.fleet

  const dialog = (
    <>
      {prompt ? (
        <ConfirmDialog
          confirmLabel={fleet.installDeviceConfirm}
          description={fleet.installDeviceDesc}
          dismissOnConfirm
          onClose={() => settle(false)}
          onConfirm={() => settle(true)}
          open
          secondaryAction={{
            label: fleet.connectExistingInstead,
            onClick: () => {
              settle(false)
              setConnectOpen(true)
            }
          }}
          title={fleet.installDeviceTitle}
        />
      ) : null}
      {connectOpen ? <FirstRunRemoteSetup onBack={() => setConnectOpen(false)} /> : null}
    </>
  )

  return { dialog, request }
}
