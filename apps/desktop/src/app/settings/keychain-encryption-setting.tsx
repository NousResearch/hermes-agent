import { useEffect, useState } from 'react'

import { useI18n } from '@/i18n'
import { notifyError } from '@/store/notifications'

import { ToggleRow } from './primitives'
import { SETTING_IDS, settingElementId } from './settings-manifest'

export interface KeychainEncryption {
  busy: boolean
  on: boolean
  set: (on: boolean) => Promise<void>
  /** Both IPC halves exist; the browser host's bridge has neither. */
  supported: boolean
}

/**
 * Opt-in OS-keychain encryption for stored gateway secrets. Read lazily via
 * IPC (never touches the keychain); flipping it re-encodes stored secrets in
 * the main process and can legitimately prompt for keychain access.
 */
export function useKeychainEncryption(): KeychainEncryption {
  const { t } = useI18n()
  const g = t.settings.gateway
  const [keychainEncryption, setKeychainEncryptionState] = useState(false)
  const [keychainEncryptionBusy, setKeychainEncryptionBusy] = useState(false)

  useEffect(() => {
    let cancelled = false

    void window.hermesDesktop
      ?.getSecretStorageEncryption?.()
      .then(res => {
        if (!cancelled && res) {
          setKeychainEncryptionState(res.on === true)
        }
      })
      .catch(() => {})

    return () => {
      cancelled = true
    }
  }, [])

  const setKeychainEncryption = async (on: boolean) => {
    const setEncryption = window.hermesDesktop?.setSecretStorageEncryption

    if (!setEncryption) {
      return
    }

    setKeychainEncryptionBusy(true)
    // Optimistic paint; the IPC result (or a failure rollback) gets the last word.
    setKeychainEncryptionState(on)

    try {
      const res = await setEncryption(on)

      setKeychainEncryptionState(res?.on === true)
    } catch (err) {
      setKeychainEncryptionState(!on)
      notifyError(err, g.keychainEncryptionFailed)
    } finally {
      setKeychainEncryptionBusy(false)
    }
  }

  return {
    busy: keychainEncryptionBusy,
    on: keychainEncryption,
    set: setKeychainEncryption,
    supported:
      typeof window.hermesDesktop?.getSecretStorageEncryption === 'function' &&
      typeof window.hermesDesktop?.setSecretStorageEncryption === 'function'
  }
}

export function KeychainEncryptionSetting({ keychain }: { keychain: KeychainEncryption }) {
  const { t } = useI18n()
  const g = t.settings.gateway

  if (!keychain.supported) {
    return null
  }

  return (
    <ToggleRow
      checked={keychain.on}
      description={g.keychainEncryptionDesc}
      disabled={keychain.busy}
      id={settingElementId(SETTING_IDS.gateway.keychainEncryption)}
      label={g.keychainEncryptionTitle}
      onChange={on => void keychain.set(on)}
    />
  )
}
