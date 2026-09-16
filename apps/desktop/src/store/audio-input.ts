import { atom } from 'nanostores'

import {
  type AudioInputDevice,
  listAudioInputDevices,
  openMicrophoneStream,
  SYSTEM_DEFAULT_AUDIO_INPUT
} from '@/lib/audio-input'
import { Codecs, persistentAtom } from '@/lib/persisted'

export { listAudioInputDevices, openMicrophoneStream, SYSTEM_DEFAULT_AUDIO_INPUT }

/** `null` = never chosen (ask). `default` = follow the OS input. Otherwise a deviceId. */
export const $audioInputDeviceId = persistentAtom<null | string>(
  'hermes.desktop.audioInputDeviceId',
  null,
  Codecs.nullableText
)

export function setAudioInputDeviceId(deviceId: null | string) {
  $audioInputDeviceId.set(deviceId)
}

interface AudioInputPickRequest {
  devices: AudioInputDevice[]
  resolve: (deviceId: null | string) => void
}

export const $audioInputPickRequest = atom<AudioInputPickRequest | null>(null)

export function requestAudioInputPick(devices: AudioInputDevice[]): Promise<null | string> {
  const previous = $audioInputPickRequest.get()

  previous?.resolve(null)

  return new Promise(resolve => {
    $audioInputPickRequest.set({ devices, resolve })
  })
}

export function settleAudioInputPick(deviceId: null | string) {
  const pending = $audioInputPickRequest.get()

  if (!pending) {
    return
  }

  $audioInputPickRequest.set(null)

  if (deviceId) {
    $audioInputDeviceId.set(deviceId)
  }

  pending.resolve(deviceId)
}

function savedDeviceStillExists(saved: null | string, devices: AudioInputDevice[]): boolean {
  if (saved === null) {
    return false
  }

  if (saved === SYSTEM_DEFAULT_AUDIO_INPUT) {
    return true
  }

  return devices.some(device => device.deviceId === saved)
}

/**
 * Ask once when this machine has several inputs and the user has not picked.
 * Cancel returns false so the voice session does not start on a silent default.
 */
export async function ensureAudioInputChoice(options?: { force?: boolean }): Promise<boolean> {
  const permitted = await window.hermesDesktop?.requestMicrophoneAccess?.()

  if (permitted === false) {
    return false
  }

  const devices = await listAudioInputDevices()
  const saved = $audioInputDeviceId.get()

  if (!options?.force && savedDeviceStillExists(saved, devices)) {
    return true
  }

  const unique = devices.filter(device => device.deviceId !== SYSTEM_DEFAULT_AUDIO_INPUT)

  if (!options?.force && unique.length <= 1) {
    $audioInputDeviceId.set(unique[0]?.deviceId ?? SYSTEM_DEFAULT_AUDIO_INPUT)

    return true
  }

  const picked = await requestAudioInputPick(devices.length > 0 ? devices : unique)

  return picked !== null
}
