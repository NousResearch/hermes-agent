/** Browser audio-input helpers. Device IDs are origin-stable after permission. */

export const SYSTEM_DEFAULT_AUDIO_INPUT = 'default'

export interface AudioInputDevice {
  deviceId: string
  groupId: string
  label: string
}

const VIRTUAL_LABEL = /\((?:virtual|loopback)\)|blackhole|soundflower|vb-?audio|cable input|camo microphone/i

/** True when the OS label is a mixer/loopback that is often silent. */
export function isVirtualAudioLabel(label: string): boolean {
  return VIRTUAL_LABEL.test(label)
}

export function audioCaptureConstraints(
  deviceId?: null | string,
  extra?: MediaTrackConstraints
): MediaTrackConstraints {
  const audio: MediaTrackConstraints = {
    echoCancellation: true,
    noiseSuppression: true,
    ...extra
  }

  // Applied after `extra` so a caller cannot accidentally unpin the device.
  if (deviceId && deviceId !== SYSTEM_DEFAULT_AUDIO_INPUT) {
    audio.deviceId = { exact: deviceId }
  }

  return audio
}

export function displayAudioInputLabel(
  deviceId: null | string,
  devices: readonly AudioInputDevice[],
  systemDefaultLabel: string
): string {
  if (!deviceId || deviceId === SYSTEM_DEFAULT_AUDIO_INPUT) {
    const namedDefault = devices.find(device => device.deviceId === 'default')

    return namedDefault?.label || systemDefaultLabel
  }

  return devices.find(device => device.deviceId === deviceId)?.label || systemDefaultLabel
}

export async function listAudioInputDevices(): Promise<AudioInputDevice[]> {
  if (!navigator.mediaDevices?.enumerateDevices) {
    return []
  }

  const devices = await navigator.mediaDevices.enumerateDevices()

  return devices
    .filter(device => device.kind === 'audioinput')
    .map(device => ({
      deviceId: device.deviceId,
      groupId: device.groupId,
      label: device.label.trim() || device.deviceId.slice(0, 8)
    }))
}

export async function openMicrophoneStream(
  deviceId?: null | string,
  extra?: MediaTrackConstraints
): Promise<MediaStream> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('getUserMedia unavailable')
  }

  try {
    return await navigator.mediaDevices.getUserMedia({
      audio: audioCaptureConstraints(deviceId, extra)
    })
  } catch (error) {
    if (deviceId && deviceId !== SYSTEM_DEFAULT_AUDIO_INPUT) {
      return await navigator.mediaDevices.getUserMedia({
        audio: audioCaptureConstraints(SYSTEM_DEFAULT_AUDIO_INPUT, extra)
      })
    }

    throw error
  }
}
