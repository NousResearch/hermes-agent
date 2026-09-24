import { HERMES_CONFIG_KEY } from '@/app/hooks/use-config-record'
import { queryClient } from '@/lib/query-client'
import type { HermesConfigRecord } from '@/types/hermes'

/**
 * Microphone / speaker selection for the desktop (part of the local GUI patch).
 *
 * Two distinct namespaces exist in the config, and mixing them would be a bug:
 *
 *  - `wake_word.input_device` is a **PortAudio** index/name, consumed by the Python side
 *    (sounddevice) for wake-word capture.
 *  - `voice.mic_device_id` / `voice.speaker_device_id` (here) are **browser device ids** from
 *    `navigator.mediaDevices.enumerateDevices()`, consumed by the renderer's recorder, its barge-in
 *    analyser and its playback element. Empty or absent = the system default, which is also what
 *    happens today.
 *
 * The reads go through the React Query cache because playback and barge-in are plain modules, not
 * components — the same `queryClient.getQueryData(...)` pattern the kanban and slash-cache modules
 * use.
 */
export const VOICE_INPUT_DEVICE_KEY = 'voice.mic_device_id'
export const VOICE_OUTPUT_DEVICE_KEY = 'voice.speaker_device_id'

function readNested(record: unknown, path: string): string {
  let node: unknown = record

  for (const part of path.split('.')) {
    if (!node || typeof node !== 'object') {
      return ''
    }

    node = (node as Record<string, unknown>)[part]
  }

  return typeof node === 'string' ? node.trim() : ''
}

function configRecord(): unknown {
  try {
    return queryClient.getQueryData<HermesConfigRecord>(HERMES_CONFIG_KEY)
  } catch {
    return undefined
  }
}

/** Configured microphone id, or '' for the system default. */
export function voiceInputDeviceId(): string {
  return readNested(configRecord(), VOICE_INPUT_DEVICE_KEY)
}

/** Configured speaker id, or '' for the system default. */
export function voiceOutputDeviceId(): string {
  return readNested(configRecord(), VOICE_OUTPUT_DEVICE_KEY)
}

/**
 * `getUserMedia` audio constraints carrying the configured microphone.
 *
 * An `exact` constraint is deliberate: a silent fallback to the default microphone is the exact
 * behaviour this setting exists to prevent. Callers should catch `OverconstrainedError` (the saved
 * device is unplugged) and retry with the plain constraints.
 */
export function audioInputConstraints(base: MediaTrackConstraints = {}): MediaTrackConstraints {
  const id = voiceInputDeviceId()

  return id ? { ...base, deviceId: { exact: id } } : base
}

/** True when *error* means "the requested device is gone", not "you denied the mic". */
export function isMissingDeviceError(error: unknown): boolean {
  const name = (error as { name?: string } | null)?.name

  return name === 'OverconstrainedError' || name === 'NotFoundError'
}

/** Point a playback element at the configured speaker. Best-effort: failures keep the default. */
export async function applyAudioOutputDevice(audio: HTMLAudioElement): Promise<void> {
  const id = voiceOutputDeviceId()

  const setSink = (audio as HTMLAudioElement & { setSinkId?: (deviceId: string) => Promise<void> })
    .setSinkId

  if (!id || typeof setSink !== 'function') {
    return
  }

  try {
    await setSink.call(audio, id)
  } catch (error) {
    console.warn('[hermes] could not switch to the configured output device', error)
  }
}

export interface VoiceDeviceLists {
  inputs: MediaDeviceInfo[]
  outputs: MediaDeviceInfo[]
}

/** Enumerate audio devices; label-less entries are normal before the mic permission is granted. */
export async function listVoiceDevices(): Promise<VoiceDeviceLists> {
  const media = typeof navigator === 'undefined' ? undefined : navigator.mediaDevices

  if (!media?.enumerateDevices) {
    return { inputs: [], outputs: [] }
  }

  try {
    const devices = await media.enumerateDevices()

    return {
      inputs: devices.filter(device => device.kind === 'audioinput'),
      outputs: devices.filter(device => device.kind === 'audiooutput')
    }
  } catch {
    return { inputs: [], outputs: [] }
  }
}

/** Label for a device entry, falling back to its position when permissions hide labels. */
export function deviceLabel(device: MediaDeviceInfo, index: number, kind: 'input' | 'output'): string {
  const fallback = `${kind === 'input' ? 'Microphone' : 'Speaker'} ${index + 1}`

  return device.label?.trim() || fallback
}
