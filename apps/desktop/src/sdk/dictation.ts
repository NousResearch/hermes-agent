// `useDictation`: the composer's voice dictation for a plugin's own text field.
// Same recorder as the chat mic (warm-up lease, live stream, recording cap, the
// toasts for a denied mic or an empty take), and the same transcription ladder
// as a session tile: the profile's own STT provider called directly when it is
// client-callable, the gateway relay otherwise. Outside a composer scope the
// owner is the active connection and profile, resolved when the mic opens.

import { useVoiceRecorder } from '@/app/chat/composer/hooks/use-voice-recorder'
import type { VoiceStatus } from '@/app/chat/composer/types'
import { blobToDataUrl } from '@/app/session/hooks/use-prompt-actions/utils'
import { type ResolvedOwner, transcribeAudio } from '@/hermes'
import { transcribeAudioClientDirect } from '@/lib/voice-client-direct'

export type DictationStatus = VoiceStatus

export interface DictationOptions {
  /** The settled transcript of one take (never empty). */
  onTranscript: (text: string) => void
  /** Called once the take settles, to hand focus back to the field. */
  focusInput?: () => void
  /** Recording cap in seconds (clamped to 1–600). */
  maxRecordingSeconds?: number
}

async function transcribeDictation(audio: Blob, owner?: ResolvedOwner): Promise<string> {
  const direct = await transcribeAudioClientDirect(audio, owner)

  if (direct !== null) {
    return direct
  }

  return (await transcribeAudio(await blobToDataUrl(audio), audio.type, owner)).transcript
}

const noop = () => {}

export function useDictation({ focusInput = noop, maxRecordingSeconds = 120, onTranscript }: DictationOptions) {
  const { dictate, voiceStatus } = useVoiceRecorder({
    focusInput,
    maxRecordingSeconds,
    onTranscribeAudio: transcribeDictation,
    onTranscript
  })

  /** Start a take, or stop the running one and transcribe it. */
  return { status: voiceStatus as DictationStatus, toggle: dictate }
}
