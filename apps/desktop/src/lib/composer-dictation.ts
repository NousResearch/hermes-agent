import type { ComposerTarget } from '@/app/chat/composer/focus'

export type DictationStatus = 'starting' | 'recording' | 'transcribing' | 'idle'

export interface DictationRecipient {
  onTranscript: (text: string) => void
  onState: (status: DictationStatus) => void
}

export interface DictationHandle {
  stop: () => void
  cancel: () => void
}

type StartDictation = (recipient: DictationRecipient) => DictationHandle | null
const surfaces = new Map<ComposerTarget, StartDictation>()

/** One visible composer owns capture, provider configuration and microphone lifecycle. */
export function registerComposerDictation(target: ComposerTarget, start: StartDictation): () => void {
  surfaces.set(target, start)

  return () => {
    if (surfaces.get(target) === start) {surfaces.delete(target)}
  }
}

export function startComposerDictation(target: ComposerTarget, recipient: DictationRecipient): DictationHandle | null {
  return surfaces.get(target)?.(recipient) ?? null
}
