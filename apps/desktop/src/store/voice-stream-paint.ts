import { atom } from 'nanostores'

// Voice replies still consume the frame-published transcript. This is streaming
// demand only: enabling voice must not disable normal idle-window throttling.
export const $voiceStreamPaintOwners = atom(0)

export function retainVoiceStreamPaint(): () => void {
  $voiceStreamPaintOwners.set($voiceStreamPaintOwners.get() + 1)

  return () => {
    $voiceStreamPaintOwners.set($voiceStreamPaintOwners.get() - 1)
  }
}
