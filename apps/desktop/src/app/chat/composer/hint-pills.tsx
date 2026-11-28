import { ModelPreloadHint } from './model-preload-hint'
import { SendTargetHint } from './send-target-hint'

/** The informational pills that sit above the composer, in the dock's strip. */
export function ComposerHintPills() {
  return (
    <>
      <SendTargetHint />
      <ModelPreloadHint />
    </>
  )
}
