import { useLayoutEffect, useRef } from 'react'

import { type DictationHandle, registerComposerDictation } from '@/lib/composer-dictation'

import type { ComposerTarget } from '../focus'
import { useComposerScope } from '../scope'

import type { useVoiceRecorder } from './use-voice-recorder'

interface Options {
  target: ComposerTarget
  sessionId: string | null | undefined
  disabled: boolean
  recorder: ReturnType<typeof useVoiceRecorder>
}

/** Routes a plugin comment through the existing recorder without writing into the chat draft. */
export function usePluginDictation(options: Options): void {
  const latest = useRef(options)
  latest.current = options
  const { connectionId, profile } = useComposerScope()

  useLayoutEffect(() => {
    let active: DictationHandle | null = null

    const unregister = registerComposerDictation(options.target, recipient => {
      const current = latest.current

      if (active || current.disabled || current.recorder.voiceStatus !== 'idle') {return null}

      const recorder = current.recorder

      const handle: DictationHandle = {
        stop: () => {
          if (active === handle) {void recorder.stop()}
        },
        cancel: () => {
          if (active !== handle) {return}
          active = null
          recorder.cancel()
          recipient.onState('idle')
        }
      }

      active = handle
      recipient.onState('starting')
      void recorder.start({
        focusInput: () => undefined,
        onTranscript: text => {
          if (active === handle) {recipient.onTranscript(text)}
        },
        onState: status => {
          if (active !== handle) {return}

          if (status === 'idle') {active = null}
          recipient.onState(status)
        }
      })

      return handle
    })

    return () => {
      unregister()
      active?.cancel()
    }
  }, [options.target, options.sessionId, connectionId, profile])
}
