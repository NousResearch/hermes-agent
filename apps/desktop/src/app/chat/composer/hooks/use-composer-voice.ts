import { useStore } from '@nanostores/react'
import { createContext, createElement, type ReactNode, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { chatMessageText, collectUnspokenTurnSpeech } from '@/lib/chat-messages'
import { triggerHaptic } from '@/lib/haptics'
import { markAssistantIdSpoken, resolveSpokenReply } from '@/lib/spoken-reply'
import { CONVERSATION_LEASE, READ_ALOUD_LEASE, syncTtsLease } from '@/lib/tts-lease'
import { clearWakeIndicator, syncWakeIndicatorWithVoice } from '@/lib/wake-indicator'
import { $voiceConversationStartRequest, takeVoiceConversationStart } from '@/store/composer'
import { resetBrowseState } from '@/store/composer-input-history'
import { $gateway } from '@/store/gateway'
import { notify, notifyError } from '@/store/notifications'
import { $autoSpeakReplies, $voiceStopPhrase, setAutoSpeakReplies } from '@/store/voice-prefs'
import { resumeWakeAfterVoice } from '@/store/wake-word'

import type { ComposerTarget } from '../focus'
import { onComposerVoiceToggleRequest } from '../focus'
import { useComposerScope } from '../scope'
import type { ChatBarProps } from '../types'

import { useAutoSpeakReplies } from './use-auto-speak-replies'
import { useVoiceConversation } from './use-voice-conversation'
import { useVoiceRecorder } from './use-voice-recorder'

interface UseComposerVoiceArgs {
  busy: boolean
  clearDraft: () => void
  disabled: boolean
  focusInput: () => void
  insertText: (text: string) => void
  maxRecordingSeconds: number
  /** Interrupt the in-flight agent turn (Stop-button seam) — fired when the
   *  user speaks over the model while it is still generating. */
  onInterrupt?: () => Promise<void> | void
  onSubmit: ChatBarProps['onSubmit']
  onTranscribeAudio: ChatBarProps['onTranscribeAudio']
  sessionId: string | null | undefined
  /** This composer's focus-bus key — voice toggles targeting another
   *  composer (or the active one, when not us) are ignored. */
  target: ComposerTarget
}

export interface ComposerVoiceAssistant {
  id: string
  pending: boolean
  text: string
}

export interface ComposerVoiceLease {
  release: () => void
}

export interface ComposerVoiceController {
  acquire: (signal?: AbortSignal) => Promise<ComposerVoiceLease | null>
  interrupt: () => boolean
  latestAssistant: () => ComposerVoiceAssistant | null
  submitText: (text: string) => boolean
  subscribeAssistant: (listener: (assistant: ComposerVoiceAssistant | null) => void) => () => void
}

export function disposeAssistantSubscriptions(disposers: Set<() => void>): void {
  for (const dispose of disposers) {
    dispose()
  }

  disposers.clear()
}

const ComposerVoiceControllerContext = createContext<ComposerVoiceController | null>(null)

export function useComposerVoiceController(): ComposerVoiceController | null {
  return useContext(ComposerVoiceControllerContext)
}

export function ComposerVoiceControllerProvider({
  children,
  controller
}: {
  children: ReactNode
  controller: ComposerVoiceController
}) {
  return createElement(ComposerVoiceControllerContext.Provider, { value: controller }, children)
}

let microphoneOwner: symbol | null = null

async function waitForPause(signal: AbortSignal | undefined, pause: () => Promise<void>): Promise<boolean> {
  if (signal?.aborted) {
    return false
  }

  try {
    // Abort cancels eligibility, not the device-release barrier. The owner must
    // remain held until pause settles so cleanup can never resume wake early.
    await pause()

    return !signal?.aborted
  } catch {
    return false
  }
}

export async function acquireMicrophoneLease({
  voiceContextIsCurrent,
  owner,
  pause,
  resume,
  signal
}: {
  voiceContextIsCurrent: () => boolean
  owner: symbol
  pause: () => Promise<void>
  resume: () => void
  signal?: AbortSignal
}): Promise<ComposerVoiceLease | null> {
  if (!voiceContextIsCurrent() || signal?.aborted || microphoneOwner !== null) {
    return null
  }

  microphoneOwner = owner
  const paused = await waitForPause(signal, pause)

  if (!paused || !voiceContextIsCurrent() || signal?.aborted || microphoneOwner !== owner) {
    if (microphoneOwner === owner) {
      microphoneOwner = null
      resume()
    }

    return null
  }

  let released = false

  return {
    release: () => {
      if (released) {
        return
      }

      released = true

      if (microphoneOwner === owner) {
        microphoneOwner = null
        resume()
      }
    }
  }
}

export async function runVoiceControllerCallback(
  isCurrent: () => boolean,
  callback: () => Promise<unknown> | unknown
): Promise<void> {
  if (!isCurrent()) {
    return
  }

  await callback()

  if (!isCurrent()) {
    return
  }
}

export function useComposerVoice({
  busy,
  clearDraft,
  disabled,
  focusInput,
  insertText,
  maxRecordingSeconds,
  onInterrupt,
  onSubmit,
  onTranscribeAudio,
  sessionId,
  target
}: UseComposerVoiceArgs) {
  const { t } = useI18n()
  // A tile's composer speaks ITS transcript, not the primary chat's.
  const { $messages } = useComposerScope()
  const [activeVoiceContextEpoch, setActiveVoiceContextEpoch] = useState<number | null>(null)
  const ownsWakeIndicatorRef = useRef(false)
  const busyRef = useRef(busy)
  busyRef.current = busy
  const disabledRef = useRef(disabled)
  disabledRef.current = disabled
  const voiceContextEpochRef = useRef(0)
  const voiceContextIdentityRef = useRef({ disabled, sessionId, target })

  if (
    voiceContextIdentityRef.current.disabled !== disabled ||
    voiceContextIdentityRef.current.sessionId !== sessionId ||
    voiceContextIdentityRef.current.target !== target
  ) {
    voiceContextEpochRef.current += 1
    voiceContextIdentityRef.current = { disabled, sessionId, target }
  }

  const voiceContextEpoch = voiceContextEpochRef.current

  const voiceContextIsCurrent = useCallback(
    () => !disabledRef.current && voiceContextEpochRef.current === voiceContextEpoch,
    [voiceContextEpoch]
  )

  const ownerRef = useRef<{ epoch: number; token: symbol }>({
    epoch: voiceContextEpoch,
    token: Symbol('composer-voice-controller')
  })

  if (ownerRef.current.epoch !== voiceContextEpoch) {
    ownerRef.current = { epoch: voiceContextEpoch, token: Symbol('composer-voice-controller') }
  }

  const owner = ownerRef.current.token
  const voiceConversationActive = activeVoiceContextEpoch === voiceContextEpoch && voiceContextIsCurrent()
  const voiceStartRequest = useStore($voiceConversationStartRequest)

  const { dictate, voiceActivityState, voiceStatus } = useVoiceRecorder({
    focusInput,
    maxRecordingSeconds,
    onTranscript: insertText,
    onTranscribeAudio
  })

  /** Auto-speak selector: the latest unspoken reply only — a backlog collapses to the newest. */
  const pendingResponse = () => {
    const messages = $messages.get()
    const last = messages.findLast(m => m.role === 'assistant' && !m.hidden)
    const spoken = resolveSpokenReply(sessionId, messages)

    if (!last || last.id === spoken?.id) {
      return null
    }

    const text = chatMessageText(last).trim()

    if (!text) {
      return null
    }

    return {
      id: last.id,
      pending: Boolean(last.pending),
      text
    }
  }

  /**
   * Voice-conversation selector: every unspoken assistant bubble of the turn,
   * in order — narration interims AND the final answer, not just whichever
   * bubble happens to be last. See `collectUnspokenTurnSpeech`.
   */
  const pendingTurnResponse = () => {
    const messages = $messages.get()

    return collectUnspokenTurnSpeech(messages, resolveSpokenReply(sessionId, messages)?.id ?? null)
  }

  const consumePendingResponse = () => {
    const messages = $messages.get()
    const last = messages.findLast(m => m.role === 'assistant' && !m.hidden)

    if (last) {
      markAssistantIdSpoken(sessionId, messages, last.id)
    }
  }

  const submitVoiceTurn = async (text: string) => {
    if (!voiceContextIsCurrent() || busyRef.current) {
      return
    }

    triggerHaptic('submit')
    resetBrowseState(sessionId)
    clearDraft()
    await runVoiceControllerCallback(voiceContextIsCurrent, () => onSubmit(text))
  }

  const interruptVoiceTurn = useCallback(async () => {
    await runVoiceControllerCallback(voiceContextIsCurrent, () => onInterrupt?.())
  }, [voiceContextIsCurrent, onInterrupt])

  const wakePausedRef = useRef(false)
  // Resolves once the in-flight wake.pause round-trip completes (mic released by
  // the wake listener). The conversation awaits this before opening its own mic
  // so the two never contend for the device — on Windows especially, opening the
  // capture device while the wake listener still holds it makes getUserMedia
  // fail and the conversation never starts listening.
  const wakePauseBarrierRef = useRef<Promise<void> | null>(null)
  const wakeResumeScheduledRef = useRef<Promise<void> | null>(null)
  const assistantSubscriptionDisposersRef = useRef(new Set<() => void>())

  const conversation = useVoiceConversation({
    busy,
    consumePendingResponse,
    enabled: voiceConversationActive,
    onFatalError: () => {
      if (voiceContextIsCurrent()) {
        setActiveVoiceContextEpoch(null)
      }
    },
    // Speaking over the model mid-generation interrupts the in-flight turn —
    // the same seam as the Stop button — so the interjection becomes the next
    // turn instead of waiting behind a reply the user already rejected.
    onInterrupt: interruptVoiceTurn,
    // A spoken stop command ("stop", "never mind", "goodbye", …) ends the
    // hands-free conversation. Flipping the flag is the authoritative off
    // switch — the enabled=false prop + effect below drive conversation.end()
    // teardown (mic close, wake re-arm).
    onStopWord: () => {
      if (voiceContextIsCurrent()) {
        setActiveVoiceContextEpoch(null)
      }
    },
    onSubmit: submitVoiceTurn,
    onTranscribeAudio,
    pendingResponse: pendingTurnResponse,
    // Before the conversation opens the mic, wait for any in-flight wake.pause
    // to finish releasing the capture device (see wakePauseBarrierRef).
    beforeMicOpen: () => wakePauseBarrierRef.current ?? undefined
  })

  // eslint-disable-next-line no-restricted-syntax -- ownership token used only by unmount cleanup
  useEffect(() => {
    if (target !== 'main') {
      return
    }

    if (syncWakeIndicatorWithVoice(voiceConversationActive, conversation.status)) {
      ownsWakeIndicatorRef.current = voiceConversationActive
    }
  }, [conversation.status, target, voiceConversationActive])

  useEffect(
    () => () => {
      if (ownsWakeIndicatorRef.current) {
        clearWakeIndicator()
      }
    },
    []
  )

  // The `composer.voice` hotkey (Ctrl+B) toggles the conversation. Starting
  // with STT unconfigured lets the conversation surface its own "configure
  // speech-to-text" notice rather than silently no-opping.
  const toggleVoiceConversation = useCallback(() => {
    if (disabled) {
      return
    }

    if (voiceConversationActive) {
      setActiveVoiceContextEpoch(null)
      void conversation.end()

      return
    }

    if (microphoneOwner === null) {
      setActiveVoiceContextEpoch(voiceContextEpoch)
    }
  }, [conversation, disabled, voiceContextEpoch, voiceConversationActive])

  useEffect(
    () => onComposerVoiceToggleRequest(toggled => toggled === target && toggleVoiceConversation()),
    [target, toggleVoiceConversation]
  )

  useEffect(() => {
    if (
      target === 'main' &&
      !disabled &&
      takeVoiceConversationStart(voiceStartRequest) &&
      !voiceConversationActive
    ) {
      if (microphoneOwner === null) {
        setActiveVoiceContextEpoch(voiceContextEpoch)
      }
    }
  }, [disabled, target, voiceContextEpoch, voiceConversationActive, voiceStartRequest])

  const resumeWakeIfPaused = useCallback(() => {
    const barrier = wakePauseBarrierRef.current

    if (!wakePausedRef.current && !barrier) {
      return
    }

    if (barrier && wakeResumeScheduledRef.current === barrier) {
      return
    }

    wakePausedRef.current = false
    wakeResumeScheduledRef.current = barrier

    const resume = () => {
      if (wakePauseBarrierRef.current !== barrier) {
        return
      }

      wakePauseBarrierRef.current = null
      wakeResumeScheduledRef.current = null
      // Reconcile, don't just resume: the wake word is a persistent setting, so
      // ending a voice chat must re-arm the listener whenever config says
      // enabled — including when the raw resume loses the mic-release race.
      void resumeWakeAfterVoice()
    }

    if (barrier) {
      void barrier.then(resume, resume)
    } else {
      resume()
    }
  }, [])

  // The ref is a request token (did WE issue wake.pause?), not an atom mirror —
  // it guards resumeWakeIfPaused from resuming a detector another surface owns.
  const pauseWakeForVoice = useCallback(() => {
    wakePausedRef.current = true

    const barrier = (async () => {
      try {
        await $gateway.get()?.request('wake.pause', {})
      } catch {
        // No wake listener / older backend — nothing held the mic.
      }
    })()

    wakePauseBarrierRef.current = barrier

    return barrier
  }, [])

  const latestAssistant = useCallback((): ComposerVoiceAssistant | null => {
    if (!voiceContextIsCurrent()) {
      return null
    }

    const last = $messages.get().findLast(message => message.role === 'assistant' && !message.hidden)

    return last ? { id: last.id, pending: Boolean(last.pending), text: chatMessageText(last).trim() } : null
  }, [$messages, voiceContextIsCurrent])

  const submitText = useCallback(
    (text: string): boolean => {
      if (!voiceContextIsCurrent() || busyRef.current || !text.trim()) {
        return false
      }

      triggerHaptic('submit')
      resetBrowseState(sessionId)
      clearDraft()
      void runVoiceControllerCallback(voiceContextIsCurrent, () => onSubmit(text))

      return true
    },
    [clearDraft, onSubmit, sessionId, voiceContextIsCurrent]
  )

  const interrupt = useCallback((): boolean => {
    if (!voiceContextIsCurrent() || !onInterrupt) {
      return false
    }

    void runVoiceControllerCallback(voiceContextIsCurrent, onInterrupt)

    return true
  }, [onInterrupt, voiceContextIsCurrent])

  const voiceController = useMemo<ComposerVoiceController>(
    () => ({
      acquire: (signal?: AbortSignal) =>
        acquireMicrophoneLease({
          owner,
          pause: pauseWakeForVoice,
          resume: resumeWakeIfPaused,
          signal,
          voiceContextIsCurrent: () => voiceContextIsCurrent() && !voiceConversationActive
        }),
      interrupt,
      latestAssistant,
      submitText,
      subscribeAssistant: listener => {
        if (!voiceContextIsCurrent()) {
          return () => undefined
        }

        const unsubscribe = $messages.subscribe(() => {
          if (voiceContextIsCurrent()) {
            listener(latestAssistant())
          }
        })

        const dispose = () => {
          unsubscribe()
          assistantSubscriptionDisposersRef.current.delete(dispose)
        }

        assistantSubscriptionDisposersRef.current.add(dispose)

        return dispose
      }
    }),
    [
      $messages,
      interrupt,
      latestAssistant,
      owner,
      pauseWakeForVoice,
      resumeWakeIfPaused,
      submitText,
      voiceContextIsCurrent,
      voiceConversationActive
    ]
  )

  useEffect(
    () => () => {
      disposeAssistantSubscriptions(assistantSubscriptionDisposersRef.current)
    },
    [voiceContextEpoch]
  )

  useEffect(
    () => () => {
      if (microphoneOwner === owner) {
        microphoneOwner = null
        resumeWakeIfPaused()
      }
    },
    [owner, resumeWakeIfPaused]
  )

  useEffect(() => {
    if (voiceConversationActive) {
      pauseWakeForVoice()
    } else {
      resumeWakeIfPaused()
    }
  }, [pauseWakeForVoice, resumeWakeIfPaused, voiceConversationActive])

  // 'Say "stop" to end the voice chat.' notice when the conversation starts.
  // Phrase comes from voice.stop_phrases (first entry) so a custom phrase
  // renders correctly; a null phrase (stop_phrases: []) shows no notice.
  useEffect(() => {
    if (!voiceConversationActive) {
      return
    }

    const phrase = $voiceStopPhrase.get()

    if (phrase) {
      notify({
        id: 'voice-stop-hint',
        kind: 'info',
        icon: 'mic',
        message: t.notifications.voice.sayStopToEnd(phrase)
      })
    }
  }, [t, voiceConversationActive])

  useEffect(() => resumeWakeIfPaused, [resumeWakeIfPaused])

  // Speech-output toggles are TTS warm-up / release signals. Entering a voice
  // conversation acquires this window's lease (pre-loads the engine so the
  // first spoken reply doesn't start with dead air); ending it releases the
  // lease, and the backend unloads resident local models once no surface holds
  // one. Fire-and-forget — the toggle never waits on or fails from this.
  useEffect(() => {
    void syncTtsLease(CONVERSATION_LEASE, voiceConversationActive)
  }, [voiceConversationActive])

  useEffect(() => () => void syncTtsLease(CONVERSATION_LEASE, false), [])

  // "Read replies aloud" is the same signal, held for as long as the toggle is
  // on (it mirrors voice.auto_tts, so this also warms at startup when the
  // preference is already set).
  const autoSpeakReplies = useStore($autoSpeakReplies)

  useEffect(() => {
    void syncTtsLease(READ_ALOUD_LEASE, autoSpeakReplies)
  }, [autoSpeakReplies])

  // Explicit start/end for the on-screen conversation controls (the hotkey uses
  // the gated toggle above).
  const startConversation = useCallback(() => {
    if (microphoneOwner === null) {
      setActiveVoiceContextEpoch(voiceContextEpoch)
    }
  }, [voiceContextEpoch])

  const endConversation = useCallback(() => {
    setActiveVoiceContextEpoch(null)
    void conversation.end()
  }, [conversation])

  const handleToggleAutoSpeak = useCallback(() => {
    void setAutoSpeakReplies(!$autoSpeakReplies.get()).catch(error =>
      notifyError(error, t.settings.config.autosaveFailed)
    )
  }, [t])

  useAutoSpeakReplies({
    conversationActive: voiceConversationActive,
    failureLabel: t.assistant.thread.readAloudFailed,
    markSpoken: consumePendingResponse,
    pendingReply: pendingResponse,
    sessionId
  })

  return {
    conversation,
    dictate,
    endConversation,
    handleToggleAutoSpeak,
    startConversation,
    voiceController,
    voiceActivityState,
    voiceConversationActive,
    voiceStatus
  }
}
