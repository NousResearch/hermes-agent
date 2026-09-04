import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useRef, useState } from 'react'

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
import { getActiveComposer, onComposerVoiceToggleRequest } from '../focus'
import { useComposerScope } from '../scope'
import type { ChatBarProps } from '../types'

import { useAutoSpeakReplies } from './use-auto-speak-replies'
import { useComposerPtt } from './use-composer-ptt'
import { useVoiceConversation } from './use-voice-conversation'
import { useVoiceRecorder } from './use-voice-recorder'

interface WakePauseCoordinatorOptions {
  pause: () => Promise<unknown> | unknown
  resume: () => Promise<unknown> | unknown
}

interface WakePauseCoordinator {
  barrier: () => Promise<void> | null
  pause: () => Promise<void>
  resume: () => Promise<void>
}

/** Serializes wake ownership so a pending pause cannot race a later resume. */
export function createWakePauseCoordinator({ pause, resume }: WakePauseCoordinatorOptions): WakePauseCoordinator {
  let generation = 0
  let paused = false
  let pauseBarrier: Promise<void> | null = null
  let resumeGeneration: number | null = null
  let resumePromise = Promise.resolve()

  return {
    barrier: () => pauseBarrier,
    pause: () => {
      const owner = ++generation
      paused = true

      const barrier = Promise.resolve()
        .then(() => pause())
        .then(() => undefined)
        .catch(() => undefined)

      pauseBarrier = barrier

      return barrier.then(() => {
        if (generation === owner && pauseBarrier === barrier && !paused) {
          pauseBarrier = null
        }
      })
    },
    resume: () => {
      if (!paused) {
        return Promise.resolve()
      }

      const owner = generation
      const barrier = pauseBarrier ?? Promise.resolve()
      paused = false

      if (resumeGeneration === owner) {
        return resumePromise
      }

      resumeGeneration = owner
      resumePromise = barrier
        .then(() => {
          if (generation !== owner) {
            return
          }

          return Promise.resolve(resume()).then(() => undefined)
        })
        .catch(() => undefined)
        .finally(() => {
          if (generation === owner && resumeGeneration === owner) {
            resumeGeneration = null
            pauseBarrier = null
          }
        })

      return resumePromise
    }
  }
}

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
  pttActive: () => boolean
  sessionId: string | null | undefined
  /** This composer's focus-bus key — voice toggles targeting another
   *  composer (or the active one, when not us) are ignored. */
  target: ComposerTarget
}

/**
 * The composer's voice engine: push-to-talk dictation (transcript → draft), the
 * full voice-conversation loop, and auto-speak of replies. Self-contained — it
 * consumes the draft/submit primitives passed in but nothing depends back on it,
 * so it lifts cleanly out of ChatBar.
 */
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
  pttActive,
  sessionId,
  target
}: UseComposerVoiceArgs) {
  const { t } = useI18n()
  // A tile's composer speaks ITS transcript, not the primary chat's.
  const { $messages } = useComposerScope()
  const [voiceConversationActive, setVoiceConversationActive] = useState(false)
  const ownsWakeIndicatorRef = useRef(false)
  const voiceStartRequest = useStore($voiceConversationStartRequest)

  const submitVoiceTurn = async (text: string) => {
    if (busy) {
      // Busy may begin after PTT starts. Keep the completed dictation visible
      // as this composer's draft instead of silently dropping it or queueing it.
      insertText(text)

      return
    }

    triggerHaptic('submit')
    resetBrowseState(sessionId)
    clearDraft()
    await onSubmit(text)
  }

  const { cancelRecording, dictate, startRecording, stopRecording, voiceActivityState, voiceStatus } = useVoiceRecorder(
    {
      focusInput,
      maxRecordingSeconds,
      onTranscript: insertText,
      onTranscribeAudio
    }
  )

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

  const wakeCoordinatorRef = useRef(
    createWakePauseCoordinator({
      pause: async () => {
        await $gateway.get()?.request('wake.pause', {})
      },
      resume: () => resumeWakeAfterVoice()
    })
  )

  const wakeCoordinator = wakeCoordinatorRef.current

  const conversation = useVoiceConversation({
    busy,
    consumePendingResponse,
    enabled: voiceConversationActive,
    onFatalError: () => setVoiceConversationActive(false),
    // Speaking over the model mid-generation interrupts the in-flight turn —
    // the same seam as the Stop button — so the interjection becomes the next
    // turn instead of waiting behind a reply the user already rejected.
    onInterrupt,
    // A spoken stop command ("stop", "never mind", "goodbye", …) ends the
    // hands-free conversation. Flipping the flag is the authoritative off
    // switch — the enabled=false prop + effect below drive conversation.end()
    // teardown (mic close, wake re-arm).
    onStopWord: () => setVoiceConversationActive(false),
    onSubmit: submitVoiceTurn,
    onTranscribeAudio,
    pendingResponse: pendingTurnResponse,
    // Before the conversation opens the mic, wait for any in-flight wake.pause
    // finish releasing the capture device (see wakeCoordinator).
    beforeMicOpen: () => wakeCoordinator.barrier() ?? undefined
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
      setVoiceConversationActive(false)
      void conversation.end()
    } else {
      setVoiceConversationActive(true)
    }
  }, [conversation, disabled, voiceConversationActive])

  useEffect(
    () => onComposerVoiceToggleRequest(toggled => toggled === target && toggleVoiceConversation()),
    [target, toggleVoiceConversation]
  )

  useEffect(() => {
    if (target === 'main' && !disabled && takeVoiceConversationStart(voiceStartRequest) && !voiceConversationActive) {
      setVoiceConversationActive(true)
    }
  }, [disabled, target, voiceConversationActive, voiceStartRequest])

  const resumeWakeIfPaused = useCallback(() => wakeCoordinator.resume(), [wakeCoordinator])
  const pauseWakeForVoice = useCallback(() => wakeCoordinator.pause(), [wakeCoordinator])

  useEffect(() => {
    if (voiceConversationActive) {
      pauseWakeForVoice()
    } else {
      resumeWakeIfPaused()
    }
  }, [pauseWakeForVoice, resumeWakeIfPaused, voiceConversationActive])

  useComposerPtt({
    active: () => pttActive() && getActiveComposer() === target,
    blocked: busy || disabled || voiceConversationActive,
    cancel: () => {
      cancelRecording()
      resumeWakeIfPaused()
    },
    maxRecordingSeconds,
    start: async () => {
      await pauseWakeForVoice()

      try {
        const started = await startRecording()

        if (!started) {
          resumeWakeIfPaused()
        }

        return started
      } catch (error) {
        resumeWakeIfPaused()
        throw error
      }
    },
    stop: async () => {
      try {
        return await stopRecording()
      } finally {
        resumeWakeIfPaused()
      }
    },
    submit: submitVoiceTurn
  })

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

  useEffect(() => () => void resumeWakeIfPaused(), [resumeWakeIfPaused])

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
  const startConversation = useCallback(() => setVoiceConversationActive(true), [])

  const endConversation = useCallback(() => {
    setVoiceConversationActive(false)
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
    voiceActivityState,
    voiceConversationActive,
    voiceStatus
  }
}
