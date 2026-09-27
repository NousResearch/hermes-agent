import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useRef, useState } from 'react'

import { getBriefing } from '@/api/briefing'
import { buildBriefingPrompt, matchesBriefingPhrase } from '@/app/jarvis/briefing'
import { useI18n } from '@/i18n'
import { chatMessageText, collectUnspokenTurnSpeech } from '@/lib/chat-messages'
import { triggerHaptic } from '@/lib/haptics'
import { isJarvisMusicPhrase, startJarvisIntroMusic, stopJarvisIntroMusic } from '@/lib/jarvis-intro-music'
import { adoptSpokenReplySession, markAssistantIdSpoken, resolveSpokenReply } from '@/lib/spoken-reply'
import { CONVERSATION_LEASE, READ_ALOUD_LEASE, syncTtsLease } from '@/lib/tts-lease'
import { playSpeechText } from '@/lib/voice-playback'
import { clearWakeIndicator, syncWakeIndicatorWithVoice } from '@/lib/wake-indicator'
import {
  $briefingRequest,
  $voiceConversationStartRequest,
  takeBriefingRequest,
  takeVoiceConversationStart
} from '@/store/composer'
import { resetBrowseState } from '@/store/composer-input-history'
import { $gateway } from '@/store/gateway'
import { notify, notifyError } from '@/store/notifications'
import { $activeSessionId } from '@/store/session'
import {
  $autoSpeakReplies,
  $briefingPhrases,
  $voiceEngine,
  $voiceStopPhrase,
  setAutoSpeakReplies
} from '@/store/voice-prefs'
import { resumeWakeAfterVoice } from '@/store/wake-word'

import type { ComposerTarget } from '../focus'
import { onComposerVoiceToggleRequest } from '../focus'
import { useComposerScope } from '../scope'
import type { ChatBarProps } from '../types'

import { submitAndAwaitReply } from './agent-reply'
import { useAutoSpeakReplies } from './use-auto-speak-replies'
import { useRealtimeConversation } from './use-realtime-conversation'
import { useVoiceConversation } from './use-voice-conversation'
import { useVoiceRecorder } from './use-voice-recorder'

/** A briefing reads the web and the workspace; give it room before giving up on speaking it. */
const BRIEFING_TIMEOUT_MS = 5 * 60_000

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
  sessionId,
  target
}: UseComposerVoiceArgs) {
  const { locale, t } = useI18n()
  // A tile's composer speaks ITS transcript, not the primary chat's.
  const { $messages } = useComposerScope()
  const [voiceConversationActive, setVoiceConversationActive] = useState(false)
  const voiceConversationActiveRef = useRef(voiceConversationActive)
  const ownsWakeIndicatorRef = useRef(false)
  const previousSessionIdRef = useRef(sessionId)
  const voiceStartRequest = useStore($voiceConversationStartRequest)
  // Live voice runs only on the main composer; tiles keep the classic loop.
  const realtime = useStore($voiceEngine) === 'realtime' && target === 'main'
  const busyRef = useRef(busy)

  const briefingRequest = useStore($briefingRequest)

  voiceConversationActiveRef.current = voiceConversationActive
  busyRef.current = busy

  // eslint-disable-next-line no-restricted-syntax -- session-id adopt token, not an atom mirror
  useEffect(() => {
    adoptSpokenReplySession(previousSessionIdRef.current, sessionId)
    previousSessionIdRef.current = sessionId
  }, [sessionId])

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

  // The daily briefing: the agent gets the gathered data, the transcript shows
  // only what was said (or the button's label).
  const submitBriefing = async (displayText: string) => {
    notify({ id: 'jarvis-briefing', kind: 'info', message: t.jarvisShell.briefing.preparing })

    const data = await getBriefing().catch(() => null)

    const freshChat = !sessionId

    await onSubmit(buildBriefingPrompt(data, locale === 'pl' ? 'pl' : 'en', displayText), { displayText })

    // A briefing that opened its own chat is named for it (an explicit title
    // outranks the auto-title, which would otherwise read the data block).
    // An ongoing chat keeps the name it has.
    const createdId = $activeSessionId.get()

    if (freshChat && createdId) {
      const date = new Date().toLocaleDateString(locale, { day: 'numeric', month: 'long' })

      void $gateway
        .get()
        ?.request('session.title', { session_id: createdId, title: `${t.jarvisShell.briefing.displayText} · ${date}` })
        .catch(() => undefined)
    }
  }

  const submitVoiceTurn = async (text: string) => {
    if (isJarvisMusicPhrase(text)) {
      startJarvisIntroMusic(true)
    }

    if (busy) {
      return
    }

    triggerHaptic('submit')
    resetBrowseState(sessionId)
    clearDraft()

    if (target === 'main' && matchesBriefingPhrase(text, $briefingPhrases.get())) {
      await submitBriefing(text)
    } else {
      await onSubmit(text)
    }
  }

  const wakePausedRef = useRef(false)
  // Resolves once the in-flight wake.pause round-trip completes (mic released by
  // the wake listener). The conversation awaits this before opening its own mic
  // so the two never contend for the device — on Windows especially, opening the
  // capture device while the wake listener still holds it makes getUserMedia
  // fail and the conversation never starts listening.
  const wakePauseBarrierRef = useRef<Promise<void> | null>(null)

  const classicConversation = useVoiceConversation({
    busy,
    consumePendingResponse,
    enabled: voiceConversationActive && !realtime,
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
    // to finish releasing the capture device (see wakePauseBarrierRef).
    beforeMicOpen: () => wakePauseBarrierRef.current ?? undefined
  })

  const liveConversation = useRealtimeConversation({
    onInterrupt,
    sessionId: sessionId ?? null,
    busy: () => busyRef.current,
    enabled: voiceConversationActive && realtime,
    failureLabel: t.notifications.voice.liveFailed,
    markSpoken: id => markAssistantIdSpoken(sessionId, $messages.get(), id),
    messages: () => $messages.get(),
    onFatalError: () => setVoiceConversationActive(false),
    onSubmit: submitVoiceTurn
  })

  const conversation = realtime ? liveConversation : classicConversation

  // Dashboard button / wake phrase. With a voice conversation open the loop
  // speaks the answer itself; otherwise it is read aloud once the turn ends.
  useEffect(() => {
    if (target !== 'main' || disabled) {
      return
    }

    const request = takeBriefingRequest(briefingRequest)

    if (!request) {
      return
    }

    void (async () => {
      const reply = await submitAndAwaitReply(
        { busy: () => busyRef.current, messages: () => $messages.get() },
        () => submitBriefing(t.jarvisShell.briefing.displayText),
        BRIEFING_TIMEOUT_MS
      )

      // Read-aloud already speaks every reply when it is on.
      if (!request.speak || voiceConversationActiveRef.current || $autoSpeakReplies.get() || reply.id === null) {
        return
      }

      markAssistantIdSpoken(sessionId, $messages.get(), reply.id)
      // Speaking is best-effort: the briefing is already on screen.
      await playSpeechText(reply.text, { messageId: reply.id, source: 'read-aloud' }).catch(error =>
        notifyError(error, t.assistant.thread.readAloudFailed)
      )
    })().catch(error => notifyError(error, t.jarvisShell.briefing.failed))
    // Only a new request starts a briefing; the rest is read at that moment.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [briefingRequest, disabled, target])

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
    } else {
      setVoiceConversationActive(true)
    }
  }, [disabled, voiceConversationActive])

  useEffect(
    () => onComposerVoiceToggleRequest(toggled => toggled === target && toggleVoiceConversation()),
    [target, toggleVoiceConversation]
  )

  useEffect(() => {
    if (target === 'main' && !disabled && takeVoiceConversationStart(voiceStartRequest) && !voiceConversationActive) {
      setVoiceConversationActive(true)
    }
  }, [disabled, target, voiceConversationActive, voiceStartRequest])

  const resumeWakeIfPaused = useCallback(() => {
    if (!wakePausedRef.current) {
      return
    }

    wakePausedRef.current = false
    wakePauseBarrierRef.current = null
    // Reconcile, don't just resume: the wake word is a persistent setting, so
    // ending a voice chat must re-arm the listener whenever config says
    // enabled — including when the raw resume loses the mic-release race.
    void resumeWakeAfterVoice()
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

  useEffect(() => {
    if (voiceConversationActive) {
      pauseWakeForVoice()
    } else {
      resumeWakeIfPaused()
    }
  }, [pauseWakeForVoice, resumeWakeIfPaused, voiceConversationActive])

  useEffect(() => {
    if (target !== 'main') {
      return
    }

    if (!voiceConversationActive) {
      stopJarvisIntroMusic()
    }

    return () => stopJarvisIntroMusic()
  }, [target, voiceConversationActive])

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
  const startConversation = useCallback(() => setVoiceConversationActive(true), [])

  const endConversation = useCallback(() => {
    if (!voiceConversationActiveRef.current) {
      return
    }

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
