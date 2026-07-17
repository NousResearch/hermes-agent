import { useCallback, useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { chatMessageText } from '@/lib/chat-messages'
import { triggerHaptic } from '@/lib/haptics'
import { resetBrowseState } from '@/store/composer-input-history'
import { notifyError } from '@/store/notifications'
import { $messages } from '@/store/session'
import { $autoSpeakReplies, setAutoSpeakReplies } from '@/store/voice-prefs'

import type { ComposerDictationOwner, ComposerTarget } from '../focus'
import {
  claimComposerDictation,
  createComposerDictationOwner,
  onComposerDictationToggleRequest,
  onComposerVoiceToggleRequest,
  ownsComposerDictation,
  releaseComposerDictation
} from '../focus'
import type { ChatBarProps } from '../types'

import { useAutoSpeakReplies } from './use-auto-speak-replies'
import { useVoiceConversation } from './use-voice-conversation'
import { useVoiceRecorder } from './use-voice-recorder'

interface UseComposerVoiceArgs {
  busy: boolean
  clearDraft: () => void
  dictationEnabled: boolean
  dictationScopeKey: string | null
  disabled: boolean
  focusInput: () => void
  insertText: (text: string) => void
  maxRecordingSeconds: number
  onSubmit: ChatBarProps['onSubmit']
  onTranscribeAudio: ChatBarProps['onTranscribeAudio']
  sessionId: string | null | undefined
  submitDraft: () => void
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
  dictationEnabled,
  dictationScopeKey,
  disabled,
  focusInput,
  insertText,
  maxRecordingSeconds,
  onSubmit,
  onTranscribeAudio,
  sessionId,
  submitDraft,
  target
}: UseComposerVoiceArgs) {
  const { t } = useI18n()
  const [voiceConversationActive, setVoiceConversationActive] = useState(false)
  const lastSpokenIdRef = useRef<string | null>(null)
  const dictationOwnerRef = useRef<ComposerDictationOwner | null>(null)
  dictationOwnerRef.current ??= createComposerDictationOwner(target)
  const dictationOwner = dictationOwnerRef.current
  const claimedScopeRef = useRef<{ key: string | null } | null>(null)
  const currentScopeKeyRef = useRef(dictationScopeKey)
  const insertTextRef = useRef(insertText)
  const submitDraftRef = useRef(submitDraft)
  currentScopeKeyRef.current = dictationScopeKey
  insertTextRef.current = insertText
  submitDraftRef.current = submitDraft

  const releaseDictation = useCallback(() => {
    claimedScopeRef.current = null
    releaseComposerDictation(dictationOwner)
  }, [dictationOwner])

  const { cancel: cancelRecorder, dictate: toggleRecorder, voiceActivityState, voiceStatus } = useVoiceRecorder({
    focusInput,
    maxRecordingSeconds,
    onIdle: releaseDictation,
    onTranscript: (text, submit) => {
      const claimedScope = claimedScopeRef.current

      // A session switch cancels capture, but transcription may already be in
      // flight. Never let a stale completion write into the newly active chat.
      if (
        !claimedScope ||
        !ownsComposerDictation(dictationOwner) ||
        claimedScope.key !== currentScopeKeyRef.current
      ) {
        return
      }

      insertTextRef.current(text)

      if (submit) {
        submitDraftRef.current()
      }
    },
    onTranscribeAudio
  })

  const cancelRecorderRef = useRef(cancelRecorder)
  cancelRecorderRef.current = cancelRecorder

  const pendingResponse = () => {
    const messages = $messages.get()
    const last = messages.findLast(m => m.role === 'assistant' && !m.hidden)

    if (!last || last.id === lastSpokenIdRef.current) {
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

  const consumePendingResponse = () => {
    const messages = $messages.get()
    const last = messages.findLast(m => m.role === 'assistant' && !m.hidden)

    if (last) {
      lastSpokenIdRef.current = last.id
    }
  }

  const submitVoiceTurn = async (text: string) => {
    if (busy) {
      return
    }

    triggerHaptic('submit')
    resetBrowseState(sessionId)
    clearDraft()
    await onSubmit(text)
  }

  const conversation = useVoiceConversation({
    busy,
    consumePendingResponse,
    enabled: voiceConversationActive,
    onFatalError: () => setVoiceConversationActive(false),
    onSubmit: submitVoiceTurn,
    onTranscribeAudio,
    pendingResponse
  })

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

  const toggleDictation = useCallback(
    (submitOnStop = false) => {
      // The owner may always finish its capture, even if the composer became
      // disabled while recording. Starting follows the microphone button.
      if (ownsComposerDictation(dictationOwner)) {
        toggleRecorder(submitOnStop)

        return
      }

      if (disabled || !dictationEnabled || voiceConversationActive || voiceStatus !== 'idle') {
        return
      }

      if (claimComposerDictation(dictationOwner)) {
        claimedScopeRef.current = { key: currentScopeKeyRef.current }
        toggleRecorder(submitOnStop)
      }
    },
    [dictationEnabled, dictationOwner, disabled, toggleRecorder, voiceConversationActive, voiceStatus]
  )

  useEffect(
    () => onComposerDictationToggleRequest(toggled => toggled === target && toggleDictation(true)),
    [target, toggleDictation]
  )

  useEffect(() => {
    const claimedScope = claimedScopeRef.current

    if (
      claimedScope &&
      claimedScope.key !== dictationScopeKey &&
      ownsComposerDictation(dictationOwner)
    ) {
      cancelRecorderRef.current()
    }
  }, [dictationOwner, dictationScopeKey])

  useEffect(
    () => () => {
      cancelRecorderRef.current()
      releaseComposerDictation(dictationOwner)
    },
    [dictationOwner]
  )

  useEffect(
    () => onComposerVoiceToggleRequest(toggled => toggled === target && toggleVoiceConversation()),
    [target, toggleVoiceConversation]
  )

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
    dictate: toggleDictation,
    endConversation,
    handleToggleAutoSpeak,
    startConversation,
    voiceActivityState,
    voiceConversationActive,
    voiceStatus
  }
}
