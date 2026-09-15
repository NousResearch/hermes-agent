import { useStore } from '@nanostores/react'
import { useEffect, useRef } from 'react'

import { chatMessageText } from '@/lib/chat-messages'
import {
  absorbSpokenReplyRewrite,
  anchorSpeechReply,
  latestSpeechUser,
  markSpokenReply,
  sameSpeechUser,
  speechSourceDelta,
  type SpokenReplyAnchor
} from '@/lib/spoken-reply'
import { playSpeechText, type SpeechStreamSession, startSpeechStream, stopVoicePlayback } from '@/lib/voice-playback'
import { ownsAmbientCue } from '@/store/ambient'
import { notifyError } from '@/store/notifications'
import { $voicePlayback } from '@/store/voice-playback'
import { $autoSpeakReplies } from '@/store/voice-prefs'

import { useComposerScope } from '../scope'

interface AutoSpeakReply {
  id: string
  pending: boolean
  text: string
  spokenText?: string
}

interface UseAutoSpeakReplies {
  conversationActive: boolean
  failureLabel: string
  /** Mark the current last reply spoken — shared dedupe with the conversation consumer. */
  markSpoken: () => void
  /** Latest unspoken assistant reply, including text that is still streaming. */
  pendingReply: () => AutoSpeakReply | null
  /** Re-arm on session switch so opening a chat never reads its existing last reply. */
  sessionId: string | null | undefined
}

/**
 * Pure-TTS auto-speak: feed text as it arrives, through the same sentence/PCM
 * pipeline as voice conversation. One reply owns playback at a time. Stop or
 * new input abandons the remainder, including async setup and fallback work.
 */
export function useAutoSpeakReplies({
  conversationActive,
  failureLabel,
  markSpoken,
  pendingReply,
  sessionId
}: UseAutoSpeakReplies) {
  const enabled = useStore($autoSpeakReplies)
  // Wake on THIS composer's transcript: a tile subscribed to the primary's
  // would never fire on its own replies (and would fire on someone else's).
  const { $messages } = useComposerScope()
  const latest = useRef({ conversationActive, failureLabel, markSpoken, pendingReply })
  latest.current = { conversationActive, failureLabel, markSpoken, pendingReply }

  useEffect(() => {
    if (!enabled || conversationActive) {
      return undefined
    }

    // Don't read whatever reply already sits at the bottom when the toggle flips
    // on (or a chat opens) — consume it so only later replies are spoken.
    latest.current.markSpoken()

    interface Attempt {
      anchor: SpokenReplyAnchor
      text: string
      submittedText: string
      sequence: number
      starting: boolean
      session: SpeechStreamSession | null
      fallback: boolean
      fallbackStarted: boolean
      finished: boolean
    }

    let active: Attempt | null = null
    let disposed = false
    let user = latestSpeechUser($messages.get())
    let suppressedTurn = false

    const activeMessage = (attempt: Attempt) => {
      const previous = attempt.anchor
      attempt.anchor = absorbSpokenReplyRewrite(previous, $messages.get()) ?? previous

      if (previous.text && attempt.anchor.text?.endsWith(previous.text)) {
        // Hydration can prepend already-consumed narration to this final row.
        const prefix = attempt.anchor.text.slice(0, attempt.anchor.text.length - previous.text.length)

        if (prefix) {
          attempt.text = prefix + attempt.text
        }
      }

      return $messages.get().find(message => message.id === attempt.anchor.id)
    }

    const markAttemptSpoken = (attempt: Attempt) => {
      const message = activeMessage(attempt)

      if (message) {
        markSpokenReply(sessionId, { ...attempt.anchor, text: attempt.text })
      }
    }

    const abandon = () => {
      const attempt = active
      active = null

      if (!attempt) {
        return
      }

      markAttemptSpoken(attempt)
      attempt.session?.cancel()

      if ($voicePlayback.get().sequence === attempt.sequence) {
        stopVoicePlayback()
      }
    }

    const complete = (attempt: Attempt) => {
      if (active !== attempt || disposed) {
        return
      }

      markAttemptSpoken(attempt)
      active = null
      speakLatest()
    }

    const feed = (attempt: Attempt) => {
      // Text completion seals input, not playback. A later history refresh
      // must not cancel PCM which is still being synthesized or drained.
      if (attempt.finished) {
        return
      }

      const message = activeMessage(attempt)

      if (!message) {
        suppressedTurn = true
        abandon()

        return
      }

      const text = chatMessageText(message).trim()

      const delta = speechSourceDelta(attempt.text, text)

      // A content rewrite is not an append; display-only whitespace is.
      if (delta === null) {
        suppressedTurn = true
        abandon()

        return
      }

      if (attempt.session && !attempt.finished) {
        attempt.session.append(delta)
        attempt.submittedText += delta
        attempt.text = text
        attempt.anchor = anchorSpeechReply($messages.get(), message.id)

        if (!message.pending) {
          attempt.finished = true
          attempt.session.finish()
        }
      }

      if (attempt.fallback && !message.pending && !attempt.fallbackStarted) {
        attempt.fallbackStarted = true
        attempt.finished = true
        attempt.submittedText += delta
        attempt.text = text
        attempt.anchor = anchorSpeechReply($messages.get(), message.id)
        markAttemptSpoken(attempt)
        attempt.starting = true // playSpeechText takes ownership synchronously
        const playback = playSpeechText(attempt.submittedText, { messageId: message.id, source: 'read-aloud' })
        attempt.sequence = $voicePlayback.get().sequence
        attempt.starting = false
        void playback.catch(error => notifyError(error, latest.current.failureLabel)).finally(() => complete(attempt))
      }
    }

    const speakLatest = () => {
      if (disposed) {
        return
      }

      const { conversationActive, pendingReply } = latest.current
      const messages = $messages.get()
      const nextUser = latestSpeechUser(messages)
      const newInput = !sameSpeechUser(user, nextUser, messages)
      user = nextUser

      if (conversationActive || newInput) {
        suppressedTurn = false
        abandon()
      }

      if (conversationActive || suppressedTurn) {
        return
      }

      if (active) {
        if (active.starting) {
          return
        }

        if ($voicePlayback.get().sequence !== active.sequence) {
          suppressedTurn = true
          abandon()

          return
        }

        feed(active)

        return
      }

      if ($voicePlayback.get().status !== 'idle') {
        return
      }

      const reply = pendingReply()

      if (!reply || !reply.text.trim()) {
        return
      }

      // A just-submitted user row can still have the old assistant above it.
      if (messages.findLastIndex(m => m.role === 'user') > messages.findIndex(m => m.id === reply.id)) {
        return
      }

      const attempt: Attempt = {
        anchor: anchorSpeechReply(messages, reply.id),
        text: reply.spokenText ?? '',
        submittedText: '',
        sequence: $voicePlayback.get().sequence,
        starting: true,
        session: null,
        fallback: false,
        fallbackStarted: false,
        finished: false
      }

      active = attempt
      // Only one window voices a given reply when the same chat is open in
      // several. Own the attempt before awaiting the claim or config lookup so
      // rapid deltas cannot create competing speech sessions.
      void (async () => {
        // A merged bubble can gain a new final suffix before the narration's
        // cross-window claim expires. Each consumed boundary owns its suffix.
        const owns = await ownsAmbientCue(`speak:${reply.id}:${reply.spokenText?.length ?? 0}`)

        if (disposed || active !== attempt) {
          return
        }

        if (!owns || $voicePlayback.get().sequence !== attempt.sequence) {
          suppressedTurn = true
          abandon()

          return
        }

        const session = await startSpeechStream({ messageId: reply.id, source: 'read-aloud' })

        if (disposed || active !== attempt) {
          session?.cancel()

          return
        }

        if (!session && $voicePlayback.get().sequence !== attempt.sequence) {
          suppressedTurn = true
          abandon()

          return
        }

        attempt.sequence = $voicePlayback.get().sequence
        attempt.starting = false
        attempt.session = session
        attempt.fallback = !session
        feed(attempt)

        if (session) {
          const outcome = await session.done

          if (disposed || active !== attempt) {
            return
          }

          if ($voicePlayback.get().sequence !== attempt.sequence) {
            suppressedTurn = true
            abandon()
          } else if (outcome === 'fallback') {
            attempt.fallback = true
            attempt.finished = false
            attempt.session = null
            feed(attempt)
          } else {
            complete(attempt)
          }
        }
      })().catch(error => {
        if (active === attempt && !disposed) {
          notifyError(error, latest.current.failureLabel)
          abandon()
        }
      })
    }

    // Subscribe directly to deltas; React rendering never gates first speech.
    const stops = [$messages.subscribe(speakLatest), $voicePlayback.listen(speakLatest)]

    return () => {
      disposed = true
      stops.forEach(f => f())
      abandon()
    }
  }, [$messages, conversationActive, enabled, sessionId])
}
