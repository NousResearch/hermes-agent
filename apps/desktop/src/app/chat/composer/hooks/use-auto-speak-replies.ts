import { useStore } from '@nanostores/react'
import { useEffect, useRef } from 'react'

import { chatMessageText } from '@/lib/chat-messages'
import { markAssistantIdSpoken } from '@/lib/spoken-reply'
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
      id: string
      indexFromEnd: number
      text: string
      sequence: number
      starting: boolean
      session: SpeechStreamSession | null
      fallback: boolean
      fallbackStarted: boolean
      finished: boolean
    }

    let active: Attempt | null = null
    let disposed = false
    let userTurn = $messages.get().findLast(m => m.role === 'user')?.id ?? ''
    let suppressedTurn: string | null = null

    // Prefer durable identity; hydration may replace a provisional id. Count
    // from the end for that fallback so prepending history cannot move it.
    const activeMessage = (attempt: Attempt) => {
      const replies = $messages.get().filter(m => m.role === 'assistant' && !m.hidden)

      return replies.find(m => m.id === attempt.id) ?? replies.at(-1 - attempt.indexFromEnd)
    }

    const markAttemptSpoken = (attempt: Attempt) => {
      const message = activeMessage(attempt)

      if (message) {
        markAssistantIdSpoken(sessionId, $messages.get(), message.id)
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
      const message = activeMessage(attempt)

      if (!message) {
        abandon()

        return
      }

      const text = chatMessageText(message).trim()

      // A rewrite is not an append; replaying the revised prefix would stutter.
      if (!text.startsWith(attempt.text)) {
        suppressedTurn = userTurn
        abandon()

        return
      }

      if (attempt.session && !attempt.finished) {
        attempt.session.append(text.slice(attempt.text.length))
        attempt.text = text

        if (!message.pending) {
          attempt.finished = true
          attempt.session.finish()
        }
      }

      if (attempt.fallback && !message.pending && !attempt.fallbackStarted) {
        attempt.fallbackStarted = true
        markAttemptSpoken(attempt)
        attempt.starting = true // playSpeechText takes ownership synchronously
        const playback = playSpeechText(text, { messageId: message.id, source: 'read-aloud' })
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
      const nextUserTurn = messages.findLast(m => m.role === 'user')?.id ?? ''

      if (conversationActive || nextUserTurn !== userTurn) {
        userTurn = nextUserTurn
        suppressedTurn = null
        abandon()
      }

      if (conversationActive || suppressedTurn === userTurn) {
        return
      }

      if (active) {
        if (active.starting) {
          return
        }

        if ($voicePlayback.get().sequence !== active.sequence) {
          suppressedTurn = userTurn
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
        id: reply.id,
        indexFromEnd: messages
          .slice(messages.findIndex(m => m.id === reply.id) + 1)
          .filter(m => m.role === 'assistant' && !m.hidden).length,
        text: '',
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
        const owns = await ownsAmbientCue(`speak:${reply.id}`)

        if (disposed || active !== attempt) {
          return
        }

        if (!owns || $voicePlayback.get().sequence !== attempt.sequence) {
          suppressedTurn = userTurn
          abandon()

          return
        }

        const session = await startSpeechStream({ messageId: reply.id, source: 'read-aloud' })

        if (disposed || active !== attempt) {
          session?.cancel()

          return
        }

        if (!session && $voicePlayback.get().sequence !== attempt.sequence) {
          suppressedTurn = userTurn
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
            suppressedTurn = userTurn
            abandon()
          } else if (outcome === 'fallback') {
            attempt.fallback = true
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
