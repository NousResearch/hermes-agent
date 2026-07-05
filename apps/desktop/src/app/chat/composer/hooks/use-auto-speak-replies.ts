import { useStore } from '@nanostores/react'
import { useEffect, useRef } from 'react'

import { playSpeechText } from '@/lib/voice-playback'
import { notifyError } from '@/store/notifications'
import { $messages } from '@/store/session'
import { $voicePlayback } from '@/store/voice-playback'
import { $autoSpeakReplies } from '@/store/voice-prefs'

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
  /** Latest completed assistant reply, or null; `pending` true while still streaming. */
  pendingReply: () => AutoSpeakReply | null
  /** Re-arm on session switch so opening a chat never reads its existing last reply. */
  sessionId: string | null | undefined
}

/**
 * Pure-TTS auto-speak: when `voice.auto_tts` is on, read each completed assistant
 * turn aloud — no dictation, no conversation loop. Stays off while a full voice
 * conversation runs (it speaks replies itself) and never overlaps clips: a reply
 * landing mid-playback is held and spoken on the playback-idle edge. Always reads
 * the latest reply, so a backlog collapses to the newest.
 */
export function useAutoSpeakReplies({
  conversationActive,
  failureLabel,
  markSpoken,
  pendingReply,
  sessionId
}: UseAutoSpeakReplies) {
  const enabled = useStore($autoSpeakReplies)
  const latest = useRef({ conversationActive, failureLabel, markSpoken, pendingReply })
  latest.current = { conversationActive, failureLabel, markSpoken, pendingReply }
  const prevSessionId = useRef(sessionId)

  useEffect(() => {
    if (!enabled) {
      return undefined
    }

    const isNewSession = prevSessionId.current !== sessionId
    prevSessionId.current = sessionId

    if (!isNewSession) {
      // Toggle flipped on mid-conversation — consume the existing last reply
      // immediately so only replies that arrive afterward are spoken.
      latest.current.markSpoken()
    }

    // Skip-count strategy to handle the $messages timing gap on session switch:
    //
    // $messages.subscribe() fires immediately with the current value. On a session
    // switch, the store still holds the OLD session's messages at that point, so
    // the first tick from subscribe is stale data. The second tick is the first
    // real $messages update — the new session's existing messages (which must be
    // silently consumed, not spoken). Only the third tick onward (genuinely new
    // replies) should trigger audio.
    //
    // For the toggle-on case (same session), skip count is 1: the initial
    // subscribe fire with the same data that markSpoken() already consumed above.
    let skipCount = isNewSession ? 2 : 1

    const speakLatest = () => {
      const { conversationActive, failureLabel, markSpoken, pendingReply } = latest.current

      if (skipCount > 0) {
        skipCount -= 1

        // On the very last skip (the first real $messages update carrying the
        // new session's existing data), consume that reply silently so it's
        // not spoken.
        if (skipCount === 0) {
          markSpoken()
        }

        return
      }

      if (conversationActive || $voicePlayback.get().status !== 'idle') {
        return
      }

      const reply = pendingReply()

      if (!reply || reply.pending) {
        return
      }

      markSpoken()
      void playSpeechText(reply.text, { messageId: reply.id, source: 'read-aloud' }).catch(error =>
        notifyError(error, failureLabel)
      )
    }

    // Re-check on a reply completing ($messages) and on the prior clip ending
    // ($voicePlayback → idle), which frees us to read the next held reply.
    const stops = [$messages.subscribe(speakLatest), $voicePlayback.listen(speakLatest)]

    return () => stops.forEach(f => f())
  }, [enabled, sessionId])
}
