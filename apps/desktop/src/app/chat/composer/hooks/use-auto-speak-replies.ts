import { useStore } from '@nanostores/react'
import { useEffect, useRef } from 'react'

import { playSpeechText } from '@/lib/voice-playback'
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
  /** Whether the agent is mid-turn (the Stop-button seam). A turn that has
   *  started is the only thing that makes a completed reply speakable. */
  busy: boolean
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
  busy,
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
  const busyRef = useRef(busy)
  // Latched true for the duration of a turn: a completed reply is only
  // speakable if this session started a turn since the gate armed. Reset when
  // the effect re-arms (session switch, toggle, mount).
  const freshReplyStartedRef = useRef(false)
  latest.current = { conversationActive, failureLabel, markSpoken, pendingReply }

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    busyRef.current = busy
  }, [busy])

  // eslint-disable-next-line no-restricted-syntax -- idempotent latch, not a mirror; read back only inside the re-arm effect below
  useEffect(() => {
    if (busy) {
      freshReplyStartedRef.current = true
    }
  }, [busy])

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    if (!enabled) {
      return undefined
    }

    // Don't read whatever reply already sits at the bottom when the toggle flips
    // on (or a chat opens) — consume it so only later replies are spoken.
    latest.current.markSpoken()
    freshReplyStartedRef.current = busyRef.current
    let waitingForFreshReply = true

    const speakLatest = () => {
      const { conversationActive, failureLabel, markSpoken, pendingReply } = latest.current
      const reply = pendingReply()

      if (waitingForFreshReply && freshReplyStartedRef.current) {
        waitingForFreshReply = false
      }

      // Session history hydrates asynchronously after `sessionId` changes. The
      // effect's eager mark above can therefore still see the previous session.
      // Ignore completed replies until this session starts a new turn (or a
      // pending assistant reply appears), consuming late-arriving history
      // instead of replaying it aloud.
      if (waitingForFreshReply && reply) {
        if (reply.pending) {
          waitingForFreshReply = false
        } else {
          markSpoken()

          return
        }
      }

      if (conversationActive || $voicePlayback.get().status !== 'idle') {
        return
      }

      if (!reply || reply.pending) {
        return
      }

      markSpoken()
      // Only one window voices a given reply when the same chat is open in
      // several (reply.id is the shared backend message id). markSpoken already
      // ran in every window, so peers just stay quiet.
      void ownsAmbientCue(`speak:${reply.id}`).then(owns => {
        if (owns) {
          void playSpeechText(reply.text, { messageId: reply.id, source: 'read-aloud' }).catch(error =>
            notifyError(error, failureLabel)
          )
        }
      })
    }

    // Re-check on a reply completing ($messages) and on the prior clip ending
    // ($voicePlayback → idle), which frees us to read the next held reply.
    const stops = [$messages.subscribe(speakLatest), $voicePlayback.listen(speakLatest)]

    return () => stops.forEach(f => f())
  }, [$messages, enabled, sessionId])
}
