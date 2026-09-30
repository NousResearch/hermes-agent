import { useStore } from '@nanostores/react'
import { useEffect, useRef } from 'react'

import { releaseUnplayedSpokenReply, spokenReplyOf } from '@/lib/spoken-reply'
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
  /** Survives the live-id rewrite. Absent callers still speak; they just cannot join an in-flight play. */
  turnKey?: string
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
 * Pure-TTS auto-speak: when the local read-aloud preference is on (the composer
 * voice menu's "Read replies aloud" toggle, `$autoSpeakReplies` — not the
 * `voice.auto_tts` config, which only drives messaging/CLI), read each completed
 * assistant turn aloud — no dictation, no conversation loop. Stays off while a full voice
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
  // Wake on THIS composer's transcript: a tile subscribed to the primary's
  // would never fire on its own replies (and would fire on someone else's).
  const { $messages, connectionId, profile } = useComposerScope()
  const latest = useRef({ connectionId, conversationActive, failureLabel, markSpoken, pendingReply, profile })
  latest.current = { connectionId, conversationActive, failureLabel, markSpoken, pendingReply, profile }

  useEffect(() => {
    if (!enabled) {
      return undefined
    }

    // Don't read whatever reply already sits at the bottom when the toggle flips
    // on (or a chat opens) — consume it so only later replies are spoken.
    latest.current.markSpoken()

    let attemptSeq = 0

    const speakLatest = () => {
      const { connectionId, conversationActive, failureLabel, markSpoken, pendingReply, profile } = latest.current

      if (conversationActive || $voicePlayback.get().status !== 'idle') {
        return
      }

      const reply = pendingReply()

      if (!reply || reply.pending) {
        return
      }

      const attempt = ++attemptSeq

      markSpoken()
      const marked = spokenReplyOf(sessionId)
      // Only one window voices a given reply when the same chat is open in
      // several. The claim key is the turn, not the row id: hydration rewrites
      // the row id, and a second claim would start a second clip.
      void ownsAmbientCue(`speak:${reply.turnKey ?? reply.id}`).then(owns => {
        if (!owns || attempt !== attemptSeq) {
          return
        }

        void playSpeechText(reply.text, {
          connectionId,
          messageId: reply.id,
          profile,
          source: 'read-aloud',
          ...(reply.turnKey ? { turnKey: reply.turnKey } : {})
        }).then(
          started => {
            if (!started && attempt === attemptSeq) {
              releaseUnplayedSpokenReply(sessionId, marked)
            }
          },
          error => {
            if (attempt === attemptSeq) {
              releaseUnplayedSpokenReply(sessionId, marked)
              notifyError(error, failureLabel)
            }
          }
        )
      })
    }

    // Re-check on a reply completing ($messages) and on the prior clip ending
    // ($voicePlayback → idle), which frees us to read the next held reply.
    const stops = [$messages.subscribe(speakLatest), $voicePlayback.listen(speakLatest)]

    return () => {
      attemptSeq += 1
      stops.forEach(f => f())
    }
  }, [$messages, enabled, sessionId])
}
