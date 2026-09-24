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

    // Text + timestamp of the reply this effect last started reading. One turn
    // can reach us twice within milliseconds: the transcript updates when the
    // stream ends and again when the live row hydrates to its durable id, and
    // each update hands back a DIFFERENT row id — so the id-keyed spoken anchor
    // misses and the turn looks brand new. Reading it again is the bug (heard as
    // the reply being spoken twice). Same text inside this window is that same
    // turn; a genuinely new turn lands later or says something else.
    let lastStarted: { at: number; text: string } | null = null
    const REPEAT_GUARD_MS = 60_000

    const speakLatest = () => {
      const { connectionId, conversationActive, failureLabel, markSpoken, pendingReply, profile } = latest.current

      if (conversationActive || $voicePlayback.get().status !== 'idle') {
        return
      }

      const reply = pendingReply()

      if (!reply || reply.pending) {
        return
      }

      const now = Date.now()

      if (lastStarted !== null && lastStarted.text === reply.text && now - lastStarted.at < REPEAT_GUARD_MS) {
        // This turn was already read; the anchor drifted. Swallow the replay.
        markSpoken()

        return
      }

      lastStarted = { at: now, text: reply.text }
      markSpoken()
      // Only one window voices a given reply when the same chat is open in
      // several (reply.id is the shared backend message id). markSpoken already
      // ran in every window, so peers just stay quiet.
      void ownsAmbientCue(`speak:${reply.id}`).then(owns => {
        if (owns) {
          void playSpeechText(reply.text, { connectionId, messageId: reply.id, profile, source: 'read-aloud' }).catch(
            error => notifyError(error, failureLabel)
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
