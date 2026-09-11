import { useStore } from '@nanostores/react'
import { useEffect, useRef } from 'react'

import { playSpeechText } from '@/lib/voice-playback'
import { ownsAmbientCue } from '@/store/ambient'
import { notifyError } from '@/store/notifications'
import { $voicePlayback } from '@/store/voice-playback'
import { $autoSpeakReplies, $ttsConclusionGraceMs, $ttsConclusionOnly } from '@/store/voice-prefs'

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
  const { $messages } = useComposerScope()
  const latest = useRef({ conversationActive, failureLabel, markSpoken, pendingReply })
  latest.current = { conversationActive, failureLabel, markSpoken, pendingReply }

  useEffect(() => {
    if (!enabled) {
      return undefined
    }

    // Don't read whatever reply already sits at the bottom when the toggle flips
    // on (or a chat opens) — consume it so only later replies are spoken.
    latest.current.markSpoken()

    // Conclusion-only mode: each newly completed reply restarts the quiet
    // window, so a burst of interim messages collapses to its last entry.
    let conclusionTimer: ReturnType<typeof setTimeout> | null = null

    const cancelConclusionTimer = () => {
      if (conclusionTimer !== null) {
        clearTimeout(conclusionTimer)
        conclusionTimer = null
      }
    }

    const speakReply = () => {
      const { conversationActive, failureLabel, markSpoken, pendingReply } = latest.current

      if (conversationActive || $voicePlayback.get().status !== 'idle') {
        return
      }

      const reply = pendingReply()

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

    const speakLatest = () => {
      if (!$ttsConclusionOnly.get()) {
        speakReply()

        return
      }

      const { conversationActive, pendingReply } = latest.current

      if (conversationActive || $voicePlayback.get().status !== 'idle') {
        return
      }

      const reply = pendingReply()

      if (!reply || reply.pending) {
        return
      }

      // markSpoken runs when the window actually closes, not now — otherwise
      // an interim reply would consume the dedupe slot of the final one.
      cancelConclusionTimer()
      conclusionTimer = setTimeout(speakReply, $ttsConclusionGraceMs.get())
    }

    // Re-check on a reply completing ($messages) and on the prior clip ending
    // ($voicePlayback → idle), which frees us to read the next held reply.
    const stops = [$messages.subscribe(speakLatest), $voicePlayback.listen(speakLatest)]

    return () => {
      cancelConclusionTimer()
      stops.forEach(f => f())
    }
  }, [$messages, enabled, sessionId])
}
