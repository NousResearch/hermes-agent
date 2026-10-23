import { useStore } from '@nanostores/react'
import { useEffect, useRef } from 'react'

import { releaseUnplayedSpokenReply, spokenReplyOf, spokenSegmentKey } from '@/lib/spoken-reply'
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
  /** Text that identifies this message within its turn: the bubble's last text
   *  part. Hydration folds a tool-call turn into one bubble whose full text
   *  changes, but whose last text part is still this message. Defaults to `text`. */
  segmentText?: string
  /** Survives the live-id rewrite. Absent callers still speak; they just cannot join an in-flight play. */
  turnKey?: string
}

// Turns this window speaks for. The window that wins a turn's first message
// voices every later message of that turn, so a second, idle window cannot
// start the final reply over the first window's progress update.
const OWNED_TURNS_MAX = 64
const ownedTurns = new Set<string>()

async function claimSpeech(turnKey: string | undefined, claimKey: string): Promise<boolean> {
  if (turnKey && !ownedTurns.has(turnKey)) {
    if (!(await ownsAmbientCue(`speak:turn:${turnKey}`))) {
      return false
    }

    ownedTurns.add(turnKey)

    if (ownedTurns.size > OWNED_TURNS_MAX) {
      ownedTurns.delete(ownedTurns.values().next().value!)
    }
  }

  return ownsAmbientCue(claimKey)
}

export function clearOwnedSpeechTurnsForTests(): void {
  ownedTurns.clear()
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
      // Only one window voices a turn when the same chat is open in several:
      // the turn's owner (see claimSpeech), then once per message. Keys are the
      // turn and its message text, not the row id: hydration rewrites the row
      // id, and a second claim would start a second clip. A turn with tool
      // calls speaks several messages, so the turn alone would let the first
      // message's claim refuse the final reply.
      const segmentKey = reply.turnKey ? spokenSegmentKey(reply.turnKey, reply.segmentText ?? reply.text) : null
      const claimKey = `speak:${segmentKey ?? reply.id}`

      void claimSpeech(reply.turnKey, claimKey).then(owns => {
        if (attempt !== attemptSeq) {
          return
        }

        if (!owns) {
          // Another window owns this turn, or this message was already voiced.
          console.debug('[auto-speak] claim already taken', claimKey)

          return
        }

        void playSpeechText(reply.text, {
          connectionId,
          messageId: reply.id,
          profile,
          source: 'read-aloud',
          ...(segmentKey ? { turnKey: segmentKey } : {})
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
