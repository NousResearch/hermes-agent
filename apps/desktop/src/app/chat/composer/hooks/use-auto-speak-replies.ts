import { useStore } from '@nanostores/react'
import { useEffect, useRef } from 'react'

import { releaseUnplayedSpokenReply, spokenReplyOf, spokenSegmentKey } from '@/lib/spoken-reply'
import { playSpeechText, type SpeechStreamSession, startSpeechStream } from '@/lib/voice-playback'
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
const pendingTurnClaims = new Map<string, Promise<boolean>>()

async function claimSpeech(turnKey: string | undefined, claimKey: string): Promise<boolean> {
  if (turnKey && !ownedTurns.has(turnKey)) {
    let claim = pendingTurnClaims.get(turnKey)

    if (!claim) {
      // An interim and final can arrive before the first IPC claim settles.
      // They share this window's turn claim, but still claim each message separately.
      claim = ownsAmbientCue(`speak:turn:${turnKey}`)
        .then(owns => {
          if (pendingTurnClaims.get(turnKey) !== claim) {
            return false
          }

          if (owns) {
            ownedTurns.add(turnKey)

            if (ownedTurns.size > OWNED_TURNS_MAX) {
              ownedTurns.delete(ownedTurns.values().next().value!)
            }
          }

          return owns
        })
        .finally(() => {
          if (pendingTurnClaims.get(turnKey) === claim) {
            pendingTurnClaims.delete(turnKey)
          }
        })
      pendingTurnClaims.set(turnKey, claim)
    }

    if (!(await claim)) {
      return false
    }
  }

  return ownsAmbientCue(claimKey)
}

export function clearOwnedSpeechTurnsForTests(): void {
  ownedTurns.clear()
  pendingTurnClaims.clear()
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
 * Pure-TTS auto-speak: when `voice.auto_tts` is on, read assistant replies aloud —
 * no dictation, no conversation loop. Streaming replies are voiced AS THEY LAND
 * (owner ruling 2026-11-10): a pending reply opens a SpeechStreamSession and every
 * message wake appends the new text tail — the server cuts sentences and speaks
 * each one while the rest of the bubble is still generating, so read-aloud keeps
 * pace with the text on screen instead of trailing the whole settle. A provider
 * that cannot stream falls back to the settled whole-text speak. Stays off while
 * a full voice conversation runs (it speaks replies itself) and never overlaps
 * clips: a reply landing mid-playback is held and spoken on the playback-idle
 * edge. Always reads the latest reply, so a backlog collapses to the newest.
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

  // Live read-aloud stream for this window: one session per message, fed by
  // wakes; `streamed` keeps a streamed message from being re-said whole.
  const streamState = useRef<{ live: { id: string; sent: number; session: SpeechStreamSession } | null; starting: Set<string>; streamed: Set<string> }>({
    live: null,
    starting: new Set(),
    streamed: new Set()
  }).current

  useEffect(() => {
    if (!enabled) {
      return undefined
    }

    // Don't read whatever reply already sits at the bottom when the toggle flips
    // on (or a chat opens) — consume it so only later replies are spoken.
    latest.current.markSpoken()

    let attemptSeq = 0

    const claimSegment = (reply: AutoSpeakReply) => {
      const segmentKey = reply.turnKey
        ? spokenSegmentKey(reply.turnKey, reply.segmentText ?? reply.text)
        : null

      // Claiming the settled segment key keeps a SECOND window from voicing
      // the message whole when this window already streamed it.
      return claimSpeech(reply.turnKey, `speak:${segmentKey ?? reply.id}`)
    }

    const feedLive = (entry: { id: string; sent: number; session: SpeechStreamSession }, reply: AutoSpeakReply) => {
      if (reply.text.length > entry.sent) {
        entry.session.append(reply.text.slice(entry.sent))
        entry.sent = reply.text.length
      }

      if (!reply.pending) {
        entry.session.finish()

        if (streamState.live === entry) {
          streamState.live = null
        }

        // Consume only at seal: pendingReply() answers null once the id is
        // marked, so marking while still growing would starve the feed.
        latest.current.markSpoken()
        void claimSegment(reply)
      }
    }

    const speakLatest = () => {
      const { connectionId, conversationActive, failureLabel, markSpoken, pendingReply, profile } = latest.current
      const reply = pendingReply()
      const entry = streamState.live

      // A growing reply with a live session streams regardless of the idle
      // gate: audio already started, keep feeding it.
      if (entry && reply && reply.id === entry.id) {
        feedLive(entry, reply)

        return
      }

      if (entry && (!reply || reply.id !== entry.id)) {
        // The reply we were voicing was replaced mid-stream: seal honestly.
        entry.session.finish()
        streamState.live = null
      }

      if (conversationActive || $voicePlayback.get().status !== 'idle') {
        return
      }

      if (!reply) {
        return
      }

      if (reply.pending && !streamState.streamed.has(reply.id) && !streamState.starting.has(reply.id)) {
        // Read aloud AS IT ARRIVES: the first wake that sees a pending reply
        // opens the stream; later wakes append the growth (see above). The
        // reply is consumed at seal, not here (feed starvation).
        streamState.starting.add(reply.id)

        void claimSpeech(reply.turnKey, `speak:live:${reply.id}`).then(async owns => {
          streamState.starting.delete(reply.id)

          if (!owns) {
            return
          }

          const session = await startSpeechStream({
            connectionId,
            messageId: reply.id,
            profile,
            source: 'read-aloud',
            ...(reply.turnKey ? { turnKey: reply.turnKey } : {})
          })

          if (!session) {
            return                          // provider cannot stream: the settled path speaks it whole
          }

          streamState.streamed.add(reply.id)
          const created = { id: reply.id, sent: 0, session }
          streamState.live = created
          // A mid-stream 'fallback' (no audio ever produced) releases the
          // message back to the whole-text path.
          void session.done.then(outcome => {
            if (outcome === 'fallback' && streamState.live === created) {
              streamState.live = null
              streamState.streamed.delete(created.id)
              speakLatest()
            }
          })
          feedLive(created, latest.current.pendingReply() ?? reply)
        })

        return
      }

      if (streamState.streamed.has(reply.id)) {
        // Already voiced live: consume it and take the segment claim so no
        // window says it twice.
        markSpoken()
        void claimSegment(reply)

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

      if (streamState.live) {
        streamState.live.session.finish()
        streamState.live = null
      }

      stops.forEach(f => f())
    }
  }, [$messages, enabled, sessionId, streamState])
}
