import type { CanonicalVoiceClaim } from '@hermes/shared'
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'

import { resolveOwnerNow } from '@/api/client'
import type { CreateBackendSessionForSend, VoiceDraftDestination } from '@/app/session/hooks/use-session-actions/create-overrides'
import { useI18n } from '@/i18n'
import { sanitizeTextForSpeech } from '@/lib/speech-text'
import { type LiveHistoryMessage, type LiveTranscriptFragment, VoiceLiveSession } from '@/lib/voice-live'
import { isVoiceStopCommand } from '@/lib/voice-stop-word'
import { notify, notifyError } from '@/store/notifications'

import { useComposerScope } from '../scope'

import { micError } from './use-mic-recorder'
import type { ConversationStatus } from './use-voice-conversation'

/** How long an accepted delegation may sit before the gateway shows the turn running. */
const SUBMIT_SETTLE_GRACE_MS = 15_000
/** Quiet after the last user transcript fragment before the utterance is judged
 *  as a whole ("stop" ends the chat; "stop the container" is a request). */
const UTTERANCE_SETTLE_MS = 1_500
interface CallTarget {
  createVoiceDraftSession?: CreateBackendSessionForSend
  storedSessionId?: null | string
  voiceDraftKey?: string
}

/** A call opened on New Chat: it creates its own chat before the mic opens. */
function isFreshDraftTarget(target: CallTarget): boolean {
  return !target.storedSessionId && !!target.voiceDraftKey && !!target.createVoiceDraftSession
}

/** Create the New Chat draft as the call's chat. Resolves its stored id, or null
 *  when the start was superseded meanwhile. */
async function createCallChat(
  target: CallTarget,
  isCurrent: () => boolean,
  onAssigned: (destination: VoiceDraftDestination) => void
): Promise<null | string> {
  if (!target.voiceDraftKey || !target.createVoiceDraftSession) {
    throw new Error('The chat does not have a resolved voice destination.')
  }

  let assigned: VoiceDraftDestination | undefined

  const runtime = await target.createVoiceDraftSession(null, {
    voiceDraft: {
      key: target.voiceDraftKey,
      isCurrent,
      onAssigned: destination => {
        assigned = destination
        onAssigned(destination)
      }
    }
  })

  if (!isCurrent()) {
    return null
  }

  const destination = assigned as VoiceDraftDestination | undefined

  if (!destination || runtime !== destination.runtimeSessionId) {
    throw new Error('The fresh chat could not be assigned a voice destination.')
  }

  return destination.storedSessionId
}

interface PendingVoiceResponse {
  id: string
  pending: boolean
  text: string
}

interface VoiceLiveConversationOptions {
  busy: boolean
  enabled: boolean
  onFatalError?: () => void
  /** Interrupt the in-flight Hermes turn (Stop-button seam). Fired when a new
   *  delegation supersedes one still running. */
  onInterrupt?: () => Promise<void> | void
  onStopWord?: () => void
  /** Submit a Hermes turn: `text` is the user's last words (the bubble and the
   *  persisted row), `voiceContext` the recent spoken exchange for unsaved calls. */
  onSubmit: (text: string, voiceContext: string, submissionId: string, storedSessionId: string, createdRuntimeId?: string, claim?: CanonicalVoiceClaim) => Promise<boolean | void> | boolean | void
  createVoiceDraftSession?: CreateBackendSessionForSend
  voiceDraftKey?: string
  voiceRouteSessionId?: string | null
  pendingResponse: () => PendingVoiceResponse | null
  consumePendingResponse: () => void
  /** Text turns to seed the live model with when the session opens. */
  seedHistory: () => LiveHistoryMessage[]
  /** Names of tools currently running in the turn (quiet progress for the voice). */
  activeToolLabel?: () => null | string
  beforeMicOpen?: () => Promise<void> | void
  storedSessionId?: null | string
}

/** Recorded claims use arrival order and leave spoken context to the owner. */
export function delegationPrompt(context: LiveTranscriptFragment[], recorded = false): {
  context: string
  eventIds: null | string[]
  prompt: string
} {
  type SpeechGroup = LiveTranscriptFragment & { eventIds: null | string[] }
  const turns: SpeechGroup[] = []

  // Only unsaved calls infer order and merge backchannels from timing. The
  // owner's recorded claim requires a contiguous group in wire order.
  const timed = !recorded && context.every(
    ({ endMs, startMs, timingMissing }) =>
      !timingMissing && Number.isFinite(startMs) && Number.isFinite(endMs) && startMs >= 0 && endMs > startMs
  )

  const fragments = timed ? [...context].sort((a, b) => a.startMs - b.startMs) : context
  let current: SpeechGroup | undefined
  let lastUser: SpeechGroup | undefined
  let overlappingReplyEnd = 0

  for (const fragment of fragments) {
    // Do not infer a turn boundary from a pause. Only bridge an intervening
    // assistant delta when it actually overlaps speech on both sides.
    const joinsUser = timed && fragment.speaker === 'user' && fragment.startMs < overlappingReplyEnd
    const previous = joinsUser ? lastUser : current?.speaker === fragment.speaker ? current : undefined

    if (previous) {
      previous.text += fragment.text
      previous.endMs = Math.max(previous.endMs, fragment.endMs)

      if (fragment.event_id && previous.eventIds) {
        previous.eventIds.push(fragment.event_id)
      } else {
        previous.eventIds = null
      }

      current = previous
    } else {
      current = { ...fragment, eventIds: fragment.event_id ? [fragment.event_id] : null }
      turns.push(current)
    }

    if (fragment.speaker === 'user') {
      lastUser = current
      overlappingReplyEnd = 0
    } else if (lastUser && fragment.startMs < lastUser.endMs) {
      overlappingReplyEnd = Math.max(overlappingReplyEnd, fragment.endMs)
    }
  }

  const prompt = (lastUser?.text ?? '').replace(/\s+/g, ' ').trim()
  const eventIds = prompt ? (lastUser?.eventIds ?? null) : null

  if (recorded) {
    return { context: '', eventIds, prompt }
  }

  const transcript = turns
    .map(turn => `${turn.speaker === 'user' ? 'User' : 'Voice assistant'}: ${turn.text.replace(/\s+/g, ' ').trim()}`)
    .filter(line => !line.endsWith(': '))
    .join('\n')

  return { context: transcript, eventIds, prompt: prompt || transcript.slice(-400) }
}

/**
 * Body of the session-end toast. `connection_lost` and `closed` are our own
 * machine reasons (`lib/voice-live.ts`) and get i18n copy; so does a blank
 * reason, which has no wording of its own. Any other reason is server-sent and
 * unbounded, so it passes through verbatim (issue #111987 — no redaction claim
 * for vendor strings).
 */
export function liveEndedMessage(
  reason: string,
  usageSeconds: null | number,
  copy: { liveEndedClosed: string; liveEndedConnectionLost: string }
): string {
  let text = reason?.trim() ?? ''

  if (text === 'connection_lost') {
    text = copy.liveEndedConnectionLost
  } else if (text === 'closed' || !text) {
    text = copy.liveEndedClosed
  }

  return usageSeconds != null ? `${text} (${Math.round(usageSeconds)}s)` : text
}

/**
 * GPT-Live conversation engine — same public shape as `useVoiceConversation`
 * so the composer can mount either from `voice.voice_chat_mode`.
 *
 * Status mapping: `listening` = session up, voice idle; `speaking` = the
 * remote track is producing audio; `thinking` = a delegation is in flight in
 * Hermes. There is no `transcribing` phase: the voice model owns speech.
 */
export function useVoiceLiveConversation({
  busy,
  enabled,
  onFatalError,
  onInterrupt,
  onStopWord,
  onSubmit,
  pendingResponse,
  consumePendingResponse,
  seedHistory,
  activeToolLabel,
  beforeMicOpen,
  storedSessionId = null,
  createVoiceDraftSession,
  voiceDraftKey,
  voiceRouteSessionId
}: VoiceLiveConversationOptions) {
  const { t } = useI18n()
  const voiceCopy = t.notifications.voice
  const [status, setStatus] = useState<ConversationStatus>('idle')
  const [muted, setMuted] = useState(false)
  const [level, setLevel] = useState(0)
  // Mirrors delegationRef for the reply-drive effect: a new delegation must
  // restart the feed loop, and a ref write alone does not re-render.
  const [activeDelegation, setActiveDelegation] = useState<null | string>(null)
  const sessionRef = useRef<null | VoiceLiveSession>(null)
  // Capture and delegation must share the same stored chat and owner for the
  // entire call. A navigation must not hand the old call to new callbacks.
  const { connectionId: ownerConnectionId, profile: ownerProfile } = useComposerScope()
  const ownerHint = { connectionId: ownerConnectionId, profile: ownerProfile }
  const owner = resolveOwnerNow(ownerHint)
  const targetKey = JSON.stringify([owner.connectionId, owner.profile, storedSessionId])
  const target = { key: targetKey, ownerHint, storedSessionId, voiceDraftKey, voiceRouteSessionId, createVoiceDraftSession }
  const targetRef = useRef(target)
  targetRef.current = target
  const ownsCallTargetRef = useRef<null | (() => boolean)>(null)
  // Bumped by every start/end so an in-flight start() that lost the race
  // (StrictMode double-effect, quick toggle) closes its session instead of
  // leaving a second billed one running.
  const startEpochRef = useRef(0)
  const startingRef = useRef(false)
  // Set at delegation submit; a turn is only "settled" once it has been seen
  // running (busy) or produced a reply — the gateway ack lags the submit.
  const turnObservedRef = useRef(false)
  const submittedAtRef = useRef(0)
  const enabledRef = useRef(enabled)
  const busyRef = useRef(busy)
  const speakingRef = useRef(false)
  const userUtteranceRef = useRef('')
  const utteranceTimerRef = useRef<null | number>(null)
  const delegationRef = useRef<null | string>(null)
  const spokenLengthRef = useRef(0)
  const spokenResponseIdRef = useRef<null | string>(null)
  const lastToolLabelRef = useRef<null | string>(null)
  const wasEnabledRef = useRef(enabled)

  const latest = useRef({
    activeToolLabel,
    beforeMicOpen,
    onFatalError,
    onInterrupt,
    onStopWord,
    onSubmit,
    pendingResponse,
    consumePendingResponse,
    seedHistory
  })

  latest.current = {
    activeToolLabel,
    beforeMicOpen,
    onFatalError,
    onInterrupt,
    onStopWord,
    onSubmit,
    pendingResponse,
    consumePendingResponse,
    seedHistory
  }

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    enabledRef.current = enabled
  }, [enabled])

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    busyRef.current = busy
  }, [busy])

  const setDelegation = useCallback((id: null | string) => {
    delegationRef.current = id
    setActiveDelegation(id)
  }, [])

  const refreshStatus = useCallback(() => {
    if (!sessionRef.current) {
      setStatus('idle')

      return
    }

    if (speakingRef.current) {
      setStatus('speaking')
    } else if (delegationRef.current) {
      setStatus('thinking')
    } else {
      setStatus('listening')
    }
  }, [])

  const end = useCallback(async () => {
    startEpochRef.current += 1
    startingRef.current = false
    ownsCallTargetRef.current = null

    if (utteranceTimerRef.current) {
      window.clearTimeout(utteranceTimerRef.current)
      utteranceTimerRef.current = null
    }

    userUtteranceRef.current = ''
    const session = sessionRef.current
    sessionRef.current = null
    setDelegation(null)
    spokenResponseIdRef.current = null
    spokenLengthRef.current = 0
    speakingRef.current = false
    session?.close()
    setMuted(false)
    setLevel(0)
    setStatus('idle')
  }, [setDelegation])

  useLayoutEffect(() => {
    if (ownsCallTargetRef.current?.() === false) {
      void end()
      latest.current.onFatalError?.()
    }
  })
  useLayoutEffect(() => () => { void end() }, [end])

  const start = useCallback(async () => {
    if (sessionRef.current || startingRef.current) {
      return
    }

    startingRef.current = true
    const epoch = ++startEpochRef.current
    const target = targetRef.current
    const initialOwner = resolveOwnerNow(target.ownerHint)
    const isFreshDraft = isFreshDraftTarget(target)
    let assigned: VoiceDraftDestination | undefined
    let reachedAssignedChat = false

    const stillTargeted = () => {
      const current = targetRef.current
      const currentOwner = resolveOwnerNow(current.ownerHint)

      if (isFreshDraft) {
        if (assigned && current.storedSessionId === assigned.storedSessionId) {
          reachedAssignedChat = true
          const profile = currentOwner.profile ?? 'default'

          return currentOwner.connectionId === assigned.owner.connectionId &&
            (profile === assigned.backendProfile || profile === assigned.owner.profile)
        }

        // Permit only the handoff announced by our own create, not a new
        // draft or an unrelated route whose stored session has yet to load.
        return !reachedAssignedChat && !current.storedSessionId && (
          current.voiceDraftKey === target.voiceDraftKey ||
          (!!assigned && current.voiceRouteSessionId === assigned.storedSessionId)
        )
      }

      return (
        current.key === target.key &&
        currentOwner.connectionId === initialOwner.connectionId &&
        currentOwner.profile === initialOwner.profile
      )
    }

    ownsCallTargetRef.current = stillTargeted

    const isCurrent = () => enabledRef.current && startEpochRef.current === epoch && stillTargeted()
    let stored = target.storedSessionId

    if (!stored) {
      try {
        stored = await createCallChat(target, isCurrent, destination => {
          assigned = destination
        })
      } catch (error) {
        if (isCurrent()) {
          void end()
          notifyError(error, voiceCopy.couldNotStartSession)
          latest.current.onFatalError?.()
        }

        return
      }

      if (!stored) {
        return
      }
    }

    const callStoredId = stored
    const callOwner = assigned ? { ...assigned.owner, profile: assigned.backendProfile } : initialOwner

    try {
      await latest.current.beforeMicOpen?.()
    } catch {
      // A wake-pause failure must not block an explicit start.
    }

    if (!isCurrent()) {
      if (startEpochRef.current === epoch) {
        startingRef.current = false
      }

      return
    }

    const session = new VoiceLiveSession(
      {
        // The voice model answers a bare "stop" itself (it just goes quiet) and
        // never delegates it, so the spoken stop phrase is judged on the user
        // transcript once the utterance settles.
        onTranscript: fragment => {
          if (sessionRef.current !== session || !stillTargeted()) {
            return
          }

          if (fragment.speaker !== 'user') {
            return
          }

          userUtteranceRef.current += fragment.text

          if (utteranceTimerRef.current) {
            window.clearTimeout(utteranceTimerRef.current)
          }

          utteranceTimerRef.current = window.setTimeout(() => {
            utteranceTimerRef.current = null
            const utterance = userUtteranceRef.current
            userUtteranceRef.current = ''

            if (sessionRef.current === session && stillTargeted() && isVoiceStopCommand(utterance)) {
              void end()
              latest.current.onStopWord?.()
            }
          }, UTTERANCE_SETTLE_MS)
        },
        onClosed: (reason, usageSeconds) => {
          if (sessionRef.current !== session) {
            return
          }

          sessionRef.current = null
          setDelegation(null)
          setStatus('idle')

          if (reason !== 'close_requested') {
            notify({
              kind: 'warning',
              message: liveEndedMessage(reason, usageSeconds, voiceCopy),
              title: voiceCopy.liveEnded
            })
            latest.current.onFatalError?.()
          }
        },
        onDelegation: (delegationId, context, submissionId, recordedCallId) => {
          if (sessionRef.current !== session || !stillTargeted()) {
            return
          }

          const { context: voiceContext, eventIds, prompt } = delegationPrompt(context, Boolean(recordedCallId))

          // A spoken stop command ends the conversation instead of becoming a turn.
          if (prompt && isVoiceStopCommand(prompt)) {
            void end()
            latest.current.onStopWord?.()

            return
          }

          if (!prompt) {
            return
          }

          // No separate interrupt for a recorded admission. It is not fenced to a
          // specific turn, and the spoken-reply bridge can track only one request.
          if (recordedCallId && (!eventIds || busyRef.current || delegationRef.current)) {
            notify({ kind: 'warning', message: '', title: voiceCopy.liveDelegationFailed })
            session.speak(delegationId, 'I could not hand that request to Hermes. Please try again.')

            return
          }

          if (busyRef.current) {
            void latest.current.onInterrupt?.()
          }

          setDelegation(delegationId)
          spokenResponseIdRef.current = null
          spokenLengthRef.current = 0
          lastToolLabelRef.current = null
          turnObservedRef.current = false
          submittedAtRef.current = Date.now()
          latest.current.consumePendingResponse()
          refreshStatus()

          const claim = recordedCallId && eventIds
            ? { call_id: recordedCallId, delegation_id: delegationId, event_ids: eventIds }
            : undefined

          const submission = claim
            ? latest.current.onSubmit(prompt, voiceContext, submissionId, callStoredId, assigned?.runtimeSessionId, claim)
            : assigned
              ? latest.current.onSubmit(prompt, voiceContext, submissionId, callStoredId, assigned.runtimeSessionId)
              : latest.current.onSubmit(prompt, voiceContext, submissionId, callStoredId)

          void Promise.resolve(submission).then(accepted => {
            if (accepted === false) {
              throw new Error(voiceCopy.liveDelegationFailed)
            }
          }).catch(error => {
            if (sessionRef.current !== session || !stillTargeted() || delegationRef.current !== delegationId) {
              return
            }

            notifyError(error, voiceCopy.liveDelegationFailed)
            session.speak(delegationId, 'Sorry, I could not reach Hermes for that request.')
            setDelegation(null)
            refreshStatus()
          })
        },
        onError: (message, fatal) => {
          notify({ kind: fatal ? 'error' : 'warning', message, title: voiceCopy.liveError })
        },
        onSpeakingChange: speaking => {
          if (sessionRef.current !== session || !stillTargeted()) {
            return
          }

          speakingRef.current = speaking
          setLevel(speaking ? 0.6 : 0)
          refreshStatus()
        }
      }
    )

    sessionRef.current = session
    startingRef.current = false
    setMuted(false)
    setStatus('thinking')

    try {
      const captureStatus = await session.start(assigned ? [] : latest.current.seedHistory(), callStoredId, callOwner)

      if (sessionRef.current !== session || startEpochRef.current !== epoch) {
        session.close()

        return
      }

      if (!stillTargeted()) {
        void end()

        return
      }

      if (captureStatus !== 'recording') {
        notify({ kind: 'warning', message: voiceCopy.liveCaptureUnavailable })
      }

      refreshStatus()
    } catch (error) {
      if (sessionRef.current === session) {
        sessionRef.current = null
      }

      session.close()

      if (startEpochRef.current !== epoch) {
        return
      }

      // Only a mic DOMException gets the recorder's copy: this catch also
      // takes non-mic start failures ('GPT-Live session already started',
      // 'Missing local SDP offer', API errors) — those keep their own message.
      notifyError(error instanceof DOMException ? micError(error, voiceCopy) : error, voiceCopy.couldNotStartSession)
      setStatus('idle')
      latest.current.onFatalError?.()
    }
  }, [end, refreshStatus, setDelegation, voiceCopy])

  // Drive the reply back into the voice: stream commentary as Hermes writes
  // it (sentence-chunked), quiet tool progress as thinking appends, and clear
  // the delegation when the turn settles.
  // eslint-disable-next-line no-restricted-syntax -- turn-coordination refs (delegation id / spoken cursor), not atom mirrors
  useEffect(() => {
    const session = sessionRef.current
    const delegationId = delegationRef.current

    if (!session || !delegationId) {
      return undefined
    }

    const tick = () => {
      if (sessionRef.current !== session || delegationRef.current !== delegationId) {
        return
      }

      if (busyRef.current) {
        turnObservedRef.current = true
      }

      const tool = latest.current.activeToolLabel?.() ?? null

      if (tool && tool !== lastToolLabelRef.current) {
        lastToolLabelRef.current = tool
        session.think(delegationId, `Hermes is working: ${tool}. Not done yet.`)
      }

      const response = latest.current.pendingResponse()

      if (response) {
        turnObservedRef.current = true

        if (spokenResponseIdRef.current !== response.id) {
          spokenResponseIdRef.current = response.id
          spokenLengthRef.current = 0
        }

        const spoken = sanitizeTextForSpeech(response.text)

        // Append only completed sentences while streaming; the tail lands on settle.
        if (response.pending || busyRef.current) {
          const boundary = spoken.lastIndexOf('. ', spoken.length - 2)
          const cut = boundary > spokenLengthRef.current ? boundary + 1 : spokenLengthRef.current

          if (cut > spokenLengthRef.current) {
            session.speak(delegationId, spoken.slice(spokenLengthRef.current, cut))
            spokenLengthRef.current = cut
          }

          return
        }

        if (spoken.length > spokenLengthRef.current) {
          session.speak(delegationId, spoken.slice(spokenLengthRef.current))
          spokenLengthRef.current = spoken.length
        }

        latest.current.consumePendingResponse()
        setDelegation(null)
        refreshStatus()

        return
      }

      // The submit ack lags: give the turn time to be seen running before
      // reading "idle and no reply" as a finished turn.
      if (
        !busyRef.current &&
        (turnObservedRef.current || Date.now() - submittedAtRef.current > SUBMIT_SETTLE_GRACE_MS)
      ) {
        // Turn settled without a speakable reply (tool-only, error, interrupted).
        if (spokenLengthRef.current === 0) {
          session.think(delegationId, 'Hermes finished that request without a spoken result.')
        }

        setDelegation(null)
        refreshStatus()
      }
    }

    const timer = window.setInterval(tick, 200)
    tick()

    return () => window.clearInterval(timer)
  }, [activeDelegation, busy, refreshStatus, setDelegation, status])

  const toggleMute = useCallback(() => {
    setMuted(value => {
      const next = !value
      sessionRef.current?.setMuted(next)

      return next
    })
  }, [])

  /** No explicit turn boundary in full duplex; a nudge tells the voice to answer now. */
  const stopTurn = useCallback(() => {
    sessionRef.current?.instruct('The user has finished speaking. Respond now to what they said.')
  }, [])

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    if (enabled && !wasEnabledRef.current) {
      void start()
    }

    if (!enabled && wasEnabledRef.current) {
      void end()
    }

    wasEnabledRef.current = enabled
  }, [enabled, end, start])

  return { end, level, muted, start, status, stopTurn, toggleMute }
}
