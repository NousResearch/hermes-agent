import { useStore } from '@nanostores/react'
import { type FC, useEffect, useMemo, useSyncExternalStore } from 'react'

import { type ChatMessage, chatMessageText } from '@/lib/chat-messages'
import { cn } from '@/lib/utils'
import { $sessionStates } from '@/store/session-states'
import {
  $voiceLiveDrafts,
  retireVoiceLiveDrafts,
  savedVoiceLiveDraftIds,
  type VoiceLiveDraftRun,
  voiceLiveSpeech
} from '@/store/voice-live-draft'

const NO_RUNS: readonly VoiceLiveDraftRun[] = []
const NO_MESSAGES: readonly ChatMessage[] = []

/** The words of a recorded GPT-Live call that are being spoken and not saved
 *  yet, after the saved transcript. Each run is retired when its own saved row
 *  lands, whether or not this thread was on screen at the time. */
export const VoiceLiveDraftRows: FC<{ sessionId?: null | string; sessionKey?: null | string }> = ({
  sessionId,
  sessionKey
}) => {
  const drafts = useStore($voiceLiveDrafts)
  const runs = (sessionKey && drafts[sessionKey]) || NO_RUNS
  const live = runs.length > 0

  // This chat's transcript, read only while it has live words.
  const messages = useSyncExternalStore(
    $sessionStates.subscribe,
    () => (live && sessionId ? ($sessionStates.get()[sessionId]?.messages ?? NO_MESSAGES) : NO_MESSAGES)
  )

  const pendingClaim = useSyncExternalStore($sessionStates.subscribe, () =>
    live && sessionId ? ($sessionStates.get()[sessionId]?.voiceClaimSubmissionId ?? null) : null
  )

  const transcript = useMemo(
    // Session state keeps one saved row per run (display joins happen at render).
    () => messages.map(message => ({ role: message.role, text: chatMessageText(message), voiceLive: message.voiceLive })),
    [messages]
  )

  const saved = useMemo(() => savedVoiceLiveDraftIds(runs, transcript), [runs, transcript])

  useEffect(() => {
    if (sessionKey && saved.size) {
      retireVoiceLiveDrafts(sessionKey, saved)
    }
  }, [saved, sessionKey])

  // While this window's voice claim is pending, its optimistic bubble already shows the
  // claimed run's words: hide that run, never retire it, so a withdrawn bubble brings it back.
  const optimistic = pendingClaim ? transcript.findLast(message => message.role === 'user') : undefined

  const visible = runs.filter(
    run =>
      !saved.has(run.id) &&
      run.text.trim() &&
      !(
        run.claimedBy === pendingClaim &&
        run.role === 'user' &&
        optimistic &&
        !optimistic.voiceLive &&
        voiceLiveSpeech(optimistic.text) === voiceLiveSpeech(run.text)
      )
  )

  if (!visible.length) {
    return null
  }

  return (
    <div aria-live="polite" className="flex flex-col gap-2 pb-(--conversation-turn-gap)" data-slot="voice-live-draft">
      {visible.map(run => (
        <div
          className={cn(
            'max-w-[85%] whitespace-pre-wrap text-sm italic text-muted-foreground',
            run.role === 'user' ? 'self-end rounded-xl border bg-(--dt-user-bubble) px-3 py-2' : 'self-start px-1'
          )}
          data-role={run.role}
          key={run.id}
        >
          {run.text.trim()}
        </div>
      ))}
    </div>
  )
}
