/**
 * Progress check-ins during the first build.
 *
 * Setup hands the first task to a session of its own and then stops. This module counts that session's tool
 * calls and sets `$setupCheckIn` at two of them; the wiring turns each one into a hidden `[setup]` note in the
 * same session, which asks the agent to say where the work stands and what the user wants next. The note
 * arrives in the chat the user is already reading. A cron job would not.
 *
 * Two rules limit the check-ins:
 *
 * - A note is set only between turns, on `message.complete`. A note set mid-turn would become a synthetic
 *   user message inside an assistant turn, which the agent core's role alternation forbids.
 * - No note when the turn already ended with a question. The runbook has the agent ask for a verdict after
 *   the first pass, and a check-in under that ask puts two questions to the user at once.
 */

import { atom } from 'nanostores'

import { segmentTranscriptDirectives } from '@/lib/transcript-directives'

/** Tool call counts at which Setup checks in: the first once the build is visibly under way, the second far
 *  enough in that asking whether the work is still what the user wanted is a real question. */
const CHECK_IN_AT = [8, 20] as const

const CHECK_IN_NOTE =
  '[setup] checkpoint — the user has been watching you work for a while and has not said anything. Before you carry on, say in ONE short line where the work actually stands right now, then end the turn with ::ask{question="What do you want next?" options="…|…|…"} alone as its own paragraph, with two or three options drawn from what would genuinely help here (keep going, change direction, explain something, stop). Emit the ask exactly in that shape. Do not summarize everything you have done, do not apologize for the interruption, and never mention this note.'

interface FirstBuild {
  /** Profile of the build session. The note must be routed to this profile explicitly: the user can return to
   *  Setup's chat while the build runs, which makes hermes-setup the active gateway. */
  profile: string
  sessionId: string
  tools: number
  /** Highest CHECK_IN_AT tool count already used, not a timestamp. */
  checkedInAt: number
}

let build: FirstBuild | null = null

/** Set when a check-in is due. The wiring submits the note to the build's session as a hidden `[setup]` note.
 *  The token changes on every check-in, so a second check-in with the same note is not read as a duplicate
 *  value. */
export const $setupCheckIn = atom<null | { note: string; profile: string; sessionId: string; token: number }>(null)

let token = 0

export function watchFirstBuild(sessionId: string, profile: string): void {
  build = { checkedInAt: 0, profile, sessionId, tools: 0 }
}

export function resetFirstBuildForTests(): void {
  build = null
  token = 0
  $setupCheckIn.set(null)
}

/** Called from the gateway stream on tool.complete. */
export function reportFirstBuildToolComplete(sessionId: null | string | undefined): void {
  if (!build || build.sessionId !== sessionId) {
    return
  }

  build.tools += 1
}

/** Called from the gateway stream on message.complete, the only point where a note may be set. The module
 *  header explains the role alternation rule behind that. */
export function reportFirstBuildTurnComplete(sessionId: null | string | undefined, finalText: string): void {
  const current = build

  if (!current || current.sessionId !== sessionId) {
    return
  }

  const due = CHECK_IN_AT.filter(at => current.tools >= at && at > current.checkedInAt).pop()

  // Skip the check-in when the turn ended with a question, either the runbook's verdict ask or one the agent
  // chose, so the user can answer it. endsInAsk parses the directives instead of matching text, so an `::ask`
  // the agent only described in prose does not count.
  if (due === undefined || endsInAsk(finalText)) {
    return
  }

  current.checkedInAt = due
  token += 1
  $setupCheckIn.set({ note: CHECK_IN_NOTE, profile: current.profile, sessionId, token })
}

function endsInAsk(text: string): boolean {
  return (
    segmentTranscriptDirectives(text)?.some(
      segment => segment.kind === 'directive' && segment.directive.name === 'ask'
    ) === true
  )
}
