/**
 * Watching the first build.
 *
 * Setup hands the first task to its own session and stops talking. What the
 * user feels next used to be nothing until they said something — the guide
 * scheduled itself a DAILY cron and that was the whole of its "proactivity",
 * which on a first run means a check-in that arrives tomorrow, about a task
 * that finished in four minutes.
 *
 * So the check-ins ride the build's own progress instead of a clock. This
 * module counts the work as it happens and, at a couple of points, raises a
 * beacon the wiring turns into a hidden `[setup]` note in that same session —
 * the agent pauses, says where things stand, and asks what the user wants
 * next. It lands where they are already looking, which a cron never does.
 *
 * Two rules keep it from becoming a nag:
 *
 * - It only ever speaks BETWEEN turns (on `message.complete`). A note injected
 *   mid-loop would be a synthetic user message in the middle of an assistant
 *   turn — the alternation the agent core forbids.
 * - It stays quiet when the turn already ended by asking something. The
 *   runbook has the agent ask for a verdict when the first pass lands; a
 *   check-in stacked under that is two questions and no answer.
 */

import { atom } from 'nanostores'

import { segmentTranscriptDirectives } from '@/lib/transcript-directives'

/** Tool calls at which Setup checks in. Two of them: one once the build is
 *  visibly underway, one deep enough in that "still what you wanted?" is a
 *  real question. A third would be nagging. */
const CHECK_IN_AT = [8, 20] as const

const CHECK_IN_NOTE =
  '[setup] checkpoint — the user has been watching you work for a while and has not said anything. Before you carry on, say in ONE short line where the work actually stands right now, then end the turn with ::ask{question="What do you want next?" options="…|…|…"} alone as its own paragraph, with two or three options drawn from what would genuinely help here (keep going, change direction, explain something, stop). Emit the ask exactly in that shape. Do not summarize everything you have done, do not apologize for the interruption, and never mention this note.'

interface FirstBuild {
  /** Profile the build session lives on. Carried because the whisper has to
   *  be routed explicitly: the user can walk back into Setup's chat while the
   *  build runs, which makes hermes-setup the ACTIVE gateway. */
  profile: string
  sessionId: string
  tools: number
  /** Highest CHECK_IN_AT threshold already spent. */
  checkedInAt: number
}

let build: FirstBuild | null = null

/** Raised when the build has earned a check-in; the wiring whispers it into
 *  the build's session as a hidden `[setup]` note. Token-bumped so two
 *  check-ins in one run can't be swallowed as a duplicate value. */
export const $setupCheckIn = atom<null | { note: string; profile: string; sessionId: string; token: number }>(null)

let token = 0

/** Start watching the session Setup just handed the first task to. */
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

/** Called from the gateway stream on message.complete — the only moment a
 *  note may be injected (see the alternation rule in the module header). */
export function reportFirstBuildTurnComplete(sessionId: null | string | undefined, finalText: string): void {
  const current = build

  if (!current || current.sessionId !== sessionId) {
    return
  }

  const due = CHECK_IN_AT.filter(at => current.tools >= at && at > current.checkedInAt).pop()

  // The turn already put a question to the user (the runbook's verdict ask, or
  // one the agent chose). Let them answer it. Parsed, not string-matched — a
  // `::ask` the agent merely talked ABOUT is not a question.
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
