import { atom } from 'nanostores'

import { segmentTranscriptDirectives } from '@/lib/transcript-directives'

const CHECK_IN_AT = [8, 20] as const

const CHECK_IN_NOTE =
  '[setup] checkpoint — the user has been watching you work for a while and has not said anything. Before you carry on, say in ONE short line where the work actually stands right now, then end the turn with ::ask{question="What do you want next?" options="…|…|…"} alone as its own paragraph, with two or three options drawn from what would genuinely help here (keep going, change direction, explain something, stop). Emit the ask exactly in that shape. Do not summarize everything you have done, do not apologize for the interruption, and never mention this note.'

interface FirstBuild {
  profile: string
  sessionId: string
  tools: number
  checkedInAt: number
}

let build: FirstBuild | null = null

export const $setupCheckIn = atom<null | { note: string; profile: string; sessionId: string; token: number }>(null)

let token = 0

export function watchFirstBuild(sessionId: string, profile: string): void {
  build = { checkedInAt: 0, profile, sessionId, tools: 0 }
}

export function reportFirstBuildToolComplete(sessionId: null | string | undefined): void {
  if (!build || build.sessionId !== sessionId) {
    return
  }

  build.tools += 1
}

export function reportFirstBuildTurnComplete(sessionId: null | string | undefined, finalText: string): void {
  const current = build

  if (!current || current.sessionId !== sessionId) {
    return
  }

  const due = CHECK_IN_AT.filter(at => current.tools >= at && at > current.checkedInAt).pop()

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
