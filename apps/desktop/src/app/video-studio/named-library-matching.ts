import type { VideoLibraryClip } from './moneyprinter-client'

export interface ScriptSegment {
  id: string
  text: string
}

export interface NamedLibraryMatchState {
  candidatesBySegment: Record<string, VideoLibraryClip[]>
  confirmedBySegment: Record<string, string>
  errorsBySegment: Record<string, string>
}

export function segmentVideoScript(script: string): ScriptSegment[] {
  const sentences = script
    .split(/\n\s*\n/)
    .flatMap(paragraph => paragraph.match(/[^。！？.!?\n]+[。！？.!?]?/g) || [])
    .map(sentence => sentence.trim())
    .filter(Boolean)

  return sentences.map((text, index) => ({ id: `segment-${index + 1}`, text }))
}

export function emptyMatchState(): NamedLibraryMatchState {
  return {
    candidatesBySegment: {},
    confirmedBySegment: {},
    errorsBySegment: {}
  }
}

export function clearLibraryMatches(_state: NamedLibraryMatchState): NamedLibraryMatchState {
  return emptyMatchState()
}

export function setSegmentCandidates(
  state: NamedLibraryMatchState,
  segmentId: string,
  clips: VideoLibraryClip[]
): NamedLibraryMatchState {
  const errorsBySegment = { ...state.errorsBySegment }
  delete errorsBySegment[segmentId]

  return {
    ...state,
    candidatesBySegment: { ...state.candidatesBySegment, [segmentId]: clips },
    errorsBySegment
  }
}

export function confirmSegmentClip(
  state: NamedLibraryMatchState,
  segmentId: string,
  clipId: string
): NamedLibraryMatchState {
  return {
    ...state,
    confirmedBySegment: { ...state.confirmedBySegment, [segmentId]: clipId }
  }
}

export function setSegmentError(
  state: NamedLibraryMatchState,
  segmentId: string,
  message: string
): NamedLibraryMatchState {
  return {
    ...state,
    errorsBySegment: { ...state.errorsBySegment, [segmentId]: message }
  }
}
