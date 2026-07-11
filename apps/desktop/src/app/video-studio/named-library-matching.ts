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

function clipRank(clip: VideoLibraryClip): number {
  return (clip.score ?? 0) * 100 + (clip.quality_score ?? 0) * 10 + (clip.confidence ?? 0)
}

export function automaticallySelectClips(
  segments: ScriptSegment[],
  candidatesBySegment: Record<string, VideoLibraryClip[]>
): Record<string, string> {
  const selected: Record<string, string> = {}
  const usedAssets = new Set<string>()

  for (const segment of segments) {
    const ranked = [...(candidatesBySegment[segment.id] || [])].sort((left, right) => clipRank(right) - clipRank(left))
    const clip = ranked.find(candidate => !usedAssets.has(candidate.asset_id)) || ranked[0]

    if (!clip) {
      continue
    }

    selected[segment.id] = clip.id
    usedAssets.add(clip.asset_id)
  }

  return selected
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
