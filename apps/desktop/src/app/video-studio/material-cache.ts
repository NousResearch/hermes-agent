export interface TimelineVideoSelection {
  assetId: string
  clipId: string
  file: string
  libraryId: string
  script: string
  segmentId: string
  sourceSha256: string
}

function record(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null
}

function text(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function safePart(value: string, fallback: string): string {
  const normalized = value.trim().replace(/[^a-zA-Z0-9_-]+/g, '_').replace(/^_+|_+$/g, '')
  return normalized || fallback
}

export function timelineVideoSelections(timeline: Record<string, unknown>): TimelineVideoSelection[] {
  const tracks = record(timeline.tracks)
  const videoRows = Array.isArray(tracks?.video) ? tracks.video : []
  const shotRows = Array.isArray(timeline.shotPlan) ? timeline.shotPlan : []
  const shotsByClip = new Map(
    shotRows
      .map(record)
      .filter((shot): shot is Record<string, unknown> => Boolean(shot && text(shot.clipId)))
      .map(shot => [text(shot.clipId), shot] as const)
  )

  return videoRows.flatMap(row => {
    const video = record(row)
    const clipId = text(video?.clipId)
    const file = text(video?.file)
    const shot = shotsByClip.get(clipId)
    if (!clipId || !file || !shot) return []
    const assetId = text(shot.assetId)
    const libraryId = text(shot.libraryId)
    const sourceSha256 = text(shot.sourceSha256)
    if (!assetId || !libraryId || !sourceSha256) return []
    return [
      {
        assetId,
        clipId,
        file,
        libraryId,
        script: text(shot.script),
        segmentId: text(shot.segmentId),
        sourceSha256
      }
    ]
  })
}

export function cacheFilenameForSelection(selection: TimelineVideoSelection): string {
  const basename = selection.file.split(/[\\/]/).pop() || 'selected-clip.mp4'
  const hash = safePart(selection.sourceSha256.slice(0, 12), 'nohash')
  return [
    safePart(selection.libraryId, 'library'),
    safePart(selection.assetId, 'asset'),
    safePart(selection.clipId, 'clip'),
    hash,
    basename
  ].join('-')
}

export function confirmedTimelineFormPatch(localMaterials: string[]) {
  return {
    localMaterials,
    matchMaterialsToScript: true as const,
    videoConcatMode: 'sequential' as const,
    videoSource: 'local' as const
  }
}
