import { type CompanionInfo } from '@/store/companion'

// Sprite row → the CompanionInfo frame-count key it resolves to (directional walks and
// aliases collapse onto their base state).
const ROW_TO_FRAME_KEY: Record<string, string> = {
  idle: 'idle',
  wave: 'wave',
  waving: 'wave',
  jump: 'jump',
  jumping: 'jump',
  run: 'run',
  running: 'run',
  'running-right': 'run',
  'running-left': 'run',
  failed: 'failed',
  review: 'review',
  waiting: 'waiting'
}

// Real frame count for a row, preferring the concrete per-row count, then the
// per-state count, then the mapped base state, then the sheet-wide default.
export function frameCountForRow(companion: CompanionInfo, row: string): number {
  const mapped = ROW_TO_FRAME_KEY[row]

  return (
    companion.framesByRow?.[row] ??
    companion.framesByState?.[row] ??
    (mapped ? companion.framesByState?.[mapped] : undefined) ??
    companion.framesPerState ??
    0
  )
}
