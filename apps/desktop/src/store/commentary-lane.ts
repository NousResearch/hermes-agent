import { atom } from 'nanostores'

// Config-backed toggle for the desktop live "Working" lane. The authoritative
// value lives in the gateway config (display.commentary_lane, default false),
// read/written via the config.get/config.set key `commentary_lane`. The backend
// only wires interim_assistant_callback when this is true, so flipping it off is
// byte-identical to upstream (Codex commentary stays on the reasoning channel).
export type CommentaryLaneRequester = (method: string, params?: Record<string, unknown>) => Promise<unknown>

export const $commentaryLane = atom<boolean>(false)

function normalizeCommentaryLane(value: unknown): boolean {
  if (typeof value === 'boolean') {
    return value
  }

  return ['1', 'true', 'on', 'yes'].includes(String(value ?? '').trim().toLowerCase())
}

export async function syncCommentaryLane(requestGateway: CommentaryLaneRequester): Promise<boolean> {
  const result = (await requestGateway('config.get', { key: 'commentary_lane' })) as { value?: unknown }
  const enabled = normalizeCommentaryLane(result?.value)
  $commentaryLane.set(enabled)

  return enabled
}

export async function setCommentaryLane(
  requestGateway: CommentaryLaneRequester,
  value: boolean
): Promise<boolean> {
  // Optimistic — reconcile with the value the gateway echoes back.
  $commentaryLane.set(value)

  const result = (await requestGateway('config.set', { key: 'commentary_lane', value })) as { value?: unknown }
  const authoritative = normalizeCommentaryLane(result?.value)
  $commentaryLane.set(authoritative)

  return authoritative
}
