import { persistentAtom } from '@/lib/persisted'

// Device-level video playback speed. Choosing a speed in any transcript video
// (via the native controls' rate menu) persists it as the rate every later
// player starts from, so a viewer who watches at 2x doesn't re-select it per
// clip. Ported from block/buzz#7336, adapted to our persistentAtom store.

const STORAGE_KEY = 'hermes.desktop.videoPlaybackSpeed'

export const DEFAULT_VIDEO_PLAYBACK_SPEED = 1

// Chromium's native rate menu offers 0.25–2; leave headroom for keyboard or
// future custom controls without admitting a garbage stored value.
const MIN_SPEED = 0.25
const MAX_SPEED = 4

/** True for rates a <video> can sensibly play at. */
export function isVideoPlaybackSpeed(speed: number): boolean {
  return Number.isFinite(speed) && speed >= MIN_SPEED && speed <= MAX_SPEED
}

export const $videoPlaybackSpeed = persistentAtom<number>(STORAGE_KEY, DEFAULT_VIDEO_PLAYBACK_SPEED, {
  decode: raw => {
    const parsed = Number(raw)

    return isVideoPlaybackSpeed(parsed) ? parsed : DEFAULT_VIDEO_PLAYBACK_SPEED
  },
  // The default doesn't need a stored record; encoding null removes the key.
  encode: value => (value === DEFAULT_VIDEO_PLAYBACK_SPEED ? null : String(value))
})

/** Persist a user-chosen rate; out-of-range values are ignored, not clamped. */
export function setVideoPlaybackSpeed(speed: number) {
  if (isVideoPlaybackSpeed(speed) && speed !== $videoPlaybackSpeed.get()) {
    $videoPlaybackSpeed.set(speed)
  }
}
