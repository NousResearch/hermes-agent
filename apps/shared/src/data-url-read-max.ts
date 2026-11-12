/**
 * Size cap for local files Desktop loads as data URLs (composer attach, image
 * preview, …).
 *
 * Main owns the persisted value; the renderer mirrors it in Settings → Chat and
 * clamps optimistically before sending. Both ends have to agree on the default
 * and the bounds, so they live here rather than as two constants with a
 * "keep these in sync" comment between them.
 *
 * The whole file is base64-buffered in main, so this is a memory guard, not a
 * model limit. The ceiling is only a typo guard — values well below it can
 * still OOM the app.
 */

export const DATA_URL_READ_DEFAULT_MAX_MB = 16
export const DATA_URL_READ_MIN_MAX_MB = 1
export const DATA_URL_READ_MAX_MAX_MB = 4096

export function clampDataUrlReadMaxMb(value: unknown): number {
  const parsed = Number(value)

  if (!Number.isFinite(parsed)) {
    return DATA_URL_READ_DEFAULT_MAX_MB
  }

  return Math.min(DATA_URL_READ_MAX_MAX_MB, Math.max(DATA_URL_READ_MIN_MAX_MB, Math.round(parsed)))
}
