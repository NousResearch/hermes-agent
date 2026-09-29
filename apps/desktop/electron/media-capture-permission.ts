import type { MediaAccessPermissionRequest } from 'electron'

/**
 * Permission strings the media-capture session hooks can receive.
 *
 * Electron's typings narrow the request handler to `PermissionRequestHandlerType`
 * (a small allow-listed union) and the check handler to a different, even smaller
 * union that predates the capture permissions. At runtime Chromium sends more
 * strings than either union admits — `audioCapture`, `videoCapture` and
 * `automatic-fullscreen` are all live on the check handler — which is why the
 * call sites widen with `as string` and delegate here. Keeping the widened
 * superset in one named type documents exactly which extra strings we accept
 * instead of scattering casts.
 */
export type MediaCapturePermissionString =
  | 'media'
  | 'audioCapture'
  | 'videoCapture'
  | 'fullscreen'
  | 'automatic-fullscreen'
  | (string & Record<never, never>)

/**
 * The metadata a media permission request may carry.
 *
 * The async request handler receives `MediaAccessPermissionRequest` (with
 * `mediaTypes`), while the sync check handler receives
 * `PermissionCheckHandlerHandlerDetails` (with a singular `mediaType`). The
 * decision logic only cares about `mediaTypes`, so both shapes flow through
 * this widened structural type.
 */
export type MediaCapturePermissionDetails = Pick<MediaAccessPermissionRequest, 'mediaTypes'>

// Microphone and camera capture. The voice composer drives mic access and
// renderer features (e.g. desktop plugins) can drive camera access, both
// through getUserMedia, which Chromium gates behind these two session hooks.
//
// The naive `details.mediaTypes.includes('audio')` check works on macOS but
// breaks on Windows: Chromium frequently fires the request with an empty or
// undefined `mediaTypes`, so a strict check denies it and getUserMedia throws
// NotAllowedError. We therefore allow the capture permissions and treat absent
// metadata as allowed.
//
// Granting here is not the last gate: the OS still applies its own capture
// permission (macOS TCC prompts on first use, per the NSMicrophone/NSCamera
// usage strings), so the user keeps a real allow/deny and can revoke it in
// System Settings afterwards.
//
// Shared by the async request handler (`setPermissionRequestHandler`, which
// receives real `mediaTypes`) and the synchronous check handler
// (`setPermissionCheckHandler`, which Chromium consults for getUserMedia on
// Windows and whose `details` carry no media-type array). Delegating both to
// the same predicate is what keeps the two paths from drifting apart again.
export function isMediaCapturePermission(
  permission: MediaCapturePermissionString,
  details: MediaCapturePermissionDetails | undefined,
): boolean {
  // HTML5 video/audio fullscreen asks the request handler for 'fullscreen'
  // and the check handler for 'automatic-fullscreen'. Both must be allowed
  // or the native fullscreen button on <video controls> does nothing.
  if (permission === 'fullscreen' || permission === 'automatic-fullscreen') {
    return true
  }

  if (permission === 'audioCapture' || permission === 'videoCapture') {
    return true
  }

  if (permission !== 'media') {
    return false
  }

  const mediaTypes = details?.mediaTypes

  // Windows: mediaTypes is often empty for a capture request. Don't deny on
  // missing metadata.
  if (!Array.isArray(mediaTypes) || mediaTypes.length === 0) {
    return true
  }

  return mediaTypes.includes('audio') || mediaTypes.includes('video')
}
