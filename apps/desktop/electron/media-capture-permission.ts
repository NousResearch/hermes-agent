import type {
  FilesystemPermissionRequest,
  MediaAccessPermissionRequest,
  OpenExternalPermissionRequest,
  PermissionRequest
} from 'electron'

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
  'media' | 'audioCapture' | 'videoCapture' | 'fullscreen' | 'automatic-fullscreen' | (string & Record<never, never>)

/**
 * The metadata a media permission request may carry.
 *
 * The async request handler receives a union of Electron request shapes
 * (`MediaAccessPermissionRequest` with `mediaTypes`, plus
 * `PermissionRequest`/`FilesystemPermissionRequest`/
 * `OpenExternalPermissionRequest` which carry none), while the sync check
 * handler receives `PermissionCheckHandlerHandlerDetails` (a singular
 * `mediaType`) — in practice `undefined` here, since the check handler is
 * called with no details. The decision logic only cares about `mediaTypes`,
 * so the parameter accepts the full request-handler union next to this
 * widened structural shape. The property stays optional (Windows fires
 * capture requests with an empty or undefined array) and its elements are
 * widened the same way as the permission strings above: Chromium can send
 * types the Electron typings do not model, and the predicate denies anything
 * that is not audio/video.
 */
export type MediaCapturePermissionDetails = {
  mediaTypes?: Array<'video' | 'audio' | (string & Record<never, never>)>
}

/** Everything the permission handlers can pass as request details. */
export type MediaPermissionRequestDetails =
  | MediaCapturePermissionDetails
  | MediaAccessPermissionRequest
  | PermissionRequest
  | FilesystemPermissionRequest
  | OpenExternalPermissionRequest

const carriesMediaTypes = (details: unknown): details is MediaCapturePermissionDetails =>
  typeof details === 'object' && details !== null && 'mediaTypes' in details

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
  details: MediaPermissionRequestDetails | undefined
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

  const mediaTypes = carriesMediaTypes(details) ? details.mediaTypes : undefined

  // Windows: mediaTypes is often empty for a capture request. Don't deny on
  // missing metadata.
  if (!Array.isArray(mediaTypes) || mediaTypes.length === 0) {
    return true
  }

  return mediaTypes.includes('audio') || mediaTypes.includes('video')
}
