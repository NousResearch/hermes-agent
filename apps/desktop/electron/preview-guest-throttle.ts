/** Partition the preview pane's `<webview>` always carries. */
export const PREVIEW_GUEST_PARTITION = 'persist:hermes-preview'

export interface PreviewGuestAttachPrefs {
  backgroundThrottling?: boolean
  preload?: string
}

/**
 * `drive_preview` aims the pointer only after one animation frame inside the
 * guest. Background throttling can suspend that frame and stall input before
 * dispatch. Keep preview guests unthrottled so background actions can proceed.
 * Chat renderers retain their stream-throttle settings, but Electron draws
 * frames for the entire containing window while any displayed guest disables
 * throttling; an open preview therefore also has an idle rendering cost.
 */
export function applyPreviewGuestAttachPrefs(
  webPreferences: PreviewGuestAttachPrefs,
  partition: string | undefined,
  preloadPath: string
): void {
  if (partition !== PREVIEW_GUEST_PARTITION) {
    return
  }

  webPreferences.preload = preloadPath
  webPreferences.backgroundThrottling = false
}
