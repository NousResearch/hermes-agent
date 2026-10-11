/** Partition the preview pane's `<webview>` always carries. */
export const PREVIEW_GUEST_PARTITION = 'persist:hermes-preview'

export interface PreviewGuestAttachPrefs {
  backgroundThrottling?: boolean
  preload?: string
}

/**
 * `drive_preview` aims the pointer only after one animation frame inside the
 * guest. Chromium does not fire that frame while the guest is
 * background-throttled, so a click or keystroke never leaves once the Desktop
 * window is not in front. This guest is the page being driven, not a Hermes
 * chat renderer — those stay on the stream-throttle dial.
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
