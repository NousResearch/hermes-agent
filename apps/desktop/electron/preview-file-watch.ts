/**
 * Owner-routed delivery for desktop preview/plugin file watches (#108189).
 *
 * Each watch is owned by the WebContents that registered it. Change events
 * must go back to THAT renderer — secondary windows filter
 * `hermes:preview-file-changed` by their own watch id, so delivering only to
 * the main window drops the event.
 */

export const PREVIEW_FILE_CHANGED_CHANNEL = 'hermes:preview-file-changed'

/** The events a watch needs from the WebContents that registered it. */
export type PreviewWatchOwner = Pick<Electron.WebContents, 'isDestroyed' | 'send'> & {
  once(event: 'destroyed', listener: () => void): unknown
  removeListener(event: 'destroyed', listener: () => void): unknown
}

export type PreviewFileChangedPayload = {
  id: string
  path: string
  url: string
}

/**
 * Deliver a change payload to the watch owner. Returns false when the owner
 * is missing or destroyed so callers can tear the watch down.
 */
export function sendPreviewFileChangedToOwner(
  owner: PreviewWatchOwner | null | undefined,
  payload: PreviewFileChangedPayload,
  onDestroyed?: () => void
): boolean {
  if (!owner || owner.isDestroyed()) {
    onDestroyed?.()

    return false
  }

  owner.send(PREVIEW_FILE_CHANGED_CHANNEL, payload)

  return true
}

/**
 * Tear `teardown` down the moment the owner window dies, not at the next
 * (never-arriving) change event: an fs.watch handle for a closed window
 * otherwise lingers until quit. Returns the unsubscribe for the watch's own
 * close().
 */
export function onPreviewWatchOwnerDestroyed(
  owner: PreviewWatchOwner | null | undefined,
  teardown: () => void
): () => void {
  if (!owner || typeof owner.once !== 'function') {
    return () => {}
  }

  owner.once('destroyed', teardown)

  return () => {
    owner.removeListener('destroyed', teardown)
  }
}
