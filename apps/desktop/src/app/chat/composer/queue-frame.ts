import type { ComposerAttachment } from '@/store/composer'

import { runComposerMiddleware, type ComposerModeFrame } from './contrib'

/**
 * Seal the composer-mode frame for a draft that is about to be QUEUED.
 *
 * Every enqueue path must capture the frame at enqueue time: the drain hands
 * the sealed frame back as submit options, so the send carries the mode the
 * user queued WITH. Re-deriving at drain time would stamp the queued send with
 * whatever mode is live then. Runs the middleware chain once; a cancel yields
 * an empty frame — queueing must never lose the words over a frame lookup.
 */
export async function sealQueuedFrame(
  text: string,
  attachments: ComposerAttachment[]
): Promise<ComposerModeFrame> {
  const sealed = await runComposerMiddleware({ text, attachments })

  if (!sealed) {
    return {}
  }

  return {
    ...(sealed.mode ? { mode: sealed.mode } : {}),
    ...(sealed.note ? { note: sealed.note } : {})
  }
}
