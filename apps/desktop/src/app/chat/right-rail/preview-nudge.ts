/**
 * OVERLAY NUDGE — the one way to say a stage to an overlay that is ALREADY on
 * the page.
 *
 * The act pipeline (preview-act.ts) injects the whole engine because it has to:
 * it needs to resolve a ref, measure a node and read the page back. The stages
 * that only narrate — the idle pulse, the read wipe — need none of that. They
 * are one function call against something the last action already left on the
 * page, so re-shipping the engine to make it would put the payload on the wire
 * on every turn boundary.
 *
 * A page the agent has never acted on has no overlay, and the nudge is a no-op
 * there. That is the right answer rather than a gap: chrome on a page the agent
 * never touched would be a lie about what it did.
 */

import type { WatchStage } from '@/lib/preview-act/watch-in-page'

import { activePreviewScriptRunner, type PreviewScriptRunner, previewScriptRunners } from './preview-script-runner'

function nudgeRunner(run: PreviewScriptRunner, stage: WatchStage): void {
  void run(`(function () {
  var w = window;
  var fn = w.__hermesWatch_fn;
  if (!fn) { return 'cold'; }
  try { fn(document, w.__hermesActHolder || {}, ${JSON.stringify(stage)}); } catch (err) {}
  return 'ok';
})()`).catch(() => {
    // The page navigated out from under us. The next action re-injects.
  })
}

/** Run one stage against the active pane's overlay, if it has one. */
export function nudgeOverlay(stage: WatchStage): void {
  const run = activePreviewScriptRunner()

  if (!run) {
    return
  }

  nudgeRunner(run, stage)
}

/** Stop the idle pulse everywhere it could still be running. `think` belongs to
 * the active page; `rest` is cleanup and must survive a tab switch. */
export function restOverlays(): void {
  for (const run of previewScriptRunners()) {
    nudgeRunner(run, 'rest')
  }
}
