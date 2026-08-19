/**
 * PREVIEW ACT — performs the agent's interactions inside the preview pane's
 * guest page, so `act_preview` drives whatever web app is open in the in-app
 * browser.
 *
 * The guest page is out-of-process; nothing here can touch its DOM directly.
 * Instead each action injects the engine SOURCE over `executeJavaScript` (see
 * lib/preview-act/act-in-page.ts's self-containment contract), parked on a
 * window global alongside the ref holder so '@e5' still resolves on the next
 * call. Injection is idempotent and vanishes with the page — a navigation
 * drops the refs, which the engine reports rather than clicking the wrong node.
 *
 * A mutating action settles briefly and returns a fresh inventory, so the
 * click → re-read → click loop costs one round trip instead of two.
 *
 * Dynamic-imported by the gateway event handler so the engine payload stays
 * out of the boot path.
 */

import { actInPage, type PreviewActAction, type PreviewActResult } from '@/lib/preview-act/act-in-page'

import { activePreviewNav, type PreviewNavHandle } from './preview-nav'
import { activePreviewScriptRunner } from './preview-script-runner'

/** Verbs the pane owns; a guest page cannot drive its own history. */
const NAV_ACTIONS: readonly (keyof PreviewNavHandle)[] = ['back', 'forward', 'reload']

/** How long a click/type is given to land before the page is re-inventoried.
 *  Long enough for a framework re-render, short enough not to stall the turn. */
const SETTLE_MS = 400

const NOTHING_OPEN = 'No live page is open in the in-app browser — open one with open_preview first.'

/** Build the idempotent inject-and-run script for one action. */
function buildActScript(action: PreviewActAction, settleMs: number): string {
  return `(function () {
  var w = window;
  if (!w.__hermesAct) {
    w.__hermesActHolder = {};
    w.__hermesAct = (${actInPage.toString()});
  }
  var act = function (a) { return w.__hermesAct(document, w.__hermesActHolder, a); };
  var result = act(${JSON.stringify(action)});
  if (!result.success || ${settleMs} <= 0) {
    return Promise.resolve(JSON.stringify(result));
  }
  // Re-inventory after the page has had a moment to react, so the agent's next
  // ref is drawn from the DOM its own click produced.
  return new Promise(function (resolve) {
    setTimeout(function () {
      var after = act({ kind: 'elements' });
      result.elements = after.elements;
      result.url = after.url;
      result.title = after.title;
      resolve(JSON.stringify(result));
    }, ${settleMs});
  });
})()`
}

/** Run one action against the ACTIVE preview tab's page. `kind` is a bare
 *  string: the verb arrives off the wire, and the history ones never reach
 *  the in-page engine. */
export async function actOnActivePreview(
  action: Omit<PreviewActAction, 'kind'> & { kind: string }
): Promise<PreviewActResult> {
  const nav = NAV_ACTIONS.find(verb => verb === action.kind)

  if (nav) {
    const handle = activePreviewNav()

    if (!handle) {
      return { error: NOTHING_OPEN, success: false }
    }

    handle[nav]()

    // Navigation is fire-and-forget through the webview; the new document has
    // its own refs, so the agent has to re-inventory either way.
    return { acted: nav, note: 'Page is loading — call elements to see what is on it.', success: true }
  }

  const run = activePreviewScriptRunner()

  if (!run) {
    return { error: NOTHING_OPEN, success: false }
  }

  const typed = action as PreviewActAction
  const settle = typed.kind === 'elements' ? 0 : SETTLE_MS
  const raw = await run(buildActScript(typed, settle))

  if (typeof raw !== 'string' || !raw) {
    return { error: 'The page did not answer the action.', success: false }
  }

  return JSON.parse(raw) as PreviewActResult
}
