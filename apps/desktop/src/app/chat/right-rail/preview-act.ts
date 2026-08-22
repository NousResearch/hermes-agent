/**
 * PREVIEW ACT — performs the agent's interactions inside the preview pane's
 * guest page, so `drive_preview` drives whatever web app is open in the in-app
 * browser.
 *
 * The guest page is out-of-process; nothing here can touch its DOM directly.
 * Two separate channels reach it, and the split matters:
 *
 *   - `executeJavaScript` injects the engine SOURCE (see
 *     lib/preview-act/act-in-page.ts's self-containment contract) to RESOLVE
 *     and READ — turn 'btn-sign-in' into a node, measure it, inventory the
 *     page. It is parked on a window global alongside the book of handles so
 *     they survive between calls, and it vanishes with the page, so a
 *     navigation retires them and the engine reports that instead of acting on
 *     the wrong node.
 *   - `sendInputEvent` (preview-drive.ts) does the ACTING, as real Chromium
 *     input. Script can only dispatch synthetic events, which the page can tell
 *     apart and which never move the browser's own hover or focus target.
 *
 * Panes with no input channel — a remote HTML preview — fall back to the
 * engine's synthetic events, which is worse but still works.
 *
 * A mutating action settles briefly and returns a fresh inventory, so the
 * click → re-read → click loop costs one round trip instead of two.
 *
 * Dynamic-imported by the gateway event handler so the engine payload stays
 * out of the boot path.
 */

import { actEngineSource, type PreviewActAction, type PreviewActResult } from '@/lib/preview-act/act-in-page'
import { watchInPage } from '@/lib/preview-act/watch-in-page'

import { clickAt, glideTo, pointerPlaced, pressKey, selectAll, typeText, wheelBy } from './preview-drive'
import { activePreviewInput, type PreviewInputHandle } from './preview-input'
import { activePreviewNav, type PreviewNavHandle } from './preview-nav'
import { activePreviewScriptRunner, type PreviewScriptRunner } from './preview-script-runner'

/** Verbs the pane owns; a guest page cannot drive its own history. */
const NAV_ACTIONS: readonly (keyof PreviewNavHandle)[] = ['back', 'forward', 'reload']

/** How long a click/type is given to land before the page is re-inventoried.
 *  Long enough for a framework re-render, short enough not to stall the turn.
 *  A render that misses this window is not lost — the NEXT action re-reads the
 *  page anyway, so the cost of being early is a slightly stale inventory, not a
 *  wrong one. That trade is worth several hundred ms on every single step. */
const SETTLE_MS = 220

/** Cap on one round trip into the guest page. A click that starts a navigation
 *  tears the document down mid-settle, so the injected promise dies with it and
 *  `executeJavaScript` never settles — without this the tool eats its whole
 *  45s bridge deadline for what was actually a successful click. */
const ACT_TIMEOUT_MS = 8_000

/** Verbs that get the look-then-act treatment: locate the target, walk the
 *  pointer there, then send real input. Actions with no single target (scroll,
 *  elements) are absent and skip it. */
const DRIVEN: readonly string[] = ['click', 'hover', 'press', 'type']

/** Verbs that put the mouse button down, and so are the only ones the pointerdown
 *  witness can speak to. `press` and `hover` never click, so demanding a witness
 *  from them would report every one of them as a failure. */
const CLICKS: readonly string[] = ['click', 'type']

const NOTHING_OPEN = 'No live page is open in the in-app browser — open one with open_preview first.'

const NAVIGATED =
  'The page stopped answering right after — it is probably navigating. Call elements to see where you landed.'

/** A fingerprint of the overlay's source, so the guest page can tell that the
 *  code it is running has changed underneath it.
 *
 *  It needs one because the overlay memoizes its own chrome: the cursor, the
 *  lock frame and the scan line are built ONCE per page and then reused, so an
 *  edit to any of their styles lands in the injected source, gets injected, and
 *  changes nothing — the layers it would have styled were built by the previous
 *  version and are still on the page. In dev that reads as "HMR didn't pick up
 *  my CSS"; in production it is a long-lived tab pinned to whichever build first
 *  touched it, across an app upgrade.
 *
 *  Content-addressed rather than a length or a version literal: the change that
 *  exposed this was one hex colour for another, which is byte-for-byte the same
 *  length, and a hand-maintained version is a step someone forgets. */
const WATCH_TAG = (() => {
  const source = watchInPage.toString()
  let hash = 5381

  for (let i = 0; i < source.length; i++) {
    hash = ((hash << 5) + hash + source.charCodeAt(i)) | 0
  }

  return hash
})()

/** Injected once per page: the engine, the overlay, and the two helpers every
 *  script below shares. Idempotent — re-running it on a live page is a no-op. */
const preamble = () => `  var w = window;
  // Reassigned on every trip rather than cached behind a guard: the source is in
  // this payload either way, so the guard saved nothing and pinned a long-lived
  // tab to whichever build first touched it. The holder is the exception — it
  // carries the aimed element from the locate trip to the act trip.
  w.__hermesActHolder = w.__hermesActHolder || {};
  w.__hermesAct = ${actEngineSource()};
  w.__hermesWatch_fn = (${watchInPage.toString()});
  w.__hermesWatchTag = ${WATCH_TAG};
  var holder = w.__hermesActHolder;
  var act = function (a) { return w.__hermesAct(document, holder, a); };
  var watch = function (stage, label) {
    try { w.__hermesWatch_fn(document, holder, stage, label); } catch (err) {}
  };
  var wait = function (ms) { return new Promise(function (resolve) { setTimeout(resolve, ms); }); };
  // Sit out a smooth scroll so the pointer is aimed at a target that has stopped
  // moving. A scroll that never started fires no scrollend, so only wait on one
  // we can actually see happening; capture catches nested scrollers, whose
  // scrollend does not bubble.
  var restAfterScroll = function () {
    return new Promise(function (resolve) {
      var sc = document.scrollingElement || document.documentElement;
      var x0 = sc ? sc.scrollLeft : 0;
      var y0 = sc ? sc.scrollTop : 0;
      requestAnimationFrame(function () {
        if (!sc || (sc.scrollLeft === x0 && sc.scrollTop === y0)) { return resolve(); }
        var done = false;
        var finish = function () {
          if (done) { return; }
          done = true;
          document.removeEventListener('scrollend', finish, true);
          clearTimeout(timer);
          resolve();
        };
        var timer = setTimeout(finish, 350);
        document.addEventListener('scrollend', finish, true);
      });
    });
  };`

/** Bring the target on screen, mark it, and report where it came to rest. */
function buildLocateScript(action: PreviewActAction, focus: boolean): string {
  const locate = { focus, kind: 'locate', ref: action.ref, selector: action.selector }

  return `(function () {
${preamble()}
  var locate = ${JSON.stringify(locate)};
  var found = act(locate);
  if (!found.success) { return Promise.resolve(JSON.stringify(found)); }
  watch('aim');
  // Arm a witness for the real input that is about to arrive. Without it a
  // click that never reached the page is indistinguishable from one the page
  // ignored, and the agent would report success either way.
  w.__hermesHit = null;
  document.addEventListener('pointerdown', function (e) {
    w.__hermesHit = { tag: e.target ? e.target.tagName : '?', trusted: e.isTrusted === true };
  }, { capture: true, once: true });
  // Measure again once the scroll has stopped: real input is aimed at a fixed
  // viewport coordinate, so it has to be where the target ENDS UP.
  return restAfterScroll().then(function () {
    var settled = act(locate);
    var best = settled.success ? settled : found;
    var at = best.point;
    // Last line of defence before the pointer is sent somewhere real. An element
    // that is still outside the viewport after we scrolled to it is hidden, not
    // placed, and aiming at it would drive the cursor off into a corner and
    // click whatever happens to be under that coordinate.
    if (at && (at.x < 0 || at.y < 0 || at.x > window.innerWidth || at.y > window.innerHeight)) {
      return JSON.stringify({
        error: 'That element is still off-screen after scrolling to it, so it is hidden rather than clickable. Call elements again for what is really on the page.',
        success: false
      });
    }
    return JSON.stringify(best);
  });
})()`
}

/** Put up a mark that outlives the action that made it. Every other cue on the
 *  overlay retires on a timer, which is right for narrating a click and no use
 *  at all for holding a finding on screen while the agent keeps working. */
function buildPinScript(action: PreviewActAction, label: string): string {
  const locate = { kind: 'locate', ref: action.ref, selector: action.selector }

  return `(function () {
${preamble()}
  var found = act(${JSON.stringify(locate)});
  if (!found.success) { return JSON.stringify(found); }
  watch('pin', ${JSON.stringify(label)});
  return JSON.stringify({ acted: 'pinned ' + String(found.acted || 'it').replace(/^looking at /, ''), success: true });
})()`
}

/** Freeze the whole visible field as marks. Needs a fresh inventory first, so
 *  the overlay has both the field to draw and the refs to number it by. */
function buildHoldScript(): string {
  return `(function () {
${preamble()}
  var found = act({ kind: 'elements' });
  if (!found.success) { return JSON.stringify(found); }
  watch('hold');
  found.acted = 'held the field';
  return JSON.stringify(found);
})()`
}

/** Rattle through the field one box at a time. Needs a fresh inventory for the
 *  overlay to pick from, and nothing else — the page is never touched. */
function buildStrobeScript(): string {
  return `(function () {
${preamble()}
  var found = act({ kind: 'elements' });
  if (!found.success) { return JSON.stringify(found); }
  watch('strobe');
  found.acted = 'strobed the field';
  return JSON.stringify(found);
})()`
}

/** Take one mark down, or — with nothing to aim at — all of them. */
function buildUnpinScript(action: PreviewActAction): string {
  const locate = { kind: 'locate', ref: action.ref, selector: action.selector }
  const one = !!(action.ref || action.selector)

  return `(function () {
${preamble()}
${
  one
    ? `  var found = act(${JSON.stringify(locate)});
  if (!found.success) { return JSON.stringify(found); }
  var gone = 'unpinned ' + String(found.acted || 'it').replace(/^looking at /, '');`
    : `  holder.aimed = null;
  var gone = 'cleared every pin';`
}
  watch('unpin');
  return JSON.stringify({ acted: gone, success: true });
})()`
}

/** Mark the hit, let the page react, and hand back a fresh inventory. */
function buildFinishScript(settleMs: number): string {
  return `(function () {
${preamble()}
  watch('strike');
  return wait(${settleMs}).then(function () {
    // A rescan that throws must still answer — an unresolved promise here costs
    // the whole bridge deadline.
    try {
      var out = act({ kind: 'elements' });
      watch('sweep');
      out.hit = w.__hermesHit || null;
      return JSON.stringify(out);
    } catch (err) {
      return JSON.stringify({ note: 'The page changed before it could be re-read: ' + err, success: true });
    }
  });
})()`
}

/** The no-real-input fallback: the engine both acts and reads, in one trip.
 *  Both reads mark the field — the explicit `elements` call and the re-read
 *  every other verb does on its way out — so the page is marked after every
 *  single action rather than only when the agent asks for an inventory outright.
 *  They mark it differently, though, and the difference is what each one MEANS:
 *  an outright `elements` is the agent reading the whole page, which is the
 *  moment worth showing, so it strobes. The re-read after a click is
 *  housekeeping, and strobing there would put five seconds of noise between
 *  every step of a task, so it sweeps. The two never collide: an `elements` call
 *  settles in 0ms and returns before the re-read. */
function buildScriptedScript(action: PreviewActAction, settleMs: number): string {
  return `(function () {
${preamble()}
  var result = act(${JSON.stringify(action)});
  ${action.kind === 'elements' ? "if (result.success) { watch('strobe'); }" : ''}
  if (!result.success || ${settleMs} <= 0) { return Promise.resolve(JSON.stringify(result)); }
  return wait(${settleMs}).then(function () {
    try {
      var after = act({ kind: 'elements' });
      watch('sweep');
      // One or the other, never both: a re-read answers with the whole
      // inventory only when it is the first look at this page.
      result.elements = after.elements;
      result.delta = after.delta;
      result.url = after.url;
      result.title = after.title;
    } catch (err) {
      result.note = 'The page changed before it could be re-read: ' + err;
    }
    return JSON.stringify(result);
  });
})()`
}

/** The outcome of one round trip into the page. `silent` is its own case on
 *  purpose: a page that stops answering mid-action is navigating, whereas one
 *  that answers with nothing is broken, and the agent needs to hear the
 *  difference. */
type Trip = { error: string; kind: 'failed' } | { kind: 'answered'; result: PreviewActResult } | { kind: 'silent' }

async function runJson(run: PreviewScriptRunner, code: string): Promise<Trip> {
  const raw = await Promise.race([
    run(code).catch((error: unknown) => new Error(String(error))),
    new Promise<undefined>(resolve => setTimeout(resolve, ACT_TIMEOUT_MS))
  ])

  if (raw === undefined) {
    return { kind: 'silent' }
  }

  if (raw instanceof Error) {
    return { error: 'The page rejected the action: ' + raw.message, kind: 'failed' }
  }

  if (typeof raw !== 'string' || !raw) {
    return { error: 'The page did not answer the action.', kind: 'failed' }
  }

  return { kind: 'answered', result: JSON.parse(raw) as PreviewActResult }
}

/** Past tense of the verb the agent asked for, against what it actually hit. */
function describeDone(action: PreviewActAction, target: string): string {
  if (action.kind === 'type') {
    return 'typed into ' + target + (action.submit ? ' and submitted' : '')
  }

  if (action.kind === 'press') {
    return 'pressed ' + (action.key || '') + ' on ' + target
  }

  if (action.kind === 'hover') {
    return 'hovered over ' + target
  }

  return 'clicked ' + target
}

/** Look at the target, walk the pointer over, and act on it for real. */
async function driveAction(
  run: PreviewScriptRunner,
  input: PreviewInputHandle,
  action: PreviewActAction
): Promise<PreviewActResult> {
  // A key press must not be preceded by a click — that would activate the
  // control rather than type into it — so the page hands it focus instead.
  const trip = await runJson(run, buildLocateScript(action, action.kind === 'press'))

  if (trip.kind === 'failed') {
    return { error: trip.error, success: false }
  }

  if (trip.kind === 'silent') {
    return { acted: action.kind, note: NAVIGATED, success: true }
  }

  const found = trip.result

  if (!found.success) {
    return found
  }

  if (!found.point) {
    return { error: 'Could not work out where that element is on screen.', success: false }
  }

  await glideTo(input, found.point)

  if (action.kind === 'click') {
    await clickAt(input)
  } else if (action.kind === 'type') {
    if (found.typable === false) {
      return {
        error: `${String(found.acted || 'That').replace(/^looking at /, '')} is not a text field, so typing into it would only select the text under the pointer. Click it if it opens one, then type into that.`,
        success: false
      }
    }

    input.focus()
    await clickAt(input)
    // Select-all inside the now-focused field, so typing replaces what is there
    // the way it would for a person. NOT a triple-click: that is a pointer
    // gesture and selects the paragraph under the cursor whenever the target
    // turns out not to be a field.
    await selectAll(input)
    await typeText(input, action.text ?? '')

    if (action.submit) {
      await pressKey(input, 'Enter')
    }
  } else if (action.kind === 'press') {
    input.focus()
    await pressKey(input, action.key || 'Enter')
  }
  // hover is the glide and nothing else — the pointer is already sitting on the
  // target, which is the whole request.

  const target = String(found.acted || '').replace(/^looking at /, '')
  const after = await runJson(run, buildFinishScript(SETTLE_MS))
  const acted = describeDone(action, target)

  // The action itself already happened as real input, so a page that will not
  // answer the read-back is a page that navigated — never a failed click.
  if (after.kind !== 'answered') {
    return { acted, note: NAVIGATED, success: true }
  }

  const { hit, ...result } = after.result as PreviewActResult & { hit?: { tag: string; trusted: boolean } | null }

  // The witness the locate trip armed. No record means the input never reached
  // the document, which the agent must hear about — every other signal here
  // travels on the script channel and would report success regardless.
  if (!hit && CLICKS.indexOf(action.kind) !== -1) {
    return {
      ...result,
      error: 'The pointer input never reached the page, so nothing was ' + acted.split(' ')[0] + '.',
      success: false
    }
  }

  return { ...result, acted, note: hitNote(hit), success: true }
}

/** Flag a click the overlay intercepted, which would otherwise look like a page
 *  that simply ignored it. */
function hitNote(hit?: { tag: string; trusted: boolean } | null): string | undefined {
  return hit && hit.tag === 'HERMES-WATCH' ? 'The action overlay intercepted the click instead of the page.' : undefined
}

/** How far a screenful is, whether there is anywhere to go, and a spot to wheel
 *  over — the renderer cannot see the guest viewport to work any of it out. */
function buildScrollAnchorScript(): string {
  return `(function () {
${preamble()}
  var sc = document.scrollingElement || document.documentElement;
  var track = sc.clientHeight || window.innerHeight;
  return Promise.resolve(JSON.stringify({
    page: Math.round(window.innerHeight * 0.9),
    point: { x: Math.round(window.innerWidth / 2), y: Math.round(track / 2) },
    span: sc.scrollHeight - track,
    success: true
  }));
})()`
}

/** Scroll the page the way a hand does — wheel it. The scripted path calls
 *  `scrollBy`, which moves the page without the page ever seeing an input
 *  event, so scroll-linked headers, reveal animations and infinite-scroll
 *  loaders all stay asleep through a scroll that looked like it happened. */
async function driveScroll(
  run: PreviewScriptRunner,
  input: PreviewInputHandle,
  action: PreviewActAction
): Promise<PreviewActResult> {
  const far = action.amount ?? 0

  const trip = await runJson(run, buildScrollAnchorScript())

  if (trip.kind === 'failed') {
    return { error: trip.error, success: false }
  }

  if (trip.kind === 'silent') {
    return { acted: 'scrolled', note: NAVIGATED, success: true }
  }

  const anchor = trip.result as PreviewActResult & { page?: number; span?: number }

  if (!anchor.span) {
    return { ...anchor, acted: 'scrolled the page', note: 'The page has nothing to scroll — it all fits already.' }
  }

  // A person does not move the mouse to scroll; the wheel turns wherever their
  // hand already is. Only send it somewhere if it has never been anywhere.
  if (!pointerPlaced() && anchor.point) {
    await glideTo(input, anchor.point)
  }

  await wheelBy(input, action.amount ?? anchor.page ?? 600)

  const after = await runJson(run, buildFinishScript(SETTLE_MS))

  if (after.kind !== 'answered') {
    return { acted: 'scrolled the page', note: NAVIGATED, success: true }
  }

  return { ...after.result, acted: 'scrolled the page', success: true }
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

  // Annotation, not interaction: nothing is clicked, nothing settles, and the
  // page is not re-read, so these skip the whole act-then-inventory path.
  if (typed.kind === 'pin' || typed.kind === 'unpin' || typed.kind === 'hold' || typed.kind === 'strobe') {
    const mark =
      typed.kind === 'hold'
        ? buildHoldScript()
        : typed.kind === 'strobe'
          ? buildStrobeScript()
          : typed.kind === 'pin'
            ? buildPinScript(typed, typed.text || '')
            : buildUnpinScript(typed)

    const trip = await runJson(run, mark)

    if (trip.kind === 'failed') {
      return { error: trip.error, success: false }
    }

    return trip.kind === 'answered' ? trip.result : { acted: typed.kind, note: NAVIGATED, success: true }
  }

  const input = activePreviewInput()

  if (input && DRIVEN.indexOf(typed.kind) !== -1) {
    return driveAction(run, input, typed)
  }

  // A plain page scroll is a wheel gesture. Jumping to an end is not — no hand
  // wheels to the bottom of a long article — so `to` stays a scripted glide.
  const plain = typed.kind === 'scroll' && !typed.to && !typed.ref && !typed.selector

  if (plain && input) {
    return driveScroll(run, input, typed)
  }

  const settle = typed.kind === 'elements' ? 0 : SETTLE_MS
  const scripted = await runJson(run, buildScriptedScript(typed, settle))

  if (scripted.kind === 'failed') {
    return { error: scripted.error, success: false }
  }

  // The action almost certainly landed — a page that stops answering right
  // after a click is one that navigated. Say so instead of failing it.
  return scripted.kind === 'silent' ? { acted: typed.kind, note: NAVIGATED, success: true } : scripted.result
}

// Self-accept so an edit here, or to the in-page sources this module
// stringifies, doesn't reload the whole renderer out from under a live session.
// The bridge re-requests this module per action in dev, so it picks the change
// up without one (see loadPreviewEngine in gateway-event/desktop-bridge.ts).
if (import.meta.hot) {
  import.meta.hot.accept()
}
