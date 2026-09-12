// @ts-check
/**
 * Launch the Hermes desktop app from a captured launch spec and click the
 * real update flow: Settings -> About -> "Update now".
 *
 * The spec is written by launch-capture/sitecustomize.py at `hermes
 * desktop`'s own spawn site, so argv, cwd, and the fully-constructed env
 * are the product's own -- this launcher only translates the npm-exec
 * source shape into a direct electron binary path (Playwright needs a
 * real executable, and the electron npm shim would re-spawn out of our
 * control).
 *
 * Usage (from the scratch dir where the driver installed @playwright/test):
 *   node launch-from-spec.mjs --spec /path/launch-spec.json \
 *     [--result $HERMES_HOME/.hermes-update-result.json] \
 *     [--expect-sha <sha> --repo-dir <install dir>] [--no-update]
 *
 * --no-update: launch + wait for the window + close. The smoke arm.
 * Otherwise: click Update now, then require a new successful handoff result
 * or source-update receipt, the expected checkout, and marker removal.
 * A checkout reset alone is not completion. No driver-assisted relaunch,
 * force-kill, or rebuild may turn an unfinished update into a passing one.
 * The Playwright close event is unreliable across the update handoff, so
 * neither signal is an app event.
 */

import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { parseArgs } from 'node:util';
import { _electron } from '@playwright/test';
import { prepareWindowForInput } from './window-input.cjs';
import { observeSourceUpdate } from './source-update-observer.mjs';

/**
 * @typedef {{argv: string[], cwd: string, env: Record<string, string>,
 *            matchedShape: 'source' | 'packaged'}} LaunchSpec
 */

/**
 * Resolve what _electron.launch needs from a captured spec.
 * @param {LaunchSpec} spec
 * @returns {{executablePath: string, args: string[], cwd: string,
 *            env: Record<string, string>}}
 */
export function resolveLaunch(spec) {
  if (spec.matchedShape === 'packaged') {
    return {
      executablePath: spec.argv[0],
      args: spec.argv.slice(1),
      cwd: spec.cwd,
      env: spec.env,
    };
  }
  // Source shape: ["npm", "exec", "--", "electron", ".", ...extra] running
  // in apps/desktop. Electron's real binary lives in the workspace-hoisted
  // node_modules; `electron/index.js` exports its path but requires the
  // module -- cheaper here to read the path file it derives from.
  const desktopDir = spec.cwd;
  const idx = spec.argv.findIndex((t) => t === 'electron');
  const extra = idx >= 0 ? spec.argv.slice(idx + 1).filter((t) => t !== '.') : [];
  const candidates = [
    path.join(desktopDir, 'node_modules', 'electron'),
    path.join(desktopDir, '..', '..', 'node_modules', 'electron'),
  ];
  for (const moduleDir of candidates) {
    const pathTxt = path.join(moduleDir, 'path.txt');
    if (!fs.existsSync(pathTxt)) continue;
    const rel = fs.readFileSync(pathTxt, 'utf8').trim();
    const exe = path.join(moduleDir, 'dist', rel);
    if (fs.existsSync(exe)) {
      return { executablePath: exe, args: ['.', ...extra], cwd: desktopDir, env: spec.env };
    }
  }
  throw new Error(`no electron binary found under ${candidates.join(' or ')}`);
}

/** @param {string} msg */
function log(msg) {
  console.log(`[launch-from-spec] ${msg}`);
}

// Coarse phase marker for the self-deadline's post-mortem line.
let currentPhase = 'init';
/** @param {string} p */
function phase(p) {
  currentPhase = p;
}

async function main() {
  // SIGKILLed Electron leaves Playwright connections and inherited pipes
  // holding node's event loop open, so the driver can outlive its own
  // finished test. Success and failure paths exit explicitly; this unref'd
  // timer is the backstop so no unknown state holds a runner past its budget.
  const SELF_DEADLINE_MS = 20 * 60 * 1000;
  const selfDeadline = setTimeout(() => {
    log(`DRIVER SELF-TIMEOUT after ${SELF_DEADLINE_MS / 60000}min - exiting 124 (phase: ${currentPhase})`);
    process.exit(124);
  }, SELF_DEADLINE_MS);
  selfDeadline.unref();

  const { values } = parseArgs({
    options: {
      spec: { type: 'string' },
      result: { type: 'string' },
      'expect-sha': { type: 'string' },
      'repo-dir': { type: 'string' },
      'no-update': { type: 'boolean', default: false },
      'timeout-ms': { type: 'string', default: '600000' },
    },
  });
  if (!values.spec) throw new Error('--spec is required');
  /** @type {LaunchSpec} */
  const spec = JSON.parse(fs.readFileSync(values.spec, 'utf8'));
  const launch = resolveLaunch(spec);
  log(`launching ${launch.executablePath} (shape: ${spec.matchedShape})`);

  phase('launch');
  const app = await _electron.launch({
    executablePath: launch.executablePath,
    args: launch.args,
    cwd: launch.cwd,
    env: launch.env,
  });
  // The app spawns several BrowserWindows (wake indicator, helper surfaces)
  // and firstWindow() grabs whichever webContents came first, which is not
  // always the main app window. Pick the window that actually renders the
  // app UI (a button renders only in the real renderer), retrying as
  // windows appear.
  await app.firstWindow({ timeout: 120_000 });
  let window = null;
  const windowDeadline = Date.now() + 120_000;
  while (!window) {
    for (const candidate of app.windows()) {
      const hasUi = await candidate
        .evaluate(() => document.querySelector('button') !== null)
        .catch(() => false);
      if (hasUi) { window = candidate; break; }
    }
    if (!window) {
      if (Date.now() > windowDeadline) {
        for (const c of app.windows()) log(`  window seen: url=${c.url()}`);
        throw new Error('no window with app UI (a <button>) appeared within 120s');
      }
      await new Promise((r) => setTimeout(r, 1_000));
    }
  }
  await window.waitForLoadState('domcontentloaded');
  log(`window up: ${await window.title()} (${app.windows().length} windows, picked url=${window.url()})`);
  await window.screenshot({ path: `${values.spec}.window.png` }).catch(() => {});

  await prepareWindowForInput(app, window);
  log('[zoom] app window prepared at 100%');

  if (values['no-update']) {
    log('smoke mode: window proven, closing');
    await app.close().catch(() => {});
    process.exit(0);
  }

  if (!values.result && !(values['expect-sha'] && values['repo-dir'])) {
    throw new Error('need --result and/or --expect-sha + --repo-dir unless --no-update');
  }
  const deadline = Date.now() + Number(values['timeout-ms']);


  // Dismiss the onboarding overlay when present. The drivers seed a
  // provider so the overlay SHOULD never mount, but it has a real boot
  // window: the renderer inits `configured` from a localStorage cache
  // (null on a fresh install) and only flips after gateway probes, so the
  // overlay can mount late - first as a buttonless boot-progress card,
  // then as the provider picker with the real escape hatch, "I'll choose
  // a provider later" (i18n en: chooseLater). Two traps this loop avoids:
  // a one-shot dismiss probe loses to the late mount, and visibility is
  // the wrong readiness signal - the settings gear is "visible" UNDER the
  // fullscreen overlay while the overlay intercepts every click. So:
  // alternate short-timeout dismiss clicks with short-timeout settings
  // clicks until a settings click actually LANDS (Playwright's hit-target
  // check makes a landed click proof the overlay is gone).
  phase('overlay-loop');
  const later = window.getByRole('button', { name: /choose a provider later|skip/i }).first()
  const settingsButton = window.getByRole('button', { name: /open settings|settings/i }).first()

  const overlayDeadline = Date.now() + 180_000
  let settingsOpened = false
  const brief = (e) => String(e && e.message || e).split('\n').slice(0, 25).join(' | ')
  // When a settings click fails, record what wins the hit-test at the
  // button's center plus the titlebar geometry, so a CI-only interception
  // is attributable from the log alone.
  const hitDump = () => window.evaluate(() => {
    const describe = (el) => el ? {
      tag: el.tagName,
      cls: (typeof el.className === 'string' ? el.className : '').slice(0, 110),
      aria: el.getAttribute?.('aria-label') || null,
      z: (() => { try { return getComputedStyle(el).zIndex } catch { return null } })(),
    } : null
    const settings = document.querySelector('button[aria-label="Open settings"]')
    const r = settings?.getBoundingClientRect()
    const cluster = settings?.closest('div[class*="fixed"]')
    const bar = document.querySelector('div[class*="h-[34px]"]')
    const cs = getComputedStyle(document.documentElement)
    const rect = (el) => { if (!el) return null; const b = el.getBoundingClientRect(); return `${Math.round(b.x)},${Math.round(b.y)} ${Math.round(b.width)}x${Math.round(b.height)}` }
    return {
      settingsRect: rect(settings),
      stack: r ? document.elementsFromPoint(r.x + r.width / 2, r.y + r.height / 2).slice(0, 6).map(describe) : null,
      cluster: cluster ? { rect: rect(cluster), z: getComputedStyle(cluster).zIndex, cls: (cluster.className || '').slice(0, 120) } : null,
      bar: bar ? { rect: rect(bar), z: getComputedStyle(bar).zIndex } : null,
      vars: {
        controlsLeft: cs.getPropertyValue('--titlebar-controls-left'),
        toolsRight: cs.getPropertyValue('--titlebar-tools-right'),
        toolsWidth: cs.getPropertyValue('--titlebar-tools-width'),
      },
      win: `${window.innerWidth}x${window.innerHeight} dpr=${window.devicePixelRatio}`,
    }
  }).then((d) => JSON.stringify(d)).catch((e) => `hit-dump failed: ${e.message}`)
  for (let iter = 1; ; iter++) {
    await prepareWindowForInput(app, window);
    await later
      .click({ timeout: 2_000 })
      .then(async () => {
        log('dismissed onboarding overlay')
        await later.waitFor({ state: 'hidden', timeout: 15_000 }).catch(() => {})
      })
      .catch((e) => log(`[overlay] iter ${iter} chooseLater click failed: ${brief(e)}`))
    try {
      await settingsButton.click({ timeout: 4_000 })
      // A landed click during shell hydration can be lost on a remount.
      // Confirm the destination before looking for its About control.
      await window.waitForURL(/[#/]settings(?:[/?]|$)/, { timeout: 4_000 })
      settingsOpened = true
      break
    } catch (e) {
      log(`[overlay] iter ${iter} settings click failed: ${brief(e)}`)
      // Every 5th failure, log the hit-test stack (every iteration would be
      // noise; the interceptor identity is what matters, not its frequency).
      if (iter === 1 || iter % 5 === 0) {
        log(`[overlay] iter ${iter} hit-test: ${await hitDump()}`)
      }
    }
    if (Date.now() > overlayDeadline) break
  }
  if (!settingsOpened) {
    await window.screenshot({ path: `${values.spec}.overlay-stuck.png` }).catch(() => {})
    throw new Error('onboarding overlay never cleared: Settings not clickable within 180s')
  }

  phase('about-update');
  // Settings is open: About -> Update now.
  await window.getByRole('tab', { name: /about/i }).or(
    window.getByRole('button', { name: /about/i })).first().click();
  const updateNow = window.getByRole('button', { name: /update now/i }).first();
  // "Update now" only renders once a check reports behind > 0, and the
  // About panel starts at "Last checked: never". The boot-time auto-check
  // can also fail transiently and latch the error UI, while a fresh check
  // succeeds. Nudge like an impatient user: click Check now whenever it is
  // clickable (not a spinner), re-test Update now, 3 minute ceiling.
  const checkNow = window.getByRole('button', { name: /check now/i }).first();
  const nudgeDeadline = Date.now() + 180_000;
  let updateVisible = await updateNow.isVisible().catch(() => false);
  while (!updateVisible && Date.now() < nudgeDeadline) {
    await checkNow.click({ timeout: 5_000 })
      .then(() => log('nudged Check now'))
      .catch(() => {}); // spinner or mid-transition - fine, just wait
    await window.waitForTimeout(15_000);
    updateVisible = await updateNow.isVisible().catch(() => false);
  }
  try {
    await updateNow.waitFor({ state: 'visible', timeout: 15_000 });
  } catch (e) {
    // The About UI flattens every check failure to a generic "couldn't
    // reach the update server", hiding the git stderr the main process
    // captured. Pull the full status over the same IPC the panel uses so
    // the log names the real error.
    const status = await window.evaluate(() =>
      window.hermesDesktop?.updates?.check?.() ?? Promise.resolve('no updates.check bridge')
    ).catch((err) => `updates.check failed: ${err?.message || err}`);
    log(`[update-status] ${JSON.stringify(status)}`);
    throw e;
  }
  const observe = observeSourceUpdate({
    home: spec.env.HERMES_HOME,
    resultPath: values.result,
    expectSha: values['expect-sha'],
  });
  await updateNow.click();
  phase('update-poll');
  log('clicked Update now; polling for result file');

  // The app may relaunch/exit during the update; completion signals are
  // product state, not Playwright events.
  const repoDir = values['repo-dir'];
  /** @returns {string} */
  const headSha = () => {
    try {
      return execFileSync('git', ['-C', /** @type {string} */ (repoDir), 'rev-parse', 'HEAD'], {
        encoding: 'utf8',
      }).trim();
    } catch {
      return '';
    }
  };
  for (;;) {
    if (observe(repoDir ? headSha() : '')) {
      log('successful update receipt and expected checkout observed');
      break;
    }
    if (Date.now() > deadline) {
      await window.screenshot({ path: `${values.spec}.timeout.png` }).catch(() => {});
      throw new Error('no successful update completion receipt; leaving the install untouched');
    }
    await new Promise((r) => setTimeout(r, 2_000));
  }

  // Record what the product did. The enclosing source driver verifies the
  // produced artifacts read-only; this helper does not repair or relaunch.
  phase('post-update');
  const handoff = await window.evaluate(() => {
    const text = document.body ? document.body.innerText : ''
    const m = text.match(/[^\n]*(update complete|reopen|relaunch)[^\n]*/i)
    return m ? m[0].trim().slice(0, 200) : null
  }).catch(() => null);
  log(handoff ? `post-update hand-off state: "${handoff}"` : 'post-update: no hand-off overlay observed (app may self-relaunch)');
  await window.screenshot({ path: `${values.spec}.post-update.png` }).catch(() => {});

  if (app.windows().length) {
    await Promise.race([
      app.close(),
      new Promise((_, reject) => setTimeout(() => reject(new Error('app did not close after completion; no force-kill attempted')), 15_000)),
    ]);
  }
  log('update completed without driver-assisted recovery (automatic relaunch not asserted here)');
  process.exit(0);
}

const invoked = process.argv[1] && path.resolve(process.argv[1]) === (await import('node:url')).fileURLToPath(import.meta.url);
if (invoked) {
  try {
    await main();
  } catch (error) {
    console.error(error);
    process.exit(1);
  }
}
