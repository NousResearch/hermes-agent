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
 * Usage (current CI checkout with locked driver dependencies):
 *   node launch-from-spec.mjs --spec /path/launch-spec.json \
 *     [--result $HERMES_HOME/.hermes-update-result.json] \
 *     [--expect-sha <sha> --repo-dir <install dir>] [--no-update]
 *
 * --no-update: require a real desktop chat, then close gracefully.
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
import { assertStagedBranch, pickAppWindow, openAbout, readManualUpdateCommand, waitForUpdate } from './update-ui.cjs';
import { installSourceBranchProbe, prepareSourceBranchEnvironment } from './source-branch-probe.cjs';
import { observeSourceUpdate } from './source-update-observer.mjs';
import { runUpdateWindowChat } from './update-window-chat.mjs';
import { isolateUpdateWindowEnvironment, isolatedElectronArgs, updateWindowEnvironment } from './smoke-env.mjs';
import { sourceRuntimeSettleCommand } from './source-runtime-settle.mjs';

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

/**
 * Settle source runtime/package replacement before Playwright owns Electron.
 * A first non-metadata startup may replace the packaged app and relaunch it;
 * doing that after `_electron.launch` attaches loses the inspection pipe.
 *
 * @param {string} root
 * @param {Record<string, string>} env
 */
function settleSourceRuntime(root, env) {
  const invocation = sourceRuntimeSettleCommand(root, env);
  log('settling source runtime under the captured launch environment');
  execFileSync(invocation.command, invocation.args, {
    cwd: root, env, stdio: 'inherit', timeout: 20 * 60_000,
    windowsVerbatimArguments: invocation.windowsVerbatimArguments,
  });
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
      'old-sha': { type: 'string' },
      'chat-out': { type: 'string' },
      'mock-url': { type: 'string' },
      'no-update': { type: 'boolean', default: false },
      'timeout-ms': { type: 'string', default: '600000' },
    },
  });
  if (!values.spec) throw new Error('--spec is required');
  if (!values['old-sha'] || !values['chat-out'] || !values['mock-url'] || !values['repo-dir']) {
    throw new Error('--old-sha, --chat-out, --mock-url and --repo-dir are required for OLD chat');
  }
  /** @type {LaunchSpec} */
  const spec = JSON.parse(fs.readFileSync(values.spec, 'utf8'));
  const launch = resolveLaunch(spec);
  const capturedEnv = updateWindowEnvironment(launch.env, values['repo-dir'], 'source');
  settleSourceRuntime(values['repo-dir'], capturedEnv);
  const launchEnv = isolateUpdateWindowEnvironment(capturedEnv);
  if (!values['no-update']) {
    // This unpublished E2E target has no R2 channel record. Only the test
    // probe selects the real checker's explicit branch path.
    prepareSourceBranchEnvironment(values['repo-dir'], values['expect-sha'],
      process.env.HERMES_E2E_REAL_GIT, capturedEnv, launchEnv);
  }
  log(`launching ${launch.executablePath} (shape: ${spec.matchedShape}, isolated userData: ${launchEnv.HERMES_DESKTOP_USER_DATA_DIR})`);

  phase('launch');
  const app = await _electron.launch({
    executablePath: launch.executablePath,
    args: isolatedElectronArgs(launch.args, launchEnv.HERMES_DESKTOP_USER_DATA_DIR),
    cwd: launch.cwd,
    env: launchEnv,
  });
  if (!values['no-update']) await installSourceBranchProbe(app);
  const window = await pickAppWindow(app, log);
  await window.screenshot({ path: `${values.spec}.window.png` }).catch(() => {});

  await prepareWindowForInput(app, window);
  log('[zoom] app window prepared at 100%');

  phase('old-chat');
  await runUpdateWindowChat(app, window, {
    mockUrl: values['mock-url'], outDir: values['chat-out'],
    expectCommit: values['old-sha'],
    root: values['repo-dir'], origin: 'source', executable: launch.executablePath,
    userData: launchEnv.HERMES_DESKTOP_USER_DATA_DIR,
  });

  if (values['no-update']) {
    log('smoke mode: OLD desktop chat proven, closing');
    await app.close();
    process.exit(0);
  }

  if (!values.result && !(values['expect-sha'] && values['repo-dir'])) {
    throw new Error('need --result and/or --expect-sha + --repo-dir unless --no-update');
  }
  await assertStagedBranch(window, values['expect-sha'], log);
  const deadline = Date.now() + Number(values['timeout-ms']);


  phase('overlay-loop');
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
  const ui = {
    log,
    shot: (page, name) => page.screenshot({ path: `${values.spec}.${name}.png` }).catch(() => {}),
  };
  await openAbout(window, { ...ui, prepare: () => prepareWindowForInput(app, window), confirmSettings: true, hitDump });
  phase('about-update');
  const updateNow = await waitForUpdate(window, ui);
  const observe = observeSourceUpdate({
    home: spec.env.HERMES_HOME,
    resultPath: values.result,
    expectSha: values['expect-sha'],
  });
  await updateNow.click();
  phase('update-poll');
  log('clicked Update now; polling for result file');

  const manualCommand = await readManualUpdateCommand(window);
  if (manualCommand) {
    fs.mkdirSync(values['chat-out'], { recursive: true });
    fs.writeFileSync(
      path.join(values['chat-out'], 'manual-update.json'),
      `${JSON.stringify({ command: manualCommand, oldSha: values['old-sha'] }, null, 2)}\n`,
    );
    log(`OLD requires the manual update path: ${manualCommand}`);
    await app.close();
    process.exit(42);
  }

  // The app may relaunch/exit during the update; completion signals are
  // product state, not Playwright events.
  const repoDir = values['repo-dir'];
  /** @returns {string} */
  const headSha = () => {
    try {
      // The driver's real git: a fresh-machine leg takes every git off PATH
      // so the product must provision its own, and an observer that cannot
      // spawn git would read '' forever instead of failing.
      return execFileSync(process.env.HERMES_E2E_REAL_GIT || 'git', ['-C', /** @type {string} */ (repoDir), 'rev-parse', 'HEAD'], {
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
