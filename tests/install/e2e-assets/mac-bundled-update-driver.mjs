// @ts-check
// mac-bundled-update-driver.mjs — click the REAL in-app update flow on the
// macOS packaged app: launch the installed OLD bundle binary under
// Playwright, reach Settings -> About, click "Update now", and wait for the
// app process to close.
//
// What this driver deliberately does NOT do:
//   - no internal apply call (no window.hermesDesktop.updates.apply or any
//     bridge invocation that would bypass the user trigger);
//   - no relaunch of the NEW app — Squirrel.Mac owns the swap and the
//     relaunch, and the external watcher
//     (mac-bundled-relaunch-watch.cjs) owns that proof;
//   - no killing of anything. The process-close contract from
//     process-close.cjs releases our stdio pipes instead of tree-killing,
//     so ShipIt's detached relaunch of the NEW bundle survives our exit.
//
// Usage (from the scratch dir with the driver's own @playwright/test):
//   node mac-bundled-update-driver.mjs --app-bin <.app/Contents/MacOS/Hermes> \
//     --shots <dir> --close-timeout-ms 420000

import fs from 'node:fs';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { _electron } from '@playwright/test';
import { createRequire } from 'node:module';
import { runUpdateWindowChat, updateWindowEnvironment } from './update-window-chat.mjs';

const require = createRequire(import.meta.url);
const { observeProcessClose } = require('./process-close.cjs');
const { prepareWindowForInput } = require('./window-input.cjs');
const { pickAppWindow, openAbout, waitForUpdate } = require('./update-ui.cjs');

const { values } = parseArgs({
  options: {
    'app-bin': { type: 'string' },
    shots: { type: 'string', default: '.' },
    'old-sha': { type: 'string' },
    'chat-out': { type: 'string' },
    'mock-url': { type: 'string' },
    'close-timeout-ms': { type: 'string', default: '420000' },
  },
});

const log = message => console.log(`[mac-bundled-update] ${message}`);
const shot = async (page, name) => {
  try { await page.screenshot({ path: path.join(values.shots, `${name}.png`), fullPage: true }); } catch { /* window may be gone */ }
};

const appBin = values['app-bin'];
if (!appBin || !values['old-sha'] || !values['chat-out'] || !values['mock-url']) {
  throw new Error('--app-bin, --old-sha, --chat-out and --mock-url are required');
}
fs.mkdirSync(values.shots, { recursive: true });
fs.mkdirSync(values['chat-out'], { recursive: true });

log(`launching ${appBin}`);
const app = await _electron.launch({
  executablePath: appBin,
  cwd: path.dirname(appBin),
  // Inherit the driver env: HERMES_HOME / HOME / updates feed config must
  // reach the main process exactly as a user's double-click would.
  env: updateWindowEnvironment(process.env, path.resolve(path.dirname(appBin), '..', 'Resources', 'agent-payload'), 'bundled'),
  timeout: 120_000,
});
const child = app.process();
const waitForProcessClose = observeProcessClose(child);
const oldPid = await app.evaluate(() => process.pid);
log(`launched Electron pid=${oldPid}`);
fs.writeFileSync(path.join(values['chat-out'], 'old-pid'), `${oldPid}\n`);

const page = await pickAppWindow(app, log);

await prepareWindowForInput(app, page);

await runUpdateWindowChat(app, page, {
  mockUrl: values['mock-url'], outDir: values['chat-out'],
  expectCommit: values['old-sha'],
  origin: 'bundled', executable: appBin,
  root: path.resolve(path.dirname(appBin), '..', 'Resources', 'agent-payload'),
});
await shot(page, '01-app-booted');

await openAbout(page, { log, shot, prepare: () => prepareWindowForInput(app, page) });
const updateNow = await waitForUpdate(page, { log, shot });

// ── The click under test ────────────────────────────────────────────────
await updateNow.click();
log('clicked: Update now');
await new Promise(resolve => setTimeout(resolve, 1_200));
await shot(page, '05-updating-overlay');

// The MacStrategy signs off with quitAndInstall: the app quits and
// Squirrel.Mac swaps the bundle and relaunches. Wait for the native close
// (never the renderer's close event) and exit WITHOUT killing anything —
// observeProcessClose released our pipes, so ShipIt's relaunch survives.
await waitForProcessClose(Number(values['close-timeout-ms']));
log('old Electron process closed — Squirrel.Mac owns the swap and relaunch');
fs.writeFileSync(path.join(values['chat-out'], 'old-exited'), new Date().toISOString() + '\n');
process.exit(0);
