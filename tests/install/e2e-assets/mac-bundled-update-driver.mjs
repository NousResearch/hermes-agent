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

const require = createRequire(import.meta.url);
const { observeProcessClose } = require('./process-close.cjs');
const { prepareWindowForInput } = require('./window-input.cjs');

const { values } = parseArgs({
  options: {
    'app-bin': { type: 'string' },
    shots: { type: 'string', default: '.' },
    'close-timeout-ms': { type: 'string', default: '420000' },
  },
});

const log = message => console.log(`[mac-bundled-update] ${message}`);
const shot = async (page, name) => {
  try { await page.screenshot({ path: path.join(values.shots, `${name}.png`), fullPage: true }); } catch { /* window may be gone */ }
};

const appBin = values['app-bin'];
fs.mkdirSync(values.shots, { recursive: true });

log(`launching ${appBin}`);
const app = await _electron.launch({
  executablePath: appBin,
  // Inherit the driver env: HERMES_HOME / HOME / updates feed config must
  // reach the main process exactly as a user's double-click would.
  env: { ...process.env },
  timeout: 120_000,
});
const child = app.process();
const oldPid = child.pid;
log(`launched Electron pid=${oldPid}`);
fs.writeFileSync(path.join(values.shots, 'old-pid'), `${oldPid}\n`);

const waitForProcessClose = observeProcessClose(child);

await app.firstWindow({ timeout: 120_000 });
let page = null;
const windowDeadline = Date.now() + 120_000;
while (!page) {
  for (const candidate of app.windows()) {
    const hasUi = await candidate.evaluate(() => document.querySelector('button') !== null).catch(() => false);
    if (hasUi) { page = candidate; break; }
  }
  if (!page) {
    if (Date.now() > windowDeadline) throw new Error('no window with app UI (a <button>) appeared within 120s');
    await new Promise(r => setTimeout(r, 1_000));
  }
}
log(`window picked (url=${page.url()})`);

await prepareWindowForInput(app, page);

// Boot: the shell is mounted once the composer exists.
await page.waitForSelector('textarea, [contenteditable="true"]', { state: 'attached', timeout: 300_000 });
log('renderer booted (composer attached)');
await page.waitForTimeout(3_000);
await shot(page, '01-app-booted');

// Reach Settings through the onboarding overlay, if it shows. A click that
// LANDS is proof the overlay is gone (Playwright checks the hit target).
const clickFirstVisible = async (locators, description, timeoutMs) => {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    for (const make of locators) {
      const locator = make(page).first();
      try {
        if (await locator.isVisible()) {
          await locator.click();
          log(`clicked: ${description}`);
          return true;
        }
      } catch { /* state moved on; try the next */ }
    }
    if (Date.now() > deadline) return false;
    await page.waitForTimeout(500);
  }
};

const laterLocators = [
  p => p.getByRole('button', { name: /choose a provider later/i }),
  p => p.getByText(/choose a provider later/i),
  p => p.getByRole('button', { name: /skip/i }),
];
const settingsLocators = [
  p => p.getByLabel('Open settings'),
  p => p.locator('[aria-label="Open settings"]'),
  p => p.locator('[title="Open settings"]'),
  p => p.getByRole('button', { name: 'Open settings' }),
];

let openedSettings = false;
const overlayDeadline = Date.now() + 180_000;
while (!openedSettings) {
  await prepareWindowForInput(app, page);
  for (const make of laterLocators) {
    try {
      await make(page).first().click({ timeout: 1_500 });
      log('dismissed onboarding overlay');
      await page.waitForTimeout(2_500);
      await shot(page, '01b-onboarding-dismissed');
      break;
    } catch { /* not up yet */ }
  }
  for (const make of settingsLocators) {
    try {
      await make(page).first().click({ timeout: 2_500 });
      openedSettings = true;
      log('clicked: Open settings');
      break;
    } catch { /* try again next pass */ }
  }
  if (!openedSettings && Date.now() > overlayDeadline) break;
}
if (!openedSettings) {
  await shot(page, 'ERROR-no-settings-button');
  throw new Error('could not find the Open settings control');
}
await page.waitForTimeout(1_500);
await shot(page, '02-settings-open');

if (!await clickFirstVisible([
  p => p.getByRole('tab', { name: 'About' }),
  p => p.getByRole('button', { name: 'About' }),
  p => p.getByText('About', { exact: true }),
], 'About section', 30_000)) {
  await shot(page, 'ERROR-no-about-tab');
  throw new Error('could not find the About section in Settings');
}
await page.waitForTimeout(1_500);
await shot(page, '03-about-panel');

// Wait for "Update now" (the loopback feed reports NEW as available). If
// the boot check has not landed yet, press "Check now" like a user would.
const updateNow = page.getByRole('button', { name: 'Update now' }).first();
let visible = await updateNow.isVisible().catch(() => false);
if (!visible) {
  log('Update now not visible yet — nudging Check now');
  const deadline = Date.now() + 180_000;
  while (!visible && Date.now() < deadline) {
    await page.getByRole('button', { name: 'Check now' }).first()
      .click({ timeout: 5_000 })
      .then(() => log('nudged Check now'))
      .catch(() => {});
    await page.waitForTimeout(15_000);
    visible = await updateNow.isVisible().catch(() => false);
  }
}
if (!visible) {
  await shot(page, 'ERROR-no-update-now');
  throw new Error('"Update now" never appeared — the loopback feed check did not report an update');
}
await shot(page, '04-update-available');

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
fs.writeFileSync(path.join(values.shots, 'old-exited'), new Date().toISOString() + '\n');
process.exit(0);
