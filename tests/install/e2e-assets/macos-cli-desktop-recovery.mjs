// #123737: a running source-built Desktop is an installed Applications copy,
// distinct from the build output that the CLI updater is allowed to replace.
import assert from 'node:assert/strict';
import { execFileSync, spawn } from 'node:child_process';
import { once } from 'node:events';
import fs from 'node:fs';
import path from 'node:path';

import { _electron } from '../../../tests-js/scripts/e2e-driver-deps.ts';
import { readChatIdentity, runDesktopChatSmoke, waitForChatReady } from '../../../tests-js/scripts/desktop-chat-smoke.ts';
import { assertBackendOrigin, localBackendProcess, readInstallationCommit, reusedDesktopBuildCommit } from '../../../tests-js/scripts/desktop-smoke-process.ts';
import { observeProcessClose } from './process-close.cjs';
import { isolatedElectronArgs, updateWindowEnvironment, within } from './smoke-env.mjs';
import { openAbout, pickAppWindow } from './update-ui.cjs';
import { runUpdateWindowChat } from './update-window-chat.mjs';
import { prepareWindowForInput } from './window-input.cjs';

const [packaged, oldSha, targetSha, outDir, separator, command, ...args] = process.argv.slice(2);
const {
  GITHUB_ACTIONS, RUNNER_TEMP, HOME, HERMES_HOME, HERMES_DESKTOP_USER_DATA_DIR,
  HERMES_E2E_MOCK_URL: mockUrl,
} = process.env;
const root = process.cwd();
assert(process.platform === 'darwin' && GITHUB_ACTIONS === 'true' && RUNNER_TEMP, 'Requires a disposable native macOS Actions runner');
assert(separator === '--' && command && HOME && HERMES_HOME && HERMES_DESKTOP_USER_DATA_DIR && mockUrl);
for (const candidate of [root, HOME, HERMES_HOME, HERMES_DESKTOP_USER_DATA_DIR, outDir]) {
  assert(candidate && within(RUNNER_TEMP, candidate), 'Recovery fixture escapes the disposable runner directory');
}
assert(packaged && within(root, packaged), 'The old bundle must belong to the source checkout');

const appBundle = path.join(HOME, 'Applications', 'Hermes.app');
const executable = path.join(appBundle, 'Contents', 'MacOS', 'Hermes');
const resources = path.join(appBundle, 'Contents', 'Resources');
const stamp = () => JSON.parse(fs.readFileSync(path.join(resources, 'install-stamp.json'), 'utf8').replace(/^\uFEFF/, ''));
const archive = () => fs.statSync(path.join(resources, 'app.asar'));
const log = (message) => console.log(`[desktop-recovery] ${message}`);
const shot = (page, name) => page.screenshot({ path: path.join(outDir, `desktop-recovery-${name}.png`) });

async function launch() {
  const env = updateWindowEnvironment(process.env, root, 'source');
  // The app and CLI must agree on the per-user Applications directory.
  env.HOME = HOME;
  const app = await _electron.launch({
    executablePath: executable,
    args: isolatedElectronArgs([], HERMES_DESKTOP_USER_DATA_DIR),
    cwd: root,
    env,
    timeout: 120_000,
  });
  return { app, waitForClose: observeProcessClose(app.process()) };
}

async function quit(window) {
  await window.app.close();
  await window.waitForClose();
}

async function updateWhileOpen() {
  const transcript = fs.openSync(path.join(outDir, 'update.log'), 'w');
  try {
    const child = spawn(command, args, { cwd: root, stdio: ['ignore', transcript, transcript], timeout: 20 * 60_000 });
    const [code, signal] = await once(child, 'close');
    assert.equal(code, 0, `CLI update failed (exit=${code}, signal=${signal}); see update.log`);
  } finally {
    fs.closeSync(transcript);
  }
}

let running;
const receipt = path.join(outDir, 'desktop-recovery.json');
const chatDir = path.join(outDir, 'desktop-recovery-chat');
try {
  fs.mkdirSync(path.dirname(appBundle), { recursive: true });
  if (!fs.existsSync(appBundle)) execFileSync('/usr/bin/ditto', [packaged, appBundle]);
  assert(within(HOME, appBundle), 'Installed bundle escapes the test home');
  assert.equal(stamp().commit, oldSha, 'The Applications copy must start at the published old commit');
  const oldArchive = archive();

  running = await launch();
  const oldPage = await pickAppWindow(running.app, log);
  await prepareWindowForInput(running.app, oldPage);
  await runUpdateWindowChat(running.app, oldPage, {
    mockUrl, outDir: chatDir, expectCommit: oldSha, root, origin: 'source',
    executable, userData: HERMES_DESKTOP_USER_DATA_DIR,
  });
  const prior = JSON.parse(fs.readFileSync(path.join(chatDir, 'desktop-chat-old.json'), 'utf8'));
  const oldRoute = new URL(oldPage.url()).hash;
  assert.match(oldRoute, /^#\/.+/, 'The seed reply must belong to an addressable conversation');

  await updateWhileOpen();
  assert.equal(readInstallationCommit(root, 'source'), targetSha, 'CLI update did not advance the runtime checkout');
  assert.equal(stamp().commit, oldSha, 'CLI updater swapped the running app');
  assert.equal(archive().ino, oldArchive.ino, 'CLI updater replaced the running app archive');
  assert.equal(await running.app.evaluate(() => process.pid), running.app.process().pid, 'CLI updater closed the running app');

  await quit(running);
  running = undefined;
  // Exercise the same installed entry point and user data. A CLI status,
  // rebuild, or second update here would mask the stranded Desktop app.
  running = await launch();
  const newPage = await pickAppWindow(running.app, log);
  const reusedBuildCommit = reusedDesktopBuildCommit(root, 'source', targetSha);
  const installed = stamp().commit;
  assert(installed === targetSha || installed === reusedBuildCommit,
    `Normal Desktop quit and relaunch left the installed app at ${installed}; expected ${targetSha}`);
  const liveResources = await running.app.evaluate(() => process.resourcesPath);
  assert.equal(fs.realpathSync(liveResources), fs.realpathSync(resources), 'Relaunch opened a different bundle');
  await prepareWindowForInput(running.app, newPage);
  await openAbout(newPage, { prepare: () => prepareWindowForInput(running.app, newPage), log, shot, confirmSettings: true });
  await newPage.getByText('Connected backend', { exact: true }).waitFor({ timeout: 30_000 });
  assert.equal(await newPage.getByText('App build out of date', { exact: true }).isVisible(), false, 'Desktop still shows a stale-build warning');
  await shot(newPage, 'about');
  await newPage.getByRole('button', { name: 'Close settings' }).click();
  await newPage.evaluate((hash) => { window.location.hash = hash; }, oldRoute);
  await waitForChatReady(newPage);
  await newPage.locator('[data-role="user"]').filter({ hasText: prior.prompt }).first().waitFor({ state: 'visible', timeout: 30_000 });
  const identity = await readChatIdentity(newPage);
  assert.equal(fs.realpathSync(identity.hermesRoot), fs.realpathSync(root), 'Relaunched app resolved a different install');
  const connection = await newPage.evaluate(() => window.hermesDesktop.getConnection());
  const address = new URL(connection.baseUrl);
  assert(connection.mode === 'local' && ['127.0.0.1', 'localhost', '[::1]'].includes(address.hostname));
  const backend = localBackendProcess(Number(address.port), running.app.process().pid, identity.hermesHome);
  assertBackendOrigin(backend, root, 'source', { appReportedRoot: identity.hermesRoot, installedCommit: targetSha });
  const health = await fetch(new URL('/api/health', address), { redirect: 'error', signal: AbortSignal.timeout(5000) });
  assert(health.ok, 'Relaunched Desktop backend health check failed');
  assert.equal((await health.json()).commit, targetSha, 'Relaunched Desktop connected to pre-update runtime code');
  await runDesktopChatSmoke(newPage, { mockUrl, outDir: chatDir, phase: 'new', expectCommit: targetSha, provenanceCommit: targetSha, reusedBuildCommit });
  await quit(running);
  running = undefined;
  fs.writeFileSync(receipt, JSON.stringify({ status: 'passed', oldSha, targetSha, installed, backendPid: backend.pid }) + '\n');
  log('PASS: normal Desktop quit and relaunch finished the CLI update and resumed the original conversation');
} catch (error) {
  fs.writeFileSync(receipt, JSON.stringify({ status: 'failed', oldSha, targetSha, error: String(error) }) + '\n');
  throw error;
} finally {
  if (running) await quit(running).catch((error) => log(`cleanup: ${error.message}`));
}
