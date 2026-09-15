// The caller owns OLD's lifetime. Record backend provenance in the same receipt
// as the shared chat assertion without launching a second app or provider.
import fs from 'node:fs';
import path from 'node:path';
import { readChatIdentity, runDesktopChatSmoke, waitForChatReady } from '../../../tests-js/scripts/desktop-chat-smoke.ts';
import { assertBackendOrigin, localBackendProcess, readInstallationCommit, within } from '../../../tests-js/scripts/desktop-smoke-process.ts';
import { smokeEnvironment } from './desktop-smoke.ts';

// Source updater processes still need the git redirect; smokeEnvironment keeps
// it while removing driver activation, credentials and remote/backend overrides.
/** @param {NodeJS.ProcessEnv} inherited @param {string} root @param {'source'|'bundled'} origin */
export function updateWindowEnvironment(inherited, root, origin) {
  const home = inherited.HERMES_HOME;
  const userData = inherited.HERMES_DESKTOP_USER_DATA_DIR;
  if (!home || !userData) throw new Error('Update chat requires isolated HERMES_HOME and HERMES_DESKTOP_USER_DATA_DIR');
  const env = smokeEnvironment(inherited, home, userData);
  // The detached source updater builds NEW in this environment too.
  for (const key of ['GITHUB_SHA', 'GITHUB_REF', 'GITHUB_REF_NAME', 'GITHUB_HEAD_REF', 'GITHUB_BASE_REF']) {
    delete env[key];
  }
  fs.mkdirSync(env.HOME, { recursive: true });
  fs.mkdirSync(userData, { recursive: true });
  for (const [filename, local] of [
    ['connection.json', { mode: 'local' }],
    ['connections.json', { primary: 'local', launchMode: 'primary', lastUsed: 'local' }],
  ]) {
    const file = path.join(userData, filename);
    if (filename === 'connections.json' && !fs.existsSync(file)) continue;
    const prior = fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, 'utf8')) : {};
    fs.writeFileSync(file, JSON.stringify({ ...prior, ...local }), { mode: 0o600 });
  }
  if (origin === 'source') {
    const editableRoot = inherited.HERMES_PYTHON_SRC_ROOT;
    if (editableRoot) {
      if (!path.isAbsolute(editableRoot) || fs.realpathSync(editableRoot) !== fs.realpathSync(root)) {
        throw new Error('Captured HERMES_PYTHON_SRC_ROOT differs from the installed source');
      }
      env.HERMES_PYTHON_SRC_ROOT = editableRoot;
    }
    for (const key of ['HERMES_DESKTOP_PYTHON', 'HERMES_DESKTOP_HERMES', 'HERMES_DESKTOP_HERMES_ROOT']) {
      if (inherited[key]) {
        if (!path.isAbsolute(inherited[key]) || (key !== 'HERMES_DESKTOP_PYTHON' && !within(root, inherited[key]))) throw new Error(`Captured ${key} escapes the installed source`);
        fs.accessSync(inherited[key]);
        env[key] = inherited[key];
      }
    }
  }
  return env;
}

/**
 * @param {import('@playwright/test').ElectronApplication} app
 * @param {import('@playwright/test').Page} page
 * @param {{mockUrl: string, outDir: string, expectCommit: string,
 *   origin: 'source'|'bundled', root: string, executable: string}} options
 */
export async function runUpdateWindowChat(app, page, options) {
  const receiptPath = path.join(options.outDir, 'desktop-chat-old.json');
  fs.mkdirSync(options.outDir, { recursive: true });
  try {
    const running = await app.evaluate(({ app: electronApp }) => ({
      pid: process.pid, executable: process.execPath, resources: process.resourcesPath,
      userData: electronApp.getPath('userData'),
    }));
    const { userData } = running;
    if (!process.env.HERMES_DESKTOP_USER_DATA_DIR || fs.realpathSync(userData) !== fs.realpathSync(process.env.HERMES_DESKTOP_USER_DATA_DIR)) {
      throw new Error('OLD update window did not honor isolated userData');
    }
    if (fs.realpathSync(running.executable) !== fs.realpathSync(options.executable)) {
      throw new Error('OLD update window executable differs from the installed app');
    }
    if (options.origin === 'bundled' && fs.realpathSync(path.join(running.resources, 'agent-payload')) !== fs.realpathSync(options.root)) {
      throw new Error('OLD update window resources differ from the installed payload');
    }
    await waitForChatReady(page);
    const identity = await readChatIdentity(page);
    const connection = await page.evaluate(() => window.hermesDesktop.getConnection());
    const base = new URL(connection.baseUrl);
    if (connection.mode !== 'local' || !['127.0.0.1', 'localhost', '[::1]'].includes(base.hostname)) {
      throw new Error('OLD update-window chat did not use the local installed backend');
    }
    if (options.origin === 'source' && fs.realpathSync(identity.hermesRoot) !== fs.realpathSync(options.root)) {
      throw new Error('OLD update-window chat resolved another source installation');
    }
    const backend = localBackendProcess(Number(base.port), running.pid);
    assertBackendOrigin(backend, options.root, options.origin);
    const provenanceCommit = readInstallationCommit(options.root, options.origin);
    if (provenanceCommit !== options.expectCommit) throw new Error('Installed OLD commit differs from the expected commit');
    const chat = await runDesktopChatSmoke(page, { ...options, phase: 'old', provenanceCommit });
    const result = { ...chat, origin: options.origin, root: options.root,
      executable: options.executable, appPid: running.pid,
      backend: { pid: backend.pid, parentPid: backend.parentPid, executable: backend.executable },
      localModeConfigured: true, userData,
      launchKind: 'pre-update-window' };
    fs.writeFileSync(receiptPath, JSON.stringify(result, null, 2) + '\n');
  } catch (error) {
    await page.screenshot({ path: path.join(options.outDir, 'desktop-chat-old-failed.png') }).catch(() => {});
    fs.writeFileSync(receiptPath, JSON.stringify({ status: 'failed', phase: 'old',
      expectedCommit: options.expectCommit, origin: options.origin, root: options.root,
      error: String(error) }, null, 2) + '\n');
    throw error;
  }
}