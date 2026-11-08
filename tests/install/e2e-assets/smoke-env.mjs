// Environment shaping shared by the desktop smoke drivers and the source-update
// window. Dependency-free on purpose: `tests/scripts/test_source_build_env.py`
// runs it under a bare `node` with no workspace install, so nothing here may
// import Playwright, zod, or the TypeScript smoke modules.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

/**
 * @param {string} root
 * @param {string} candidate
 * @returns {boolean}
 */
export function within(root, candidate) {
  const relative = path.relative(fs.realpathSync(root), fs.realpathSync(candidate));
  return relative === '' || (relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative));
}

/**
 * The launch environment for an isolated desktop smoke: inherited display/toolchain
 * variables minus credentials, driver activation, and every HERMES_* override.
 * @param {NodeJS.ProcessEnv} inherited
 * @param {string} home
 * @param {string} userData
 * @returns {Record<string, string> & {HOME: string, USERPROFILE: string, HERMES_HOME: string,
 *   HERMES_DESKTOP_USER_DATA_DIR: string, XDG_CONFIG_HOME: string, XDG_DATA_HOME: string, XDG_CACHE_HOME: string}}
 */
export function smokeEnvironment(inherited, home, userData) {
  /** @type {Record<string, string>} */
  const clean = {};
  const keepHermes = new Set(['HERMES_TEST_INSTALL_REF', 'HERMES_TEST_INSTALL_REPO']);
  for (const [key, value] of Object.entries(inherited)) {
    const name = key.toUpperCase();
    if (!value || /(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)/.test(name)
        || /^(?:PYTHON|VIRTUAL_ENV|CONDA|NODE_|NPM_|ELECTRON_|VITE_|UV_|PM_)/.test(name)
        || /^(?:OPENAI|ANTHROPIC|OPENROUTER|OLLAMA|GEMINI|GROQ|XAI)_/.test(name)
        || (name.startsWith('HERMES_') && !keepHermes.has(name))
        || ['INIT_CWD', 'TERMINAL_CWD', 'LD_PRELOAD', 'DYLD_INSERT_LIBRARIES', 'SSH_AUTH_SOCK', 'SSH_ASKPASS', 'GIT_ASKPASS'].includes(name)) {
      continue;
    }
    clean[key] = value;
  }
  const electronTemp = process.platform === 'win32' ? os.tmpdir() : '/tmp';
  return {
    ...clean, HOME: path.join(home, '.desktop-smoke-home'), USERPROFILE: path.join(home, '.desktop-smoke-home'),
    XDG_CONFIG_HOME: path.join(home, '.desktop-smoke-home', '.config'),
    XDG_DATA_HOME: path.join(home, '.desktop-smoke-home', '.local', 'share'),
    XDG_CACHE_HOME: path.join(home, '.desktop-smoke-home', '.cache'),
    HERMES_HOME: home, HERMES_DESKTOP_USER_DATA_DIR: userData,
    // Chromium puts ProcessSingleton's Unix socket below TMPDIR. Deep CI
    // workspaces exceed sockaddr_un.sun_path before Electron reaches ready.
    TEMP: electronTemp, TMP: electronTemp, TMPDIR: electronTemp,
    // app.close() cannot answer a native modal. This bypasses confirmation only;
    // normal backend teardown and the driver's process-exit checks still run.
    HERMES_DESKTOP_SKIP_QUIT_CONFIRM: '1',
    APPDATA: path.join(home, '.desktop-smoke-home', 'AppData', 'Roaming'),
    LOCALAPPDATA: path.join(home, '.desktop-smoke-home', 'AppData', 'Local'),
  };
}

const UPDATE_WINDOW_STATE = ['connection.json', 'connections.json'];

/**
 * Give the independently driven update window its own Electron instance route.
 * Copy only Hermes-owned connection contracts. Cloning Chromium's profile
 * carries browser locks and process state from the prior app into a supposedly
 * isolated launch. HERMES_HOME remains shared so the app updates the actual
 * installed runtime.
 *
 * @template {Record<string, string>} T
 * @param {T & {HERMES_DESKTOP_USER_DATA_DIR: string}} env
 * @returns {T & {HERMES_DESKTOP_USER_DATA_DIR: string}}
 */
export function isolateUpdateWindowEnvironment(env) {
  const source = env.HERMES_DESKTOP_USER_DATA_DIR;
  const isolated = fs.mkdtempSync(path.join(path.dirname(source), `${path.basename(source)}-app-update-`));
  for (const filename of UPDATE_WINDOW_STATE) {
    const from = path.join(source, filename);
    if (fs.existsSync(from)) fs.copyFileSync(from, path.join(isolated, filename));
  }
  return { ...env, HERMES_DESKTOP_USER_DATA_DIR: isolated };
}

/**
 * Route Electron's native ProcessSingleton before app JavaScript requests the lock.
 * Older packaged apps honor the environment override after Electron has initialized,
 * which can still leave Playwright attached to a lock-losing secondary instance.
 * @param {string[]} args
 * @param {string} userData
 * @returns {string[]}
 */
export function isolatedElectronArgs(args, userData) {
  const prefix = '--user-data-dir=';
  return [`${prefix}${userData}`, ...args.filter((arg) => !arg.startsWith(prefix))];
}

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
