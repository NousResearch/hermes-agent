"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.getToken = getToken;
exports.getServerUrl = getServerUrl;
exports.startWebUiServer = startWebUiServer;
exports.stopWebUiServer = stopWebUiServer;
const node_child_process_1 = require("node:child_process");
const node_fs_1 = require("node:fs");
const node_net_1 = require("node:net");
const node_os_1 = require("node:os");
const node_path_1 = require("node:path");
const node_crypto_1 = require("node:crypto");
const node_util_1 = require("node:util");
const electron_1 = require("electron");
const paths_1 = require("./paths");
const DEFAULT_PORT = 8748;
const DEFAULT_READY_TIMEOUT_MS = 120_000;
const DEFAULT_FULL_STARTUP_WAIT_MS = 0;
const DEFAULT_STOP_TIMEOUT_MS = 20_000;
const DEFAULT_GRACEFUL_STOP_TIMEOUT_MS = 18_000;
const AGENT_BRIDGE_STARTED_MARKER = '[bootstrap] agent bridge started';
const AGENT_BRIDGE_FAILED_MARKER = '[bootstrap] agent bridge failed to start';
const execFileAsync = (0, node_util_1.promisify)(node_child_process_1.execFile);
let serverProc = null;
let cachedToken = null;
let currentServerPort = DEFAULT_PORT;
function killProcessTree(proc) {
    if (!proc.pid || proc.killed)
        return;
    if (process.platform === 'win32') {
        try {
            const killer = (0, node_child_process_1.spawn)('taskkill.exe', ['/PID', String(proc.pid), '/T', '/F'], {
                stdio: 'ignore',
                windowsHide: true,
            });
            killer.once('error', () => undefined);
            return;
        }
        catch {
            /* fall through */
        }
    }
    try {
        proc.kill('SIGKILL');
    }
    catch {
        /* ignore */
    }
}
function envPositiveInt(name) {
    const raw = process.env[name];
    if (!raw)
        return undefined;
    const value = Number(raw);
    return Number.isFinite(value) && value > 0 ? value : undefined;
}
function readyTimeoutMs() {
    return envPositiveInt('HERMES_DESKTOP_READY_TIMEOUT_MS') || DEFAULT_READY_TIMEOUT_MS;
}
function fullStartupWaitMs() {
    const raw = process.env.HERMES_DESKTOP_FULL_STARTUP_WAIT_MS;
    if (raw === undefined)
        return DEFAULT_FULL_STARTUP_WAIT_MS;
    const value = Number(raw);
    return Number.isFinite(value) && value >= 0 ? value : DEFAULT_FULL_STARTUP_WAIT_MS;
}
function gracefulStopTimeoutMs() {
    return envPositiveInt('HERMES_DESKTOP_GRACEFUL_STOP_TIMEOUT_MS') || DEFAULT_GRACEFUL_STOP_TIMEOUT_MS;
}
function timeoutAfter(ms, message) {
    return new Promise((_, reject) => {
        const timer = setTimeout(() => reject(new Error(message)), ms);
        timer.unref?.();
    });
}
function createAgentBridgeStartupTracker() {
    let output = '';
    let state = 'pending';
    let resolveReady = null;
    let rejectReady = null;
    const settle = (nextState) => {
        if (state !== 'pending')
            return;
        state = nextState;
        if (nextState === 'started') {
            resolveReady?.();
        }
        else {
            rejectReady?.(new Error('Agent bridge failed to start'));
        }
    };
    const observe = (chunk) => {
        if (state !== 'pending')
            return;
        output = (output + chunk.toString('utf-8')).slice(-4096);
        if (output.includes(AGENT_BRIDGE_STARTED_MARKER)) {
            settle('started');
        }
        else if (output.includes(AGENT_BRIDGE_FAILED_MARKER)) {
            settle('failed');
        }
    };
    const wait = (timeoutMs) => {
        if (state === 'started')
            return Promise.resolve();
        if (state === 'failed')
            return Promise.reject(new Error('Agent bridge failed to start'));
        return new Promise((resolve, reject) => {
            const timer = setTimeout(() => {
                if (state !== 'pending')
                    return;
                state = 'failed';
                reject(new Error(`Agent bridge did not become ready within ${timeoutMs}ms`));
            }, timeoutMs);
            resolveReady = () => {
                clearTimeout(timer);
                resolve();
            };
            rejectReady = (err) => {
                clearTimeout(timer);
                reject(err);
            };
        });
    };
    return { observe, wait };
}
function ensureToken() {
    if (cachedToken)
        return cachedToken;
    const file = (0, paths_1.tokenFile)();
    (0, node_fs_1.mkdirSync)((0, node_path_1.dirname)(file), { recursive: true });
    if ((0, node_fs_1.existsSync)(file)) {
        cachedToken = (0, node_fs_1.readFileSync)(file, 'utf-8').trim();
        if (cachedToken)
            return cachedToken;
    }
    cachedToken = (0, node_crypto_1.randomBytes)(32).toString('hex');
    (0, node_fs_1.writeFileSync)(file, cachedToken + '\n', { mode: 0o600 });
    return cachedToken;
}
function ensureNativeModules() {
    try {
        const helper = (0, node_path_1.join)((0, paths_1.webuiDir)(), 'node_modules', 'node-pty', 'prebuilds', `${process.platform}-${process.arch}`, 'spawn-helper');
        if ((0, node_fs_1.existsSync)(helper))
            (0, node_fs_1.chmodSync)(helper, 0o755);
    }
    catch {
        /* ignore */
    }
}
const COMMON_USER_BIN_DIRS = process.platform === 'win32'
    ? []
    : [
        '/opt/homebrew/bin',
        '/usr/local/bin',
        '/usr/bin',
        '/bin',
        '/usr/sbin',
        '/sbin',
    ];
const PATH_MARKER_START = '__HERMES_DESKTOP_PATH_START__';
const PATH_MARKER_END = '__HERMES_DESKTOP_PATH_END__';
function mergePathEntries(...paths) {
    const seen = new Set();
    const entries = [];
    for (const rawPath of paths) {
        if (!rawPath)
            continue;
        for (const entry of rawPath.split(node_path_1.delimiter)) {
            const trimmed = entry.trim();
            if (!trimmed)
                continue;
            const key = process.platform === 'win32' ? trimmed.toLowerCase() : trimmed;
            if (seen.has(key))
                continue;
            seen.add(key);
            entries.push(trimmed);
        }
    }
    return entries.join(node_path_1.delimiter);
}
function extractMarkedPath(output) {
    const start = output.lastIndexOf(PATH_MARKER_START);
    const end = output.lastIndexOf(PATH_MARKER_END);
    if (start < 0 || end <= start)
        return null;
    const value = output.slice(start + PATH_MARKER_START.length, end).trim();
    return value || null;
}
function compareNodeVersionDesc(left, right) {
    const leftParts = left.replace(/^v/, '').split('.').map(part => Number.parseInt(part, 10) || 0);
    const rightParts = right.replace(/^v/, '').split('.').map(part => Number.parseInt(part, 10) || 0);
    for (let index = 0; index < Math.max(leftParts.length, rightParts.length); index += 1) {
        const diff = (rightParts[index] || 0) - (leftParts[index] || 0);
        if (diff !== 0)
            return diff;
    }
    return right.localeCompare(left);
}
function getNvmNodeBinPaths() {
    if (process.platform === 'win32')
        return '';
    const nvmDir = process.env.NVM_DIR?.trim() || (0, node_path_1.join)((0, node_os_1.homedir)(), '.nvm');
    const versionsDir = (0, node_path_1.join)(nvmDir, 'versions', 'node');
    if (!(0, node_fs_1.existsSync)(versionsDir))
        return '';
    try {
        return (0, node_fs_1.readdirSync)(versionsDir, { withFileTypes: true })
            .filter(entry => entry.isDirectory())
            .map(entry => entry.name)
            .sort(compareNodeVersionDesc)
            .map(version => (0, node_path_1.join)(versionsDir, version, 'bin'))
            .filter(binDir => (0, node_fs_1.existsSync)(binDir))
            .join(node_path_1.delimiter);
    }
    catch {
        return '';
    }
}
async function getLoginShellPath() {
    if (process.platform === 'win32')
        return null;
    const shell = process.env.SHELL?.trim() || (process.platform === 'darwin' ? '/bin/zsh' : '/bin/sh');
    if (!(0, node_fs_1.existsSync)(shell))
        return null;
    try {
        const { stdout } = await execFileAsync(shell, ['-l', '-c', `printf '\\n${PATH_MARKER_START}%s${PATH_MARKER_END}\\n' "$PATH"`], {
            encoding: 'utf-8',
            timeout: 1500,
            windowsHide: true,
            env: process.env,
        });
        return extractMarkedPath(stdout) || stdout.trim() || null;
    }
    catch {
        return null;
    }
}
function getToken() {
    return ensureToken();
}
function getServerUrl(port = DEFAULT_PORT) {
    return `http://127.0.0.1:${port}`;
}
async function getFreeTcpPort() {
    return await new Promise((resolveFreePort, rejectFreePort) => {
        const server = (0, node_net_1.createServer)();
        server.unref();
        server.once('error', rejectFreePort);
        server.listen(0, '127.0.0.1', () => {
            const address = server.address();
            server.close(() => {
                if (typeof address === 'object' && address?.port) {
                    resolveFreePort(address.port);
                }
                else {
                    rejectFreePort(new Error('Unable to allocate local TCP port'));
                }
            });
        });
    });
}
async function canBindTcpPort(port) {
    return await new Promise((resolveCanBind) => {
        const server = (0, node_net_1.createServer)();
        server.unref();
        server.once('error', () => resolveCanBind(false));
        server.listen(port, '127.0.0.1', () => {
            server.close(() => resolveCanBind(true));
        });
    });
}
async function getFreeTcpPortInRange(min, max) {
    for (let attempt = 0; attempt < 100; attempt += 1) {
        const port = min + ((0, node_crypto_1.randomBytes)(2).readUInt16BE(0) % (max - min + 1));
        if (await canBindTcpPort(port))
            return port;
    }
    return getFreeTcpPort();
}
async function startWebUiServer(port = DEFAULT_PORT) {
    ensureNativeModules();
    const token = ensureToken();
    currentServerPort = port;
    const entry = (0, paths_1.webuiServerEntry)();
    if (!(0, node_fs_1.existsSync)(entry)) {
        throw new Error(`Web UI server entry not found at ${entry}. Run: npm run build:webui`);
    }
    const home = (0, paths_1.webUiHome)();
    const agentHome = (0, paths_1.hermesHome)();
    (0, node_fs_1.mkdirSync)(home, { recursive: true });
    (0, node_fs_1.mkdirSync)(agentHome, { recursive: true });
    const isWin = process.platform === 'win32';
    const bundledPython = isWin
        ? (0, node_path_1.join)((0, paths_1.pythonDir)(), 'python.exe')
        : (0, node_path_1.join)((0, paths_1.pythonDir)(), 'bin', 'python3');
    const bundledAgentBrowserBin = isWin
        ? (0, node_path_1.join)((0, paths_1.pythonDir)(), 'node')
        : (0, node_path_1.join)((0, paths_1.pythonDir)(), 'node', 'bin');
    const bundledNodeBin = (0, paths_1.nodeBinDir)();
    const bundledGitPath = (0, paths_1.gitPathDirs)().join(node_path_1.delimiter);
    const bridgePort = await getFreeTcpPort();
    const workerPortBase = await getFreeTcpPortInRange(20000, 59000);
    const loginShellPath = await getLoginShellPath();
    const nvmNodeBinPaths = getNvmNodeBinPaths();
    const runtimePath = mergePathEntries((0, node_path_1.dirname)((0, paths_1.hermesBin)()), bundledAgentBrowserBin, bundledNodeBin, bundledGitPath, loginShellPath, nvmNodeBinPaths, process.env.PATH, process.env.Path, COMMON_USER_BIN_DIRS.join(node_path_1.delimiter));
    const browserExecutableOverride = process.env.AGENT_BROWSER_EXECUTABLE_PATH?.trim();
    const gitBin = (0, paths_1.bundledGit)();
    const env = {
        ...process.env,
        ELECTRON_RUN_AS_NODE: '1',
        NODE_ENV: 'production',
        HERMES_DESKTOP: 'true',
        HERMES_BIN: (0, paths_1.hermesBin)(),
        HERMES_AGENT_BRIDGE_PYTHON: bundledPython,
        HERMES_AGENT_CLI_PYTHON: bundledPython,
        HERMES_AGENT_ROOT: (0, paths_1.pythonDir)(),
        HERMES_AGENT_NODE: (0, paths_1.bundledNode)(),
        HERMES_AGENT_NODE_ROOT: isWin ? bundledNodeBin : (0, node_path_1.dirname)(bundledNodeBin),
        AGENT_BROWSER_HOME: process.env.AGENT_BROWSER_HOME?.trim() || (0, paths_1.bundledAgentBrowserHome)(),
        ...(browserExecutableOverride ? { AGENT_BROWSER_EXECUTABLE_PATH: browserExecutableOverride } : {}),
        PLAYWRIGHT_BROWSERS_PATH: process.env.PLAYWRIGHT_BROWSERS_PATH || (0, node_path_1.join)((0, paths_1.pythonDir)(), 'ms-playwright'),
        ...(gitBin ? { HERMES_AGENT_GIT: gitBin } : {}),
        HERMES_AGENT_BRIDGE_ENDPOINT: `tcp://127.0.0.1:${bridgePort}`,
        HERMES_AGENT_BRIDGE_CONNECT_RETRY_MS: process.env.HERMES_AGENT_BRIDGE_CONNECT_RETRY_MS ?? '120000',
        HERMES_AGENT_BRIDGE_WORKER_TRANSPORT: 'tcp',
        HERMES_AGENT_BRIDGE_WORKER_PORT_BASE: String(workerPortBase),
        HERMES_WEB_UI_PREVIEW_AGENT_BRIDGE_TRANSPORT: 'tcp',
        HERMES_WEB_UI_DISABLE_UPDATE_CHECK: 'true',
        GATEWAY_ALLOW_ALL_USERS: process.env.GATEWAY_ALLOW_ALL_USERS ?? 'true',
        HERMES_HOME: agentHome,
        HERMES_WEB_UI_HOME: home,
        HERMES_WEBUI_STATE_DIR: home,
        AUTH_TOKEN: token,
        PORT: String(port),
        PATH: runtimePath,
    };
    serverProc = (0, node_child_process_1.spawn)(process.execPath, [entry], {
        cwd: (0, paths_1.webuiDir)(),
        env,
        stdio: ['ignore', 'pipe', 'pipe'],
        windowsHide: true,
    });
    const bridgeStartup = createAgentBridgeStartupTracker();
    serverProc.stdout?.on('data', (chunk) => {
        bridgeStartup.observe(chunk);
        try {
            process.stdout.write(`[webui] ${chunk}`);
        } catch {
            /* EPIPE: parent process closed stdout, ignore */
        }
    });
    serverProc.stdout?.on('error', () => {
        /* EPIPE: child stdout stream error, ignore */
    });
    serverProc.stderr?.on('data', (chunk) => {
        bridgeStartup.observe(chunk);
        try {
            process.stderr.write(`[webui] ${chunk}`);
        } catch {
            /* EPIPE: parent process closed stderr, ignore */
        }
    });
    serverProc.stderr?.on('error', () => {
        /* EPIPE: child stderr stream error, ignore */
    });
    serverProc.on('exit', (code, signal) => {
        console.error(`[webui] server exited code=${code} signal=${signal}`);
        serverProc = null;
        if (!electron_1.app.isReady() || code !== 0) {
            // Best-effort: if server dies abnormally during startup, surface to user
        }
    });
    const timeoutMs = readyTimeoutMs();
    const bridgeReady = bridgeStartup.wait(timeoutMs);
    await waitForReady(port, timeoutMs);
    const fullStartupTimeoutMs = fullStartupWaitMs();
    if (fullStartupTimeoutMs > 0) {
        await Promise.race([
            bridgeReady,
            timeoutAfter(fullStartupTimeoutMs, `Agent bridge did not become ready within ${fullStartupTimeoutMs}ms`),
        ]).catch(err => {
            console.warn(`[webui] agent bridge was not ready during startup: ${err instanceof Error ? err.message : String(err)}`);
        });
        void bridgeReady.catch(() => undefined);
    }
    else {
        void bridgeReady.catch(err => {
            console.warn(`[webui] agent bridge was not ready during startup: ${err instanceof Error ? err.message : String(err)}`);
        });
    }
    return getServerUrl(port);
}
async function waitForReady(port, timeoutMs) {
    const deadline = Date.now() + timeoutMs;
    const url = `http://127.0.0.1:${port}/`;
    while (Date.now() < deadline) {
        try {
            const res = await fetch(url, { signal: AbortSignal.timeout(1000) });
            if (res.ok)
                return;
        }
        catch {
            /* not ready yet */
        }
        await new Promise(r => setTimeout(r, 300));
    }
    throw new Error(`Web UI shell did not become ready within ${timeoutMs}ms`);
}
async function requestGracefulShutdown(port, token) {
    const timeoutMs = gracefulStopTimeoutMs();
    const response = await fetch(`http://127.0.0.1:${port}/api/desktop/shutdown`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        signal: AbortSignal.timeout(timeoutMs),
    });
    if (!response.ok && response.status !== 202) {
        throw new Error(`desktop shutdown returned HTTP ${response.status}`);
    }
}
async function stopWebUiServer() {
    if (!serverProc || serverProc.killed)
        return;
    const proc = serverProc;
    const exited = new Promise(resolve => {
        proc.once('exit', () => resolve());
    });
    const forceAfter = new Promise(resolve => {
        const timer = setTimeout(() => {
            killProcessTree(proc);
            resolve();
        }, envPositiveInt('HERMES_DESKTOP_STOP_TIMEOUT_MS') || DEFAULT_STOP_TIMEOUT_MS);
        proc.once('exit', () => {
            clearTimeout(timer);
            resolve();
        });
    });
    try {
        await requestGracefulShutdown(currentServerPort, ensureToken());
    }
    catch (err) {
        console.warn(`[webui] graceful shutdown request failed: ${err instanceof Error ? err.message : String(err)}`);
        killProcessTree(proc);
    }
    await Promise.race([exited, forceAfter]);
}
