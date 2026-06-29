"use strict";
var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __esm = (fn, res, err) => function __init() {
  if (err) throw err[0];
  try {
    return fn && (res = (0, fn[__getOwnPropNames(fn)[0]])(fn = 0)), res;
  } catch (e) {
    throw err = [e], e;
  }
};
var __commonJS = (cb, mod) => function __require() {
  try {
    return mod || (0, cb[__getOwnPropNames(cb)[0]])((mod = { exports: {} }).exports, mod), mod.exports;
  } catch (e) {
    throw mod = 0, e;
  }
};
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
  // If the importer is in node compatibility mode or this is not an ESM
  // file that has been converted to a CommonJS file using a Babel-
  // compatible transform (i.e. "__esModule" has not been set), then set
  // "default" to the CommonJS "module.exports" for node compatibility.
  isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
  mod
));
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// electron/bootstrap-platform.cjs
var require_bootstrap_platform = __commonJS({
  "electron/bootstrap-platform.cjs"(exports2, module2) {
    "use strict";
    var fs2 = require("node:fs");
    function isWslEnvironment2(env2 = process.env, platform = process.platform, kernelRelease = null) {
      if (platform !== "linux") return false;
      if (env2.WSL_DISTRO_NAME || env2.WSL_INTEROP) return true;
      try {
        const release = kernelRelease ?? fs2.readFileSync("/proc/sys/kernel/osrelease", "utf8");
        return /microsoft|wsl/i.test(release);
      } catch {
        return false;
      }
    }
    function isWindowsBinaryPathInWsl2(filePath, options = {}) {
      const isWsl = options.isWsl ?? isWslEnvironment2(options.env, options.platform);
      if (!isWsl) return false;
      const normalized = String(filePath || "").replace(/\\/g, "/").toLowerCase();
      return normalized.endsWith(".exe") || normalized.endsWith(".cmd") || normalized.endsWith(".bat") || normalized.endsWith(".ps1");
    }
    function bundledRuntimeImportCheck(platform = process.platform) {
      return platform === "win32" ? "import fastapi, uvicorn, winpty" : "import fastapi, uvicorn, ptyprocess";
    }
    var GPU_OVERRIDE_ON = /* @__PURE__ */ new Set(["1", "true", "yes", "on"]);
    var GPU_OVERRIDE_OFF = /* @__PURE__ */ new Set(["0", "false", "no", "off"]);
    function detectRemoteDisplay2(options = {}) {
      const env2 = options.env ?? process.env;
      const platform = options.platform ?? process.platform;
      const override = String(env2.HERMES_DESKTOP_DISABLE_GPU || "").trim().toLowerCase();
      if (GPU_OVERRIDE_ON.has(override)) return "override (HERMES_DESKTOP_DISABLE_GPU)";
      if (GPU_OVERRIDE_OFF.has(override)) return null;
      if (env2.SSH_CONNECTION || env2.SSH_CLIENT || env2.SSH_TTY) return "ssh-session";
      if (platform === "linux") {
        const display = String(env2.DISPLAY || "");
        if (display.includes(":") && display.split(":")[0]) {
          return `x11-forwarding (DISPLAY=${display})`;
        }
      }
      if (platform === "win32") {
        const sessionName = String(env2.SESSIONNAME || "");
        if (/^rdp-/i.test(sessionName)) return `rdp (SESSIONNAME=${sessionName})`;
      }
      return null;
    }
    module2.exports = {
      bundledRuntimeImportCheck,
      detectRemoteDisplay: detectRemoteDisplay2,
      isWindowsBinaryPathInWsl: isWindowsBinaryPathInWsl2,
      isWslEnvironment: isWslEnvironment2
    };
  }
});

// electron/bootstrap-runner.cjs
var require_bootstrap_runner = __commonJS({
  "electron/bootstrap-runner.cjs"(exports2, module2) {
    "use strict";
    var fs2 = require("node:fs");
    var fsp = require("node:fs/promises");
    var path2 = require("node:path");
    var https2 = require("node:https");
    var { spawn: spawn2 } = require("node:child_process");
    var IS_WINDOWS2 = process.platform === "win32";
    function hiddenWindowsChildOptions2(options = {}) {
      if (!IS_WINDOWS2 || Object.prototype.hasOwnProperty.call(options, "windowsHide")) {
        return options;
      }
      return { ...options, windowsHide: true };
    }
    var STAMP_COMMIT_RE = /^[0-9a-f]{7,40}$/i;
    function installScriptName() {
      return process.platform === "win32" ? "install.ps1" : "install.sh";
    }
    function installScriptKind() {
      return process.platform === "win32" ? "powershell" : "posix";
    }
    function resolveLocalInstallScript(sourceRepoRoot) {
      if (!sourceRepoRoot) return null;
      const candidate = path2.join(sourceRepoRoot, "scripts", installScriptName());
      try {
        fs2.accessSync(candidate, fs2.constants.R_OK);
        return candidate;
      } catch {
        return null;
      }
    }
    function bootstrapCacheDir(hermesHome) {
      return path2.join(hermesHome, "bootstrap-cache");
    }
    function installedAgentInstallScript(hermesHome) {
      if (!hermesHome) return null;
      const candidate = path2.join(hermesHome, "hermes-agent", "scripts", installScriptName());
      try {
        fs2.accessSync(candidate, fs2.constants.R_OK);
        return candidate;
      } catch {
        return null;
      }
    }
    function cachedScriptPath(hermesHome, commit) {
      return path2.join(bootstrapCacheDir(hermesHome), `install-${commit}.${process.platform === "win32" ? "ps1" : "sh"}`);
    }
    function downloadInstallScript(commit, destPath) {
      const scriptName = installScriptName();
      const url = `https://raw.githubusercontent.com/NousResearch/hermes-agent/${commit}/scripts/${scriptName}`;
      return new Promise((resolve, reject) => {
        fs2.mkdirSync(path2.dirname(destPath), { recursive: true });
        const tmpPath = destPath + ".tmp";
        const out = fs2.createWriteStream(tmpPath);
        https2.get(url, (res) => {
          if (res.statusCode === 301 || res.statusCode === 302) {
            out.close();
            fs2.unlinkSync(tmpPath);
            https2.get(res.headers.location, (res2) => {
              if (res2.statusCode !== 200) {
                reject(
                  new Error(
                    `Failed to download ${scriptName}: HTTP ${res2.statusCode} from redirect ${res.headers.location}`
                  )
                );
                return;
              }
              const out2 = fs2.createWriteStream(tmpPath);
              res2.pipe(out2);
              out2.on("finish", () => {
                out2.close();
                fs2.renameSync(tmpPath, destPath);
                resolve(destPath);
              });
              out2.on("error", reject);
            }).on("error", reject);
            return;
          }
          if (res.statusCode !== 200) {
            out.close();
            try {
              fs2.unlinkSync(tmpPath);
            } catch {
            }
            reject(new Error(`Failed to download ${scriptName}: HTTP ${res.statusCode} from ${url}`));
            return;
          }
          res.pipe(out);
          out.on("finish", () => {
            out.close();
            fs2.renameSync(tmpPath, destPath);
            resolve(destPath);
          });
          out.on("error", (err) => {
            try {
              fs2.unlinkSync(tmpPath);
            } catch {
            }
            reject(err);
          });
        }).on("error", (err) => {
          try {
            fs2.unlinkSync(tmpPath);
          } catch {
          }
          reject(err);
        });
      });
    }
    async function resolveInstallScript({ installStamp, sourceRepoRoot, hermesHome, emit, _download = downloadInstallScript }) {
      const localScript = resolveLocalInstallScript(sourceRepoRoot);
      if (localScript) {
        emit({ type: "log", line: `[bootstrap] using local ${installScriptName()} at ${localScript}` });
        return { path: localScript, source: "local", kind: installScriptKind() };
      }
      if (!installStamp || !installStamp.commit || !STAMP_COMMIT_RE.test(installStamp.commit)) {
        throw new Error(
          `Cannot resolve ${installScriptName()}: no SOURCE_REPO_ROOT and no install stamp. This packaged build was produced without a valid build-time stamp.`
        );
      }
      const cached = cachedScriptPath(hermesHome, installStamp.commit);
      try {
        await fsp.access(cached, fs2.constants.R_OK);
        emit({
          type: "log",
          line: `[bootstrap] using cached ${installScriptName()} for ${installStamp.commit.slice(0, 12)}`
        });
        return { path: cached, source: "cache", commit: installStamp.commit, kind: installScriptKind() };
      } catch {
      }
      emit({
        type: "log",
        line: `[bootstrap] fetching ${installScriptName()} for ${installStamp.commit.slice(0, 12)} from GitHub`
      });
      try {
        await _download(installStamp.commit, cached);
        emit({ type: "log", line: `[bootstrap] saved to ${cached}` });
        return { path: cached, source: "download", commit: installStamp.commit, kind: installScriptKind() };
      } catch (err) {
        const installed = installedAgentInstallScript(hermesHome);
        if (installed) {
          emit({
            type: "log",
            line: `[bootstrap] GitHub fetch failed (${err.message}); falling back to installed agent ${installScriptName()} at ${installed}`
          });
          try {
            fs2.mkdirSync(path2.dirname(cached), { recursive: true });
            fs2.copyFileSync(installed, cached);
            return { path: cached, source: "installed-agent", commit: installStamp.commit, kind: installScriptKind() };
          } catch {
            return { path: installed, source: "installed-agent", commit: installStamp.commit, kind: installScriptKind() };
          }
        }
        throw err;
      }
    }
    function powershellUnderRoot(root) {
      return path2.join(root, "System32", "WindowsPowerShell", "v1.0", "powershell.exe");
    }
    function resolveWindowsPowerShell() {
      for (const v of ["SystemRoot", "windir"]) {
        const root = process.env[v];
        if (root) {
          const candidate = powershellUnderRoot(root);
          try {
            if (fs2.statSync(candidate).isFile()) return candidate;
          } catch {
          }
        }
      }
      const pathDirs = (process.env.PATH || process.env.Path || "").split(path2.delimiter).filter(Boolean);
      for (const exe of ["powershell.exe", "pwsh.exe"]) {
        for (const dir of pathDirs) {
          const candidate = path2.join(dir, exe);
          try {
            if (fs2.statSync(candidate).isFile()) return candidate;
          } catch {
          }
        }
      }
      return "powershell.exe";
    }
    function spawnPowerShell(scriptPath, args, { emit, stageName, abortSignal, hermesHome } = {}) {
      return new Promise((resolve, reject) => {
        const ps = process.platform === "win32" ? resolveWindowsPowerShell() : "pwsh";
        const fullArgs = ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", scriptPath, ...args];
        const child = spawn2(ps, fullArgs, hiddenWindowsChildOptions2({
          stdio: ["ignore", "pipe", "pipe"],
          env: {
            ...process.env,
            // Pass HERMES_HOME through so install.ps1 respects the caller's
            // choice rather than re-computing the default.
            HERMES_HOME: hermesHome || process.env.HERMES_HOME || ""
          }
        }));
        let stdout = "";
        let stderr = "";
        let killed = false;
        const onAbort = () => {
          killed = true;
          try {
            child.kill("SIGTERM");
          } catch {
          }
        };
        if (abortSignal) {
          if (abortSignal.aborted) {
            onAbort();
          } else {
            abortSignal.addEventListener("abort", onAbort, { once: true });
          }
        }
        child.stdout.setEncoding("utf8");
        child.stderr.setEncoding("utf8");
        let stdoutBuf = "";
        child.stdout.on("data", (chunk) => {
          stdout += chunk;
          stdoutBuf += chunk;
          let nl;
          while ((nl = stdoutBuf.indexOf("\n")) !== -1) {
            const line = stdoutBuf.slice(0, nl).replace(/\r$/, "");
            stdoutBuf = stdoutBuf.slice(nl + 1);
            if (line) emit && emit({ type: "log", stage: stageName, line, stream: "stdout" });
          }
        });
        let stderrBuf = "";
        child.stderr.on("data", (chunk) => {
          stderr += chunk;
          stderrBuf += chunk;
          let nl;
          while ((nl = stderrBuf.indexOf("\n")) !== -1) {
            const line = stderrBuf.slice(0, nl).replace(/\r$/, "");
            stderrBuf = stderrBuf.slice(nl + 1);
            if (line) emit && emit({ type: "log", stage: stageName, line, stream: "stderr" });
          }
        });
        child.on("error", (err) => {
          if (abortSignal) abortSignal.removeEventListener("abort", onAbort);
          reject(err);
        });
        child.on("close", (code, signal) => {
          if (abortSignal) abortSignal.removeEventListener("abort", onAbort);
          if (stdoutBuf) emit && emit({ type: "log", stage: stageName, line: stdoutBuf, stream: "stdout" });
          if (stderrBuf) emit && emit({ type: "log", stage: stageName, line: stderrBuf, stream: "stderr" });
          resolve({ stdout, stderr, code, signal, killed });
        });
      });
    }
    function spawnBash(scriptPath, args, { emit, stageName, abortSignal, hermesHome } = {}) {
      return new Promise((resolve, reject) => {
        const child = spawn2("bash", [scriptPath, ...args], {
          stdio: ["ignore", "pipe", "pipe"],
          env: {
            ...process.env,
            HERMES_HOME: hermesHome || process.env.HERMES_HOME || ""
          }
        });
        let stdout = "";
        let stderr = "";
        let killed = false;
        const onAbort = () => {
          killed = true;
          try {
            child.kill("SIGTERM");
          } catch {
          }
        };
        if (abortSignal) {
          if (abortSignal.aborted) {
            onAbort();
          } else {
            abortSignal.addEventListener("abort", onAbort, { once: true });
          }
        }
        child.stdout.setEncoding("utf8");
        child.stderr.setEncoding("utf8");
        let stdoutBuf = "";
        child.stdout.on("data", (chunk) => {
          stdout += chunk;
          stdoutBuf += chunk;
          let nl;
          while ((nl = stdoutBuf.indexOf("\n")) !== -1) {
            const line = stdoutBuf.slice(0, nl).replace(/\r$/, "");
            stdoutBuf = stdoutBuf.slice(nl + 1);
            if (line) emit && emit({ type: "log", stage: stageName, line, stream: "stdout" });
          }
        });
        let stderrBuf = "";
        child.stderr.on("data", (chunk) => {
          stderr += chunk;
          stderrBuf += chunk;
          let nl;
          while ((nl = stderrBuf.indexOf("\n")) !== -1) {
            const line = stderrBuf.slice(0, nl).replace(/\r$/, "");
            stderrBuf = stderrBuf.slice(nl + 1);
            if (line) emit && emit({ type: "log", stage: stageName, line, stream: "stderr" });
          }
        });
        child.on("error", (err) => {
          if (abortSignal) abortSignal.removeEventListener("abort", onAbort);
          reject(err);
        });
        child.on("close", (code, signal) => {
          if (abortSignal) abortSignal.removeEventListener("abort", onAbort);
          if (stdoutBuf) emit && emit({ type: "log", stage: stageName, line: stdoutBuf, stream: "stdout" });
          if (stderrBuf) emit && emit({ type: "log", stage: stageName, line: stderrBuf, stream: "stderr" });
          resolve({ stdout, stderr, code, signal, killed });
        });
      });
    }
    function buildPinArgs(installStamp) {
      const args = [];
      if (installStamp && installStamp.commit) {
        args.push("-Commit", installStamp.commit);
      }
      if (installStamp && installStamp.branch) {
        args.push("-Branch", installStamp.branch);
      }
      return args;
    }
    function buildPosixPinArgs({ installStamp, activeRoot, hermesHome }) {
      const args = ["--dir", activeRoot, "--hermes-home", hermesHome];
      if (installStamp && installStamp.branch) {
        args.push("--branch", installStamp.branch);
      }
      if (installStamp && installStamp.commit) {
        args.push("--commit", installStamp.commit);
      }
      return args;
    }
    async function fetchManifest({ scriptPath, installerKind, emit, hermesHome, activeRoot, installStamp }) {
      const isPosix = installerKind === "posix";
      const args = isPosix ? ["--manifest", ...buildPosixPinArgs({ installStamp, activeRoot, hermesHome })] : ["-Manifest", ...buildPinArgs(installStamp)];
      const result = await (isPosix ? spawnBash : spawnPowerShell)(scriptPath, args, {
        emit,
        stageName: "__manifest__",
        hermesHome
      });
      if (result.code !== 0) {
        throw new Error(
          `${isPosix ? "install.sh --manifest" : "install.ps1 -Manifest"} failed: exit ${result.code}
${result.stderr || result.stdout}`
        );
      }
      const lines = result.stdout.split(/\r?\n/).filter(Boolean);
      for (let i = lines.length - 1; i >= 0; i--) {
        try {
          const parsed = JSON.parse(lines[i]);
          if (parsed && Array.isArray(parsed.stages)) {
            return parsed;
          }
        } catch {
        }
      }
      throw new Error(
        `${isPosix ? "install.sh --manifest" : "install.ps1 -Manifest"} produced no parseable JSON payload
${result.stdout}`
      );
    }
    function parseStageResult(stdout) {
      const lines = stdout.split(/\r?\n/).filter(Boolean);
      for (let i = lines.length - 1; i >= 0; i--) {
        try {
          const parsed = JSON.parse(lines[i]);
          if (parsed && typeof parsed.ok === "boolean" && typeof parsed.stage === "string") {
            return parsed;
          }
        } catch {
        }
      }
      return null;
    }
    async function runStage({ scriptPath, installerKind, stage, emit, hermesHome, activeRoot, abortSignal, installStamp }) {
      const startedAt = Date.now();
      emit({ type: "stage", name: stage.name, state: "running" });
      const isPosix = installerKind === "posix";
      const args = isPosix ? [
        "--stage",
        stage.name,
        "--non-interactive",
        "--json",
        ...buildPosixPinArgs({ installStamp, activeRoot, hermesHome })
      ] : ["-Stage", stage.name, "-NonInteractive", "-Json", ...buildPinArgs(installStamp)];
      const result = await (isPosix ? spawnBash : spawnPowerShell)(scriptPath, args, {
        emit,
        stageName: stage.name,
        abortSignal,
        hermesHome
      });
      const durationMs = Date.now() - startedAt;
      if (result.killed) {
        const ev2 = { type: "stage", name: stage.name, state: "failed", durationMs, error: "cancelled by user" };
        emit(ev2);
        return ev2;
      }
      const json = parseStageResult(result.stdout);
      if (!json) {
        const ev2 = {
          type: "stage",
          name: stage.name,
          state: "failed",
          durationMs,
          error: `${isPosix ? "install.sh --stage" : "install.ps1 -Stage"} ${stage.name} produced no JSON result frame (exit=${result.code})`,
          json: null
        };
        emit(ev2);
        return ev2;
      }
      if (json.ok && json.skipped) {
        const ev2 = { type: "stage", name: stage.name, state: "skipped", durationMs, json };
        emit(ev2);
        return ev2;
      }
      if (json.ok) {
        const ev2 = { type: "stage", name: stage.name, state: "succeeded", durationMs, json };
        emit(ev2);
        return ev2;
      }
      const ev = {
        type: "stage",
        name: stage.name,
        state: "failed",
        durationMs,
        json,
        error: json.reason || `exit code ${result.code}`
      };
      emit(ev);
      return ev;
    }
    function openRunLog(logRoot) {
      fs2.mkdirSync(logRoot, { recursive: true });
      const ts = (/* @__PURE__ */ new Date()).toISOString().replace(/[:.]/g, "-");
      const logPath = path2.join(logRoot, `bootstrap-${ts}.log`);
      const stream = fs2.createWriteStream(logPath, { flags: "a" });
      return { path: logPath, stream };
    }
    async function runBootstrap2(opts) {
      const {
        installStamp,
        activeRoot,
        sourceRepoRoot,
        hermesHome,
        logRoot,
        onEvent,
        abortSignal,
        writeMarker
        // callback to write the bootstrap-complete marker; main.cjs provides
      } = opts;
      if (abortSignal && abortSignal.aborted) {
        if (typeof onEvent === "function") {
          try {
            onEvent({ type: "failed", error: "bootstrap cancelled by user" });
          } catch {
          }
        }
        return { ok: false, cancelled: true };
      }
      const runLog = openRunLog(logRoot || path2.join(hermesHome, "logs"));
      const emit = (ev) => {
        try {
          runLog.stream.write(JSON.stringify(ev) + "\n");
        } catch {
        }
        try {
          if (typeof onEvent === "function") onEvent(ev);
        } catch (err) {
          runLog.stream.write(`emit error: ${err && err.message}
`);
        }
      };
      emit({
        type: "log",
        line: `[bootstrap] starting at ${(/* @__PURE__ */ new Date()).toISOString()}; activeRoot=${activeRoot}; stamp=${installStamp ? installStamp.commit.slice(0, 12) : "<none>"}; runLog=${runLog.path}`
      });
      try {
        const scriptInfo = await resolveInstallScript({ installStamp, sourceRepoRoot, hermesHome, emit });
        const installerKind = scriptInfo.kind || "powershell";
        const manifest = await fetchManifest({
          scriptPath: scriptInfo.path,
          installerKind,
          emit,
          hermesHome,
          activeRoot,
          installStamp
        });
        emit({
          type: "manifest",
          stages: manifest.stages,
          protocolVersion: manifest.protocol_version || manifest.protocolVersion || null
        });
        for (const stage of manifest.stages) {
          if (abortSignal && abortSignal.aborted) {
            emit({ type: "failed", error: "bootstrap cancelled by user" });
            return { ok: false, cancelled: true };
          }
          const ev = await runStage({
            scriptPath: scriptInfo.path,
            installerKind,
            stage,
            emit,
            hermesHome,
            activeRoot,
            abortSignal,
            installStamp
          });
          if (ev.state === "failed") {
            emit({ type: "failed", stage: stage.name, error: ev.error || "stage failed" });
            return { ok: false, failedStage: stage.name, error: ev.error };
          }
        }
        const markerPayload = {
          pinnedCommit: installStamp ? installStamp.commit : null,
          pinnedBranch: installStamp ? installStamp.branch : null
        };
        const marker = typeof writeMarker === "function" ? writeMarker(markerPayload) : markerPayload;
        emit({ type: "complete", marker });
        return { ok: true, marker };
      } catch (err) {
        emit({ type: "failed", error: err.message || String(err) });
        return { ok: false, error: err.message || String(err) };
      } finally {
        try {
          runLog.stream.end();
        } catch {
        }
      }
    }
    module2.exports = {
      runBootstrap: runBootstrap2,
      // Exposed for testability
      parseStageResult,
      resolveLocalInstallScript,
      resolveInstallScript,
      installedAgentInstallScript,
      cachedScriptPath
    };
  }
});

// electron/session-windows.cjs
var require_session_windows = __commonJS({
  "electron/session-windows.cjs"(exports2, module2) {
    "use strict";
    var { pathToFileURL: pathToFileURL2 } = require("node:url");
    var SESSION_WINDOW_MIN_WIDTH2 = 420;
    var SESSION_WINDOW_MIN_HEIGHT2 = 620;
    function chatWindowWebPreferences2(preloadPath) {
      return {
        preload: preloadPath,
        contextIsolation: true,
        webviewTag: true,
        sandbox: true,
        nodeIntegration: false,
        devTools: true,
        backgroundThrottling: false
      };
    }
    function buildSessionWindowUrl2(sessionId, { devServer, rendererIndexPath, watch, newSession } = {}) {
      const query = `?win=secondary${newSession ? "&new=1" : ""}${watch ? "&watch=1" : ""}`;
      const route = newSession ? "#/" : `#/${encodeURIComponent(sessionId)}`;
      if (devServer) {
        const base = devServer.endsWith("/") ? devServer.slice(0, -1) : devServer;
        return `${base}/${query}${route}`;
      }
      return `${pathToFileURL2(rendererIndexPath).toString()}${query}${route}`;
    }
    function createSessionWindowRegistry2() {
      const windows = /* @__PURE__ */ new Map();
      function openOrFocus(sessionId, factory) {
        const key = typeof sessionId === "string" ? sessionId.trim() : "";
        if (!key) {
          return null;
        }
        const existing = windows.get(key);
        if (existing && !existing.isDestroyed()) {
          if (typeof existing.isMinimized === "function" && existing.isMinimized()) {
            existing.restore?.();
          }
          if (typeof existing.isVisible === "function" && !existing.isVisible()) {
            existing.show?.();
          }
          existing.focus?.();
          return existing;
        }
        const win = factory(key);
        if (!win) {
          return null;
        }
        windows.set(key, win);
        win.on?.("closed", () => {
          if (windows.get(key) === win) {
            windows.delete(key);
          }
        });
        return win;
      }
      return {
        openOrFocus,
        get: (key) => windows.get(key),
        has: (key) => windows.has(key),
        get size() {
          return windows.size;
        }
      };
    }
    module2.exports = {
      buildSessionWindowUrl: buildSessionWindowUrl2,
      chatWindowWebPreferences: chatWindowWebPreferences2,
      createSessionWindowRegistry: createSessionWindowRegistry2,
      SESSION_WINDOW_MIN_HEIGHT: SESSION_WINDOW_MIN_HEIGHT2,
      SESSION_WINDOW_MIN_WIDTH: SESSION_WINDOW_MIN_WIDTH2
    };
  }
});

// electron/backend-probes.cjs
var require_backend_probes = __commonJS({
  "electron/backend-probes.cjs"(exports2, module2) {
    "use strict";
    var { execFileSync: execFileSync2 } = require("node:child_process");
    var PROBE_TIMEOUT_MS = 5e3;
    function canImportHermesCli2(pythonPath) {
      if (!pythonPath) return false;
      try {
        execFileSync2(pythonPath, ["-c", "import hermes_cli"], {
          stdio: "ignore",
          timeout: PROBE_TIMEOUT_MS,
          windowsHide: true
        });
        return true;
      } catch {
        return false;
      }
    }
    function verifyHermesCli2(hermesCommand, opts = {}) {
      if (!hermesCommand) return false;
      try {
        execFileSync2(hermesCommand, ["--version"], {
          stdio: "ignore",
          timeout: PROBE_TIMEOUT_MS,
          shell: Boolean(opts.shell),
          windowsHide: true
        });
        return true;
      } catch {
        return false;
      }
    }
    module2.exports = {
      canImportHermesCli: canImportHermesCli2,
      verifyHermesCli: verifyHermesCli2,
      PROBE_TIMEOUT_MS
    };
  }
});

// electron/link-title-window.cjs
var require_link_title_window = __commonJS({
  "electron/link-title-window.cjs"(exports2, module2) {
    "use strict";
    function linkTitleWindowOptions(partitionSession) {
      return {
        show: false,
        width: 1280,
        height: 800,
        webPreferences: {
          backgroundThrottling: false,
          contextIsolation: true,
          javascript: true,
          nodeIntegration: false,
          sandbox: true,
          session: partitionSession,
          webSecurity: true
        }
      };
    }
    function createLinkTitleWindow2(BrowserWindow2, partitionSession) {
      const window2 = new BrowserWindow2(linkTitleWindowOptions(partitionSession));
      try {
        window2.webContents.setAudioMuted(true);
      } catch {
      }
      return window2;
    }
    module2.exports = { createLinkTitleWindow: createLinkTitleWindow2, linkTitleWindowOptions };
  }
});

// electron/gateway-ws-probe.cjs
var require_gateway_ws_probe = __commonJS({
  "electron/gateway-ws-probe.cjs"(exports2, module2) {
    "use strict";
    var DEFAULT_CONNECT_TIMEOUT_MS = 1e4;
    var DEFAULT_READY_GRACE_MS = 750;
    function probeGatewayWebSocket2(wsUrl, options = {}) {
      const WebSocketImpl = options.WebSocketImpl;
      const connectTimeoutMs = options.connectTimeoutMs ?? DEFAULT_CONNECT_TIMEOUT_MS;
      const readyGraceMs = options.readyGraceMs ?? DEFAULT_READY_GRACE_MS;
      if (typeof WebSocketImpl !== "function") {
        return Promise.resolve({
          ok: false,
          reason: "WebSocket is not available in this runtime."
        });
      }
      return new Promise((resolve) => {
        let settled = false;
        let opened = false;
        let connectTimer = null;
        let graceTimer = null;
        let socket;
        const clearTimers = () => {
          if (connectTimer !== null) {
            clearTimeout(connectTimer);
            connectTimer = null;
          }
          if (graceTimer !== null) {
            clearTimeout(graceTimer);
            graceTimer = null;
          }
        };
        const finish = (result) => {
          if (settled) return;
          settled = true;
          clearTimers();
          try {
            socket?.close?.();
          } catch {
          }
          resolve(result);
        };
        try {
          socket = new WebSocketImpl(wsUrl);
        } catch (error) {
          finish({
            ok: false,
            reason: error instanceof Error ? error.message : String(error)
          });
          return;
        }
        const onOpen = () => {
          if (settled) return;
          opened = true;
          graceTimer = setTimeout(() => {
            finish({ ok: true });
          }, readyGraceMs);
        };
        const onMessage = () => {
          finish({ ok: true });
        };
        const onError = (event) => {
          finish({
            ok: false,
            reason: extractErrorReason(event) || "WebSocket connection failed."
          });
        };
        const onClose = (event) => {
          if (settled) return;
          if (opened) {
            finish({
              ok: false,
              reason: closeReason(event, "The gateway accepted the connection then closed it (credential rejected?).")
            });
            return;
          }
          finish({
            ok: false,
            reason: closeReason(event, "The gateway closed the WebSocket before it opened.")
          });
        };
        addListener(socket, "open", onOpen);
        addListener(socket, "message", onMessage);
        addListener(socket, "error", onError);
        addListener(socket, "close", onClose);
        if (connectTimeoutMs > 0) {
          connectTimer = setTimeout(() => {
            finish({
              ok: false,
              reason: `Timed out after ${connectTimeoutMs}ms waiting for the WebSocket to open.`
            });
          }, connectTimeoutMs);
        }
      });
    }
    function addListener(socket, type, handler) {
      if (typeof socket.addEventListener === "function") {
        socket.addEventListener(type, handler);
        return;
      }
      if (typeof socket.on === "function") {
        socket.on(type, handler);
      }
    }
    function extractErrorReason(event) {
      if (!event) return "";
      if (event instanceof Error) return event.message;
      const err = event.error || event.message;
      if (err instanceof Error) return err.message;
      if (typeof err === "string") return err;
      return "";
    }
    function closeReason(event, fallback) {
      const code = event && typeof event.code === "number" ? event.code : null;
      const reason = event && typeof event.reason === "string" ? event.reason.trim() : "";
      if (code && reason) return `${fallback} (code ${code}: ${reason})`;
      if (code) return `${fallback} (code ${code})`;
      if (reason) return `${fallback} (${reason})`;
      return fallback;
    }
    module2.exports = {
      DEFAULT_CONNECT_TIMEOUT_MS,
      DEFAULT_READY_GRACE_MS,
      probeGatewayWebSocket: probeGatewayWebSocket2
    };
  }
});

// electron/dashboard-token.cjs
var require_dashboard_token = __commonJS({
  "electron/dashboard-token.cjs"(exports2, module2) {
    "use strict";
    var DEFAULT_TOKEN_FETCH_TIMEOUT_MS = 3e3;
    async function fetchPublicText(url, options = {}) {
      const { protocol: protocol2 } = new URL(url);
      if (protocol2 !== "http:" && protocol2 !== "https:") {
        throw new Error(`Unsupported Hermes backend URL protocol: ${protocol2}`);
      }
      const timeoutMs = options.timeoutMs ?? DEFAULT_TOKEN_FETCH_TIMEOUT_MS;
      const res = await fetch(url, { signal: AbortSignal.timeout(timeoutMs) }).catch((error) => {
        if (error.name === "TimeoutError") {
          throw new Error(`Timed out connecting to Hermes backend after ${timeoutMs}ms`);
        }
        throw error;
      });
      const text = await res.text();
      if (!res.ok) throw new Error(`${res.status}: ${text || res.statusText}`);
      return text;
    }
    function extractInjectedDashboardToken(html) {
      const match = /window\.__HERMES_SESSION_TOKEN__\s*=\s*("(?:\\.|[^"\\])*")/.exec(String(html || ""));
      if (!match) return null;
      try {
        return JSON.parse(match[1]);
      } catch {
        return null;
      }
    }
    function dashboardIndexUrl(baseUrl) {
      return `${String(baseUrl || "").replace(/\/+$/, "")}/`;
    }
    async function resolveServedDashboardToken(baseUrl, fallbackToken, options = {}) {
      const fetchText = options.fetchText || fetchPublicText;
      const html = await fetchText(dashboardIndexUrl(baseUrl), {
        timeoutMs: options.timeoutMs ?? DEFAULT_TOKEN_FETCH_TIMEOUT_MS
      });
      const servedToken = extractInjectedDashboardToken(html);
      if (servedToken && servedToken !== fallbackToken && typeof options.rememberLog === "function") {
        options.rememberLog("[boot] dashboard served a different session token; using served token for WebSocket auth");
      }
      return servedToken || fallbackToken;
    }
    function isForeignBackendToken({ servedToken, spawnToken, childAlive }) {
      return Boolean(servedToken) && servedToken !== spawnToken && !childAlive;
    }
    async function adoptServedDashboardToken2(baseUrl, spawnToken, { childAlive, label = "Hermes backend", ...options }) {
      const servedToken = await resolveServedDashboardToken(baseUrl, spawnToken, options).catch((error) => {
        options.rememberLog?.(`[boot] could not read served dashboard token (${label}): ${error.message}`);
        return spawnToken;
      });
      if (isForeignBackendToken({ servedToken, spawnToken, childAlive: childAlive() })) {
        throw new Error(
          `${label} exited and ${dashboardIndexUrl(baseUrl)} is served by a process we did not spawn; refusing its session token.`
        );
      }
      return servedToken;
    }
    module2.exports = {
      DEFAULT_TOKEN_FETCH_TIMEOUT_MS,
      adoptServedDashboardToken: adoptServedDashboardToken2,
      dashboardIndexUrl,
      extractInjectedDashboardToken,
      fetchPublicText,
      isForeignBackendToken,
      resolveServedDashboardToken
    };
  }
});

// electron/backend-ready.cjs
var require_backend_ready = __commonJS({
  "electron/backend-ready.cjs"(exports2, module2) {
    "use strict";
    var fs2 = require("node:fs");
    var _READY_RE = /^HERMES_DASHBOARD_READY port=(\d+)/m;
    var DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS = 9e4;
    var MIN_PORT_ANNOUNCE_TIMEOUT_MS = 45e3;
    function resolvePortAnnounceTimeoutMs(env2 = process.env) {
      const parsed = Number(env2.HERMES_DESKTOP_PORT_ANNOUNCE_TIMEOUT_MS);
      if (Number.isFinite(parsed) && parsed > 0) {
        return Math.max(MIN_PORT_ANNOUNCE_TIMEOUT_MS, Math.round(parsed));
      }
      return DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS;
    }
    function waitForDashboardPort(child, timeoutMs = resolvePortAnnounceTimeoutMs()) {
      return new Promise((resolve, reject) => {
        let buf = "";
        let done = false;
        function cleanup() {
          if (done) return;
          done = true;
          clearTimeout(timer);
          child.stdout.off("data", onData);
          child.off("exit", onExit);
          child.off("error", onError);
        }
        function onData(chunk) {
          buf += chunk.toString();
          let nl;
          while ((nl = buf.indexOf("\n")) !== -1) {
            const line = buf.slice(0, nl);
            buf = buf.slice(nl + 1);
            const m = line.match(_READY_RE);
            if (m) {
              cleanup();
              resolve(parseInt(m[1], 10));
              return;
            }
          }
        }
        function onExit(code, signal) {
          cleanup();
          reject(new Error(`Hermes backend: exited before port announcement (${signal || code})`));
        }
        function onError(err) {
          cleanup();
          reject(err);
        }
        const timer = setTimeout(() => {
          cleanup();
          reject(new Error(`Timed out waiting for Hermes backend port announcement (${timeoutMs}ms)`));
        }, timeoutMs);
        child.stdout.on("data", onData);
        child.on("exit", onExit);
        child.on("error", onError);
      });
    }
    function readDashboardReadyFile(readyFile) {
      if (!readyFile) return null;
      try {
        const parsed = JSON.parse(fs2.readFileSync(readyFile, "utf8"));
        const port = Number(parsed?.port);
        return Number.isInteger(port) && port > 0 ? port : null;
      } catch {
        return null;
      }
    }
    function waitForDashboardReadyFile(readyFile, child, timeoutMs = resolvePortAnnounceTimeoutMs()) {
      return new Promise((resolve, reject) => {
        let done = false;
        let interval = null;
        function cleanup() {
          if (done) return;
          done = true;
          clearTimeout(timer);
          if (interval) clearInterval(interval);
          child.off("exit", onExit);
          child.off("error", onError);
        }
        function check() {
          const port = readDashboardReadyFile(readyFile);
          if (port) {
            cleanup();
            resolve(port);
          }
        }
        function onExit(code, signal) {
          cleanup();
          reject(new Error(`Hermes backend: exited before port announcement (${signal || code})`));
        }
        function onError(err) {
          cleanup();
          reject(err);
        }
        const timer = setTimeout(() => {
          cleanup();
          reject(new Error(`Timed out waiting for Hermes backend port announcement (${timeoutMs}ms)`));
        }, timeoutMs);
        child.on("exit", onExit);
        child.on("error", onError);
        interval = setInterval(check, 50);
        if (typeof interval.unref === "function") interval.unref();
        check();
      });
    }
    function waitForDashboardPortAnnouncement2(child, options = {}) {
      const timeoutMs = options.timeoutMs ?? resolvePortAnnounceTimeoutMs();
      if (options.readyFile) {
        return waitForDashboardReadyFile(options.readyFile, child, timeoutMs);
      }
      return waitForDashboardPort(child, timeoutMs);
    }
    module2.exports = {
      waitForDashboardPort,
      waitForDashboardPortAnnouncement: waitForDashboardPortAnnouncement2,
      waitForDashboardReadyFile,
      readDashboardReadyFile,
      resolvePortAnnounceTimeoutMs,
      DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS,
      MIN_PORT_ANNOUNCE_TIMEOUT_MS
    };
  }
});

// electron/oauth-net-request.cjs
var require_oauth_net_request = __commonJS({
  "electron/oauth-net-request.cjs"(exports2, module2) {
    "use strict";
    function serializeJsonBody2(body) {
      return body === void 0 ? void 0 : Buffer.from(JSON.stringify(body));
    }
    function setJsonRequestHeaders2(request) {
      request.setHeader("Content-Type", "application/json");
    }
    module2.exports = {
      serializeJsonBody: serializeJsonBody2,
      setJsonRequestHeaders: setJsonRequestHeaders2
    };
  }
});

// electron/vscode-marketplace.cjs
var require_vscode_marketplace = __commonJS({
  "electron/vscode-marketplace.cjs"(exports2, module2) {
    "use strict";
    var https2 = require("node:https");
    var zlib = require("node:zlib");
    var GALLERY_QUERY_URL = "https://marketplace.visualstudio.com/_apis/public/gallery/extensionquery";
    var VSIX_ASSET_TYPE = "Microsoft.VisualStudio.Services.VSIXPackage";
    var MAX_VSIX_BYTES = 40 * 1024 * 1024;
    var MAX_REDIRECTS = 5;
    var REQUEST_TIMEOUT_MS = 2e4;
    var ID_RE = /^[\w-]+\.[\w-]+$/;
    function request(url, { method = "GET", headers = {}, body = null, maxBytes = MAX_VSIX_BYTES } = {}, redirectsLeft = MAX_REDIRECTS) {
      return new Promise((resolve, reject) => {
        const req = https2.request(url, { method, headers }, (res) => {
          const status = res.statusCode ?? 0;
          if (status >= 300 && status < 400 && res.headers.location) {
            if (redirectsLeft <= 0) {
              res.resume();
              reject(new Error("Too many redirects."));
              return;
            }
            const next = new URL(res.headers.location, url).toString();
            res.resume();
            resolve(request(next, { method: "GET", headers: { "User-Agent": headers["User-Agent"] }, maxBytes }, redirectsLeft - 1));
            return;
          }
          if (status < 200 || status >= 300) {
            res.resume();
            reject(new Error(`Request failed (${status}) for ${url}`));
            return;
          }
          const chunks = [];
          let total = 0;
          res.on("data", (chunk) => {
            total += chunk.length;
            if (total > maxBytes) {
              req.destroy();
              reject(new Error("Response exceeded the size limit."));
              return;
            }
            chunks.push(chunk);
          });
          res.on("end", () => resolve(Buffer.concat(chunks)));
        });
        req.on("error", reject);
        req.setTimeout(REQUEST_TIMEOUT_MS, () => req.destroy(new Error("Request timed out.")));
        if (body) {
          req.write(body);
        }
        req.end();
      });
    }
    async function resolveExtension(id) {
      const json = await queryGallery({
        // FilterType 7 = ExtensionName (the full publisher.extension id).
        filters: [{ criteria: [{ filterType: 7, value: id }], pageNumber: 1, pageSize: 1 }],
        // Flags: IncludeFiles | IncludeVersionProperties | IncludeAssetUri |
        // IncludeCategoryAndTags | IncludeLatestVersionOnly = 914.
        flags: 914
      });
      const extension = json?.results?.[0]?.extensions?.[0];
      if (!extension) {
        throw new Error(`Extension "${id}" was not found on the Marketplace.`);
      }
      const version = extension.versions?.[0];
      if (!version) {
        throw new Error(`Extension "${id}" has no published versions.`);
      }
      const asset = (version.files ?? []).find((file) => file.assetType === VSIX_ASSET_TYPE);
      const vsixUrl = asset?.source;
      if (!vsixUrl) {
        throw new Error(`Could not find a downloadable package for "${id}".`);
      }
      return { displayName: extension.displayName || id, vsixUrl };
    }
    async function queryGallery(payload, { maxBytes = 4 * 1024 * 1024 } = {}) {
      const body = JSON.stringify(payload);
      const raw = await request(GALLERY_QUERY_URL, {
        method: "POST",
        headers: {
          Accept: "application/json;api-version=3.0-preview.1",
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(body),
          "User-Agent": "Hermes-Desktop"
        },
        body,
        maxBytes
      });
      return JSON.parse(raw.toString("utf8"));
    }
    function looksLikeIconTheme(extension) {
      const tags = (extension.tags ?? []).map((tag) => String(tag).toLowerCase());
      if (tags.includes("icon-theme") || tags.includes("product-icon-theme")) {
        return true;
      }
      const text = `${extension.displayName ?? ""} ${extension.shortDescription ?? ""}`.toLowerCase();
      return /\b(icon theme|file icons?|product icons?|icon pack|fileicons)\b/.test(text);
    }
    async function searchMarketplaceThemes2(query, limit = 20) {
      const text = String(query || "").trim();
      const pageSize = Math.min(Math.max(Number(limit) || 20, 1), 50);
      const criteria = [
        { filterType: 8, value: "Microsoft.VisualStudio.Code" },
        { filterType: 5, value: "Themes" },
        { filterType: 12, value: "4096" }
        // Exclude unpublished (Unpublished = 0x1000).
      ];
      if (text) {
        criteria.push({ filterType: 10, value: text });
      }
      const json = await queryGallery({
        // Over-fetch so the icon-theme filter below still leaves a full page.
        filters: [{ criteria, pageNumber: 1, pageSize: Math.min(pageSize * 2, 50), sortBy: 4, sortOrder: 0 }],
        // IncludeStatistics (0x100) | IncludeLatestVersionOnly (0x200) | IncludeCategoryAndTags (0x4).
        flags: 772
      });
      const extensions = json?.results?.[0]?.extensions ?? [];
      return extensions.filter((extension) => !looksLikeIconTheme(extension)).slice(0, pageSize).map((extension) => {
        const publisherName = extension.publisher?.publisherName ?? "";
        const installStat = (extension.statistics ?? []).find((stat) => stat.statisticName === "install");
        return {
          extensionId: `${publisherName}.${extension.extensionName}`,
          displayName: extension.displayName || extension.extensionName,
          publisher: extension.publisher?.displayName || publisherName,
          description: extension.shortDescription || "",
          installs: Math.round(installStat?.value ?? 0)
        };
      });
    }
    function findEndOfCentralDirectory(buf) {
      for (let i = buf.length - 22; i >= 0; i--) {
        if (buf.readUInt32LE(i) === 101010256) {
          return i;
        }
      }
      throw new Error("Not a valid zip archive (no end-of-central-directory).");
    }
    function readCentralDirectory(buf) {
      const eocd = findEndOfCentralDirectory(buf);
      const count = buf.readUInt16LE(eocd + 10);
      let offset = buf.readUInt32LE(eocd + 16);
      const records = /* @__PURE__ */ new Map();
      for (let i = 0; i < count; i++) {
        if (buf.readUInt32LE(offset) !== 33639248) {
          break;
        }
        const method = buf.readUInt16LE(offset + 10);
        const compressedSize = buf.readUInt32LE(offset + 20);
        const nameLen = buf.readUInt16LE(offset + 28);
        const extraLen = buf.readUInt16LE(offset + 30);
        const commentLen = buf.readUInt16LE(offset + 32);
        const localOffset = buf.readUInt32LE(offset + 42);
        const name = buf.toString("utf8", offset + 46, offset + 46 + nameLen);
        records.set(name, { method, compressedSize, localOffset });
        offset += 46 + nameLen + extraLen + commentLen;
      }
      return records;
    }
    function extractEntry(buf, record) {
      if (buf.readUInt32LE(record.localOffset) !== 67324752) {
        throw new Error("Corrupt zip: bad local file header.");
      }
      const nameLen = buf.readUInt16LE(record.localOffset + 26);
      const extraLen = buf.readUInt16LE(record.localOffset + 28);
      const dataStart = record.localOffset + 30 + nameLen + extraLen;
      const data = buf.subarray(dataStart, dataStart + record.compressedSize);
      return record.method === 0 ? data.toString("utf8") : zlib.inflateRawSync(data).toString("utf8");
    }
    function themeEntryName(themePath) {
      const clean = String(themePath).replace(/^\.\//, "").replace(/^\//, "");
      return `extension/${clean}`;
    }
    function extractThemes(vsixBuffer) {
      const records = readCentralDirectory(vsixBuffer);
      const pkgRecord = records.get("extension/package.json");
      if (!pkgRecord) {
        throw new Error("Package manifest missing from the extension.");
      }
      const pkg = JSON.parse(extractEntry(vsixBuffer, pkgRecord));
      const contributed = pkg?.contributes?.themes;
      if (!Array.isArray(contributed) || contributed.length === 0) {
        return [];
      }
      const themes = [];
      for (const entry of contributed) {
        if (!entry?.path) {
          continue;
        }
        const record = records.get(themeEntryName(entry.path));
        if (!record) {
          continue;
        }
        try {
          themes.push({
            label: entry.label || entry.id || pkg.displayName || pkg.name || "VS Code Theme",
            uiTheme: entry.uiTheme,
            contents: extractEntry(vsixBuffer, record)
          });
        } catch {
        }
      }
      return themes;
    }
    async function fetchMarketplaceThemes2(id) {
      const trimmed = String(id || "").trim();
      if (!ID_RE.test(trimmed)) {
        throw new Error('Expected a Marketplace id like "publisher.extension".');
      }
      const { displayName, vsixUrl } = await resolveExtension(trimmed);
      const vsix = await request(vsixUrl, { headers: { "User-Agent": "Hermes-Desktop" } });
      const themes = extractThemes(vsix);
      return { extensionId: trimmed, displayName, themes };
    }
    module2.exports = {
      fetchMarketplaceThemes: fetchMarketplaceThemes2,
      searchMarketplaceThemes: searchMarketplaceThemes2,
      extractThemes,
      readCentralDirectory,
      __testing: { themeEntryName, looksLikeIconTheme }
    };
  }
});

// electron/backend-env.cjs
var require_backend_env = __commonJS({
  "electron/backend-env.cjs"(exports2, module2) {
    "use strict";
    var path2 = require("node:path");
    var POSIX_SANE_PATH_ENTRIES = Object.freeze([
      "/opt/homebrew/bin",
      "/opt/homebrew/sbin",
      "/usr/local/sbin",
      "/usr/local/bin",
      "/usr/sbin",
      "/usr/bin",
      "/sbin",
      "/bin"
    ]);
    function delimiterForPlatform(platform = process.platform) {
      return platform === "win32" ? ";" : ":";
    }
    function pathModuleForPlatform(platform = process.platform) {
      return platform === "win32" ? path2.win32 : path2.posix;
    }
    function pathEnvKey(env2 = process.env, platform = process.platform) {
      if (platform !== "win32") return "PATH";
      return Object.keys(env2 || {}).find((key) => key.toUpperCase() === "PATH") || "PATH";
    }
    function currentPathValue(env2 = process.env, platform = process.platform) {
      const key = pathEnvKey(env2, platform);
      return env2?.[key] || "";
    }
    function appendUniquePathEntries(entries, { delimiter = path2.delimiter } = {}) {
      const seen = /* @__PURE__ */ new Set();
      const ordered = [];
      for (const entry of entries) {
        if (!entry) continue;
        const parts = Array.isArray(entry) ? entry : String(entry).split(delimiter);
        for (const part of parts) {
          if (!part || seen.has(part)) continue;
          seen.add(part);
          ordered.push(part);
        }
      }
      return ordered.join(delimiter);
    }
    function buildDesktopBackendPath({
      hermesHome,
      venvRoot,
      currentPath = "",
      platform = process.platform,
      pathModule = pathModuleForPlatform(platform)
    } = {}) {
      const delimiter = delimiterForPlatform(platform);
      const hermesNodeBin = hermesHome ? pathModule.join(hermesHome, "node", "bin") : null;
      const venvBin = venvRoot ? pathModule.join(venvRoot, platform === "win32" ? "Scripts" : "bin") : null;
      const saneEntries = platform === "win32" ? [] : POSIX_SANE_PATH_ENTRIES;
      return appendUniquePathEntries(
        [hermesNodeBin, venvBin, currentPath, saneEntries],
        { delimiter }
      );
    }
    function normalizeHermesHomeRoot2(hermesHome, { pathModule = pathModuleForPlatform(process.platform) } = {}) {
      if (!hermesHome) return hermesHome;
      const resolved = pathModule.resolve(String(hermesHome));
      const parent = pathModule.dirname(resolved);
      if (pathModule.basename(parent).toLowerCase() === "profiles") {
        return pathModule.dirname(parent);
      }
      return resolved;
    }
    function buildDesktopBackendEnv2({
      hermesHome,
      pythonPathEntries = [],
      venvRoot,
      currentEnv = process.env,
      platform = process.platform,
      pathModule = pathModuleForPlatform(platform)
    } = {}) {
      const delimiter = delimiterForPlatform(platform);
      const currentPythonPath = currentEnv?.PYTHONPATH || "";
      const key = pathEnvKey(currentEnv, platform);
      return {
        PYTHONPATH: appendUniquePathEntries([...pythonPathEntries, currentPythonPath], { delimiter }),
        [key]: buildDesktopBackendPath({
          hermesHome,
          venvRoot,
          currentPath: currentPathValue(currentEnv, platform),
          platform,
          pathModule
        })
      };
    }
    module2.exports = {
      POSIX_SANE_PATH_ENTRIES,
      appendUniquePathEntries,
      buildDesktopBackendEnv: buildDesktopBackendEnv2,
      buildDesktopBackendPath,
      delimiterForPlatform,
      normalizeHermesHomeRoot: normalizeHermesHomeRoot2,
      pathEnvKey
    };
  }
});

// electron/windows-user-env.cjs
var require_windows_user_env = __commonJS({
  "electron/windows-user-env.cjs"(exports2, module2) {
    "use strict";
    var { execFileSync: execFileSync2 } = require("node:child_process");
    function parseRegQueryValue(stdout, name) {
      if (!stdout || !name) return null;
      const typePattern = /^(\S+)\s+(?:REG_SZ|REG_EXPAND_SZ|REG_MULTI_SZ|REG_DWORD|REG_QWORD|REG_BINARY|REG_NONE)\s+(.*)$/;
      for (const rawLine of String(stdout).split(/\r?\n/)) {
        const line = rawLine.trim();
        const match = line.match(typePattern);
        if (match && match[1].toLowerCase() === name.toLowerCase()) {
          return match[2];
        }
      }
      return null;
    }
    function expandWindowsEnvRefs(value, env2 = process.env) {
      if (!value) return value;
      return value.replace(/%([^%]+)%/g, (whole, name) => {
        const key = Object.keys(env2).find((k) => k.toUpperCase() === String(name).toUpperCase());
        return key != null && env2[key] != null ? env2[key] : whole;
      });
    }
    function readWindowsUserEnvVar2(name, { platform = process.platform, env: env2 = process.env, exec = execFileSync2 } = {}) {
      if (platform !== "win32" || !name) return null;
      let stdout;
      try {
        stdout = exec("reg", ["query", "HKCU\\Environment", "/v", name], {
          encoding: "utf8",
          windowsHide: true,
          timeout: 5e3
        });
      } catch {
        return null;
      }
      const raw = parseRegQueryValue(stdout, name);
      if (raw == null) return null;
      const expanded = expandWindowsEnvRefs(raw, env2).trim();
      return expanded || null;
    }
    module2.exports = {
      expandWindowsEnvRefs,
      parseRegQueryValue,
      readWindowsUserEnvVar: readWindowsUserEnvVar2
    };
  }
});

// electron/hardening.cjs
var require_hardening = __commonJS({
  "electron/hardening.cjs"(exports2, module2) {
    "use strict";
    var fs2 = require("node:fs");
    var os2 = require("node:os");
    var path2 = require("node:path");
    var { fileURLToPath } = require("node:url");
    var DEFAULT_FETCH_TIMEOUT_MS2 = 15e3;
    var DATA_URL_READ_MAX_BYTES2 = 16 * 1024 * 1024;
    var TEXT_PREVIEW_SOURCE_MAX_BYTES2 = 64 * 1024 * 1024;
    var SAFE_ENV_SUFFIXES = /* @__PURE__ */ new Set(["dist", "example", "sample", "template"]);
    var SENSITIVE_EXTENSIONS = /* @__PURE__ */ new Set([".kdbx", ".p12", ".pem", ".pfx"]);
    function resolveTimeoutMs2(timeoutMs, fallbackMs = DEFAULT_FETCH_TIMEOUT_MS2) {
      const fallback = Number.isFinite(fallbackMs) && Number(fallbackMs) > 0 ? Math.round(Number(fallbackMs)) : DEFAULT_FETCH_TIMEOUT_MS2;
      const parsed = Number(timeoutMs);
      if (Number.isFinite(parsed) && parsed > 0) {
        return Math.round(parsed);
      }
      return fallback;
    }
    function encryptDesktopSecret2(value, safeStorageApi) {
      const raw = String(value || "");
      if (!raw) {
        return null;
      }
      let encryptionAvailable = false;
      try {
        encryptionAvailable = Boolean(safeStorageApi?.isEncryptionAvailable?.());
      } catch {
        encryptionAvailable = false;
      }
      if (!encryptionAvailable) {
        throw new Error(
          "Secure token storage is unavailable, so Hermes Desktop cannot save remote gateway tokens. Set HERMES_DESKTOP_REMOTE_URL and HERMES_DESKTOP_REMOTE_TOKEN in your environment, or enable OS keychain access and try again."
        );
      }
      try {
        return {
          encoding: "safeStorage",
          value: safeStorageApi.encryptString(raw).toString("base64")
        };
      } catch (error) {
        const detail = error instanceof Error && error.message ? ` (${error.message})` : "";
        throw new Error(
          `Failed to encrypt the remote gateway token for secure storage${detail}. Set HERMES_DESKTOP_REMOTE_URL and HERMES_DESKTOP_REMOTE_TOKEN in your environment as a fallback.`
        );
      }
    }
    function sensitiveFileBlockReason(filePath) {
      const normalized = String(filePath || "").replace(/\\/g, "/").toLowerCase();
      const basename = path2.basename(normalized);
      const ext = path2.extname(basename);
      if (!basename) {
        return null;
      }
      if (normalized.includes("/.ssh/")) {
        return "SSH key/config files are blocked.";
      }
      if (normalized.includes("/.gnupg/")) {
        return "GPG key material is blocked.";
      }
      if (normalized.endsWith("/.aws/credentials")) {
        return "AWS credential files are blocked.";
      }
      if (basename === ".env") {
        return ".env files are blocked because they commonly contain secrets.";
      }
      if (basename.startsWith(".env.")) {
        const suffix = basename.slice(".env.".length);
        if (!SAFE_ENV_SUFFIXES.has(suffix)) {
          return `${basename} is blocked because it appears to contain environment secrets.`;
        }
      }
      if (/^id_(rsa|dsa|ecdsa|ed25519)(?:\..+)?$/.test(basename) && !basename.endsWith(".pub")) {
        return "SSH private key files are blocked.";
      }
      if (SENSITIVE_EXTENSIONS.has(ext)) {
        return `${ext} key/certificate files are blocked.`;
      }
      if (basename === ".npmrc" || basename === ".netrc" || basename === ".pypirc") {
        return `${basename} is blocked because it may include auth credentials.`;
      }
      return null;
    }
    function ipcPathError(code, message) {
      const error = new Error(message);
      error.code = code;
      return error;
    }
    function rejectUnsafePathSyntax(filePath, purpose = "File read") {
      if (typeof filePath !== "string") {
        throw ipcPathError("invalid-path", `${purpose} failed: file path is required.`);
      }
      const raw = filePath.trim();
      if (!raw) {
        throw ipcPathError("invalid-path", `${purpose} failed: file path is required.`);
      }
      if (raw.includes("\0")) {
        throw ipcPathError("invalid-path", `${purpose} failed: file path is invalid.`);
      }
      const normalized = raw.replace(/\\/g, "/").toLowerCase();
      if (normalized.startsWith("//?/") || normalized.startsWith("//./") || normalized.startsWith("globalroot/device/") || normalized.includes("/globalroot/device/")) {
        throw ipcPathError("device-path", `${purpose} blocked: Windows device paths are not allowed.`);
      }
      return raw;
    }
    function resolveRequestedPathForIpc2(filePath, options = {}) {
      const purpose = String(options.purpose || "File read");
      let raw = rejectUnsafePathSyntax(filePath, purpose);
      if (raw === "~" || raw.startsWith("~/") || raw.startsWith("~\\")) {
        raw = path2.join(os2.homedir(), raw.slice(1));
      }
      if (/^file:/i.test(raw)) {
        let resolvedPath2;
        try {
          const parsed = new URL(raw);
          if (parsed.protocol !== "file:") {
            throw new Error("not a file URL");
          }
          resolvedPath2 = fileURLToPath(parsed);
        } catch {
          throw ipcPathError("invalid-path", `${purpose} failed: file URL is invalid.`);
        }
        rejectUnsafePathSyntax(resolvedPath2, purpose);
        return path2.resolve(resolvedPath2);
      }
      const baseInput = typeof options.baseDir === "string" && options.baseDir.trim() ? options.baseDir : process.cwd();
      const safeBaseInput = rejectUnsafePathSyntax(baseInput, purpose);
      const resolvedBase = path2.resolve(safeBaseInput);
      rejectUnsafePathSyntax(resolvedBase, purpose);
      const resolvedPath = path2.resolve(resolvedBase, raw);
      rejectUnsafePathSyntax(resolvedPath, purpose);
      return resolvedPath;
    }
    async function statForIpc(fsImpl, resolvedPath, purpose, typeLabel) {
      try {
        return await fsImpl.promises.stat(resolvedPath);
      } catch (error) {
        const code = error && typeof error === "object" ? error.code : "";
        if (code === "ENOENT" || code === "ENOTDIR") {
          throw ipcPathError(code || "ENOENT", `${purpose} failed: ${typeLabel} does not exist.`);
        }
        throw ipcPathError(code || "read-error", `${purpose} failed: ${error instanceof Error ? error.message : String(error)}`);
      }
    }
    async function realpathForIpc(fsImpl, resolvedPath, purpose) {
      if (typeof fsImpl.promises.realpath !== "function") {
        return resolvedPath;
      }
      try {
        const realPath = await fsImpl.promises.realpath(resolvedPath);
        rejectUnsafePathSyntax(realPath, purpose);
        return realPath;
      } catch (error) {
        const code = error && typeof error === "object" ? error.code : "";
        throw ipcPathError(code || "read-error", `${purpose} failed: ${error instanceof Error ? error.message : String(error)}`);
      }
    }
    function rejectSensitiveFilePath(filePath, purpose) {
      const blockReason = sensitiveFileBlockReason(filePath);
      if (blockReason) {
        throw ipcPathError("sensitive-file", `${purpose} blocked for sensitive file: ${blockReason}`);
      }
    }
    async function resolveDirectoryForIpc(dirPath, options = {}) {
      const purpose = String(options.purpose || "Directory read");
      const fsImpl = options.fs || fs2;
      const resolvedPath = resolveRequestedPathForIpc2(dirPath, { baseDir: options.baseDir, purpose });
      const stat = await statForIpc(fsImpl, resolvedPath, purpose, "directory");
      if (!stat.isDirectory()) {
        throw ipcPathError("ENOTDIR", `${purpose} failed: path is not a directory.`);
      }
      const realPath = await realpathForIpc(fsImpl, resolvedPath, purpose);
      return { realPath, resolvedPath, stat };
    }
    async function resolveReadableFileForIpc2(filePath, options = {}) {
      const purpose = String(options.purpose || "File read");
      const fsImpl = options.fs || fs2;
      const resolvedPath = resolveRequestedPathForIpc2(filePath, { baseDir: options.baseDir, purpose });
      if (options.blockSensitive !== false) {
        rejectSensitiveFilePath(resolvedPath, purpose);
      }
      const stat = await statForIpc(fsImpl, resolvedPath, purpose, "file");
      if (stat.isDirectory()) {
        throw ipcPathError("EISDIR", `${purpose} failed: path points to a directory.`);
      }
      if (!stat.isFile()) {
        throw ipcPathError("EINVAL", `${purpose} failed: only regular files can be read.`);
      }
      const realPath = await realpathForIpc(fsImpl, resolvedPath, purpose);
      if (options.blockSensitive !== false) {
        rejectSensitiveFilePath(realPath, purpose);
      }
      const maxBytes = Number.isFinite(options.maxBytes) && Number(options.maxBytes) > 0 ? Number(options.maxBytes) : null;
      if (maxBytes && stat.size > maxBytes) {
        throw ipcPathError("EFBIG", `${purpose} failed: file is too large (${stat.size} bytes; limit ${maxBytes} bytes).`);
      }
      try {
        await fsImpl.promises.access(resolvedPath, fs2.constants.R_OK);
      } catch {
        throw ipcPathError("EACCES", `${purpose} failed: file is not readable.`);
      }
      return { realPath, resolvedPath, stat };
    }
    module2.exports = {
      DATA_URL_READ_MAX_BYTES: DATA_URL_READ_MAX_BYTES2,
      DEFAULT_FETCH_TIMEOUT_MS: DEFAULT_FETCH_TIMEOUT_MS2,
      TEXT_PREVIEW_SOURCE_MAX_BYTES: TEXT_PREVIEW_SOURCE_MAX_BYTES2,
      encryptDesktopSecret: encryptDesktopSecret2,
      rejectUnsafePathSyntax,
      resolveDirectoryForIpc,
      resolveReadableFileForIpc: resolveReadableFileForIpc2,
      resolveRequestedPathForIpc: resolveRequestedPathForIpc2,
      resolveTimeoutMs: resolveTimeoutMs2,
      sensitiveFileBlockReason
    };
  }
});

// electron/fs-read-dir.cjs
var require_fs_read_dir = __commonJS({
  "electron/fs-read-dir.cjs"(exports2, module2) {
    "use strict";
    var fs2 = require("node:fs");
    var path2 = require("node:path");
    var { resolveDirectoryForIpc } = require_hardening();
    var FS_READDIR_STAT_CONCURRENCY = 16;
    var FS_READDIR_HIDDEN = /* @__PURE__ */ new Set([
      ".git",
      ".hg",
      ".svn",
      ".cache",
      ".next",
      ".turbo",
      ".venv",
      "__pycache__",
      "build",
      "dist",
      "node_modules",
      "target",
      "venv"
    ]);
    function direntIsDirectory(dirent) {
      return typeof dirent.isDirectory === "function" && dirent.isDirectory();
    }
    function direntIsFile(dirent) {
      return typeof dirent.isFile === "function" && dirent.isFile();
    }
    function direntIsSymbolicLink(dirent) {
      return typeof dirent.isSymbolicLink === "function" && dirent.isSymbolicLink();
    }
    function shouldStatDirent(dirent) {
      if (direntIsDirectory(dirent)) return false;
      return direntIsSymbolicLink(dirent) || !direntIsFile(dirent);
    }
    async function entryForDirent(dirent, resolved, fsImpl) {
      const fullPath = path2.join(resolved, dirent.name);
      let isDirectory = direntIsDirectory(dirent);
      if (!isDirectory && shouldStatDirent(dirent)) {
        try {
          isDirectory = (await fsImpl.promises.stat(fullPath)).isDirectory();
        } catch {
          isDirectory = false;
        }
      }
      return { name: dirent.name, path: fullPath, isDirectory };
    }
    async function mapWithStatConcurrency(items, mapper) {
      const results = new Array(items.length);
      let nextIndex = 0;
      async function runWorker() {
        while (nextIndex < items.length) {
          const index = nextIndex;
          nextIndex += 1;
          results[index] = await mapper(items[index]);
        }
      }
      const workerCount = Math.min(FS_READDIR_STAT_CONCURRENCY, items.length);
      const workers = Array.from({ length: workerCount }, () => runWorker());
      await Promise.all(workers);
      return results;
    }
    async function readDirForIpc2(dirPath, options = {}) {
      const fsImpl = options.fs || fs2;
      let resolved;
      try {
        ;
        ({ resolvedPath: resolved } = await resolveDirectoryForIpc(dirPath, {
          fs: fsImpl,
          purpose: "Directory read"
        }));
      } catch (error) {
        return { entries: [], error: error?.code || "read-error" };
      }
      try {
        const dirents = await fsImpl.promises.readdir(resolved, { withFileTypes: true });
        const visibleDirents = dirents.filter((dirent) => !FS_READDIR_HIDDEN.has(dirent.name));
        const entries = await mapWithStatConcurrency(
          visibleDirents,
          (dirent) => entryForDirent(dirent, resolved, fsImpl)
        );
        entries.sort((a, b) => Number(b.isDirectory) - Number(a.isDirectory) || a.name.localeCompare(b.name));
        return { entries };
      } catch (error) {
        return { entries: [], error: error?.code || "read-error" };
      }
    }
    module2.exports = {
      readDirForIpc: readDirForIpc2
    };
  }
});

// electron/update-marker.cjs
var require_update_marker = __commonJS({
  "electron/update-marker.cjs"(exports2, module2) {
    "use strict";
    var fs2 = require("fs");
    var path2 = require("path");
    var UPDATE_MARKER_MAX_AGE_MS = 20 * 60 * 1e3;
    function markerPath(hermesHome) {
      return path2.join(hermesHome, ".hermes-update-in-progress");
    }
    function isPidAlive(pid, kill = process.kill.bind(process)) {
      if (!Number.isInteger(pid) || pid <= 0) return false;
      try {
        kill(pid, 0);
        return true;
      } catch (err) {
        return Boolean(err && err.code === "EPERM");
      }
    }
    function readLiveUpdateMarker2(hermesHome, { kill, now = Date.now, maxAgeMs = UPDATE_MARKER_MAX_AGE_MS } = {}) {
      const file = markerPath(hermesHome);
      let raw;
      try {
        raw = fs2.readFileSync(file, "utf8");
      } catch {
        return null;
      }
      const [pidLine, startedLine] = String(raw).split("\n");
      const pid = Number.parseInt((pidLine || "").trim(), 10);
      const startedAt = Number.parseInt((startedLine || "").trim(), 10);
      const ageMs = Number.isFinite(startedAt) ? now() - startedAt * 1e3 : Infinity;
      const alive = Number.isInteger(pid) && isPidAlive(pid, kill);
      if (!alive || ageMs > maxAgeMs) {
        try {
          fs2.unlinkSync(file);
        } catch {
        }
        return null;
      }
      return { pid, ageMs };
    }
    module2.exports = {
      UPDATE_MARKER_MAX_AGE_MS,
      markerPath,
      isPidAlive,
      readLiveUpdateMarker: readLiveUpdateMarker2
    };
  }
});

// electron/update-relaunch.cjs
var require_update_relaunch = __commonJS({
  "electron/update-relaunch.cjs"(exports2, module2) {
    "use strict";
    var path2 = require("node:path");
    function unpackedDirName(platform) {
      if (platform === "darwin") return "mac-unpacked";
      if (platform === "win32") return "win-unpacked";
      return "linux-unpacked";
    }
    function resolveUnpackedRelease2(execPath, updateRoot, platform) {
      if (!execPath || !updateRoot) return null;
      const releaseDir = path2.join(updateRoot, "apps", "desktop", "release");
      const unpacked = path2.join(releaseDir, unpackedDirName(platform));
      const normalizedExec = path2.resolve(String(execPath));
      const withSep = unpacked.endsWith(path2.sep) ? unpacked : unpacked + path2.sep;
      if (normalizedExec === unpacked || normalizedExec.startsWith(withSep)) {
        return unpacked;
      }
      return null;
    }
    function decideRelaunchOutcome2({ underUnpacked, sandboxOk }) {
      if (!underUnpacked) return "guiSkew";
      if (!sandboxOk) return "manual";
      return "relaunch";
    }
    function sandboxPreflight2(unpackedDir, statSync) {
      if (!unpackedDir) return { ok: false, reason: "no-unpacked-dir", path: null };
      const sandboxPath = path2.join(unpackedDir, "chrome-sandbox");
      let st;
      try {
        st = statSync(sandboxPath);
      } catch {
        return { ok: true, reason: "no-sandbox-helper", path: sandboxPath };
      }
      const ownedByRoot = st.uid === 0;
      const hasSetuid = (st.mode & 2048) !== 0;
      if (ownedByRoot && hasSetuid) {
        return { ok: true, reason: "launchable", path: sandboxPath };
      }
      if (!ownedByRoot && !hasSetuid) {
        return { ok: false, reason: "not-root-not-setuid", path: sandboxPath };
      }
      if (!ownedByRoot) return { ok: false, reason: "not-root", path: sandboxPath };
      return { ok: false, reason: "not-setuid", path: sandboxPath };
    }
    function sandboxFallbackFromEnv2(env2, launchArgs) {
      const disable = String(env2 && env2.ELECTRON_DISABLE_SANDBOX || "").trim();
      if (disable === "1" || disable.toLowerCase() === "true") return true;
      if (Array.isArray(launchArgs) && launchArgs.some((a) => a === "--no-sandbox")) return true;
      return false;
    }
    function shellQuote2(value) {
      return `'${String(value).replace(/'/g, `'\\''`)}'`;
    }
    var INTERNAL_ARG_PREFIXES = [
      "--type=",
      // renderer/gpu/zygote child markers
      "--user-data-dir=",
      "--enable-features=",
      "--disable-features=",
      "--field-trial-handle=",
      "--enable-logging",
      "--log-file=",
      // NB: --no-sandbox is deliberately NOT stripped — it reflects the user's /
      // environment's SUID-sandbox opt-out (some hardened kernels/containers require
      // it) and is the signal sandboxFallbackFromEnv() uses to allow a relaunch when
      // chrome-sandbox isn't setuid. Dropping it would make exactly that relaunch
      // fail ("quit and never came back").
      "--disable-gpu-sandbox",
      "--lang=",
      "--inspect",
      "--remote-debugging-port="
    ];
    function collectRelaunchArgs2(argv) {
      if (!Array.isArray(argv)) return [];
      return argv.filter((arg) => {
        if (typeof arg !== "string" || arg.length === 0) return false;
        return !INTERNAL_ARG_PREFIXES.some(
          (prefix) => prefix.endsWith("=") ? arg.startsWith(prefix) : arg === prefix || arg.startsWith(prefix + "=")
        );
      });
    }
    var PRESERVED_ENV_KEYS = ["HERMES_HOME", "ELECTRON_DISABLE_SANDBOX"];
    var PRESERVED_ENV_PREFIXES = ["HERMES_DESKTOP_"];
    function collectRelaunchEnv2(env2) {
      const out = {};
      if (!env2 || typeof env2 !== "object") return out;
      for (const [key, value] of Object.entries(env2)) {
        if (value == null) continue;
        if (PRESERVED_ENV_KEYS.includes(key) || PRESERVED_ENV_PREFIXES.some((p) => key.startsWith(p))) {
          out[key] = String(value);
        }
      }
      return out;
    }
    function buildRelaunchScript2({ pid, execPath, args, env: env2, cwd }) {
      const exports3 = Object.entries(env2 || {}).map(([k, v]) => `export ${k}=${shellQuote2(v)}`).join("\n");
      const quotedArgs = (args || []).map(shellQuote2).join(" ");
      const cwdLine = cwd ? `cd ${shellQuote2(cwd)} 2>/dev/null || true` : "";
      return `#!/bin/bash
set -u
APP_PID=${Number(pid)}
# Wait up to ~30s for a graceful exit, then SIGKILL: a hung/zombie parent must
# be gone before we relaunch, or the new instance bails on the single-instance
# lock. (#45205)
for _ in $(seq 1 60); do
  kill -0 "$APP_PID" 2>/dev/null || break
  sleep 0.5
done
if kill -0 "$APP_PID" 2>/dev/null; then
  kill -9 "$APP_PID" 2>/dev/null || true
  sleep 0.5
fi
# Self-delete so temp watchers don't accumulate across updates.
rm -f -- "$0" 2>/dev/null || true
${cwdLine}
${exports3}
exec ${shellQuote2(execPath)}${quotedArgs ? " " + quotedArgs : ""}
`;
    }
    module2.exports = {
      unpackedDirName,
      resolveUnpackedRelease: resolveUnpackedRelease2,
      decideRelaunchOutcome: decideRelaunchOutcome2,
      sandboxPreflight: sandboxPreflight2,
      sandboxFallbackFromEnv: sandboxFallbackFromEnv2,
      collectRelaunchArgs: collectRelaunchArgs2,
      collectRelaunchEnv: collectRelaunchEnv2,
      buildRelaunchScript: buildRelaunchScript2,
      shellQuote: shellQuote2,
      INTERNAL_ARG_PREFIXES,
      PRESERVED_ENV_KEYS,
      PRESERVED_ENV_PREFIXES
    };
  }
});

// electron/git-root.cjs
var require_git_root = __commonJS({
  "electron/git-root.cjs"(exports2, module2) {
    "use strict";
    var fs2 = require("node:fs");
    var path2 = require("node:path");
    var { resolveRequestedPathForIpc: resolveRequestedPathForIpc2 } = require_hardening();
    function findGitRoot(start, fsImpl = fs2) {
      let dir = start;
      for (let i = 0; i < 50; i += 1) {
        try {
          if (fsImpl.existsSync(path2.join(dir, ".git"))) {
            return dir;
          }
        } catch {
          return null;
        }
        const parent = path2.dirname(dir);
        if (parent === dir) {
          return null;
        }
        dir = parent;
      }
      return null;
    }
    async function gitRootForIpc2(startPath, options = {}) {
      const fsImpl = options.fs || fs2;
      let resolved;
      try {
        resolved = resolveRequestedPathForIpc2(startPath, { purpose: "Git root" });
      } catch {
        return null;
      }
      try {
        const stat = await fsImpl.promises.stat(resolved);
        const start = stat.isDirectory() ? resolved : path2.dirname(resolved);
        return findGitRoot(start, fsImpl);
      } catch {
        return findGitRoot(resolved, fsImpl);
      }
    }
    module2.exports = {
      findGitRoot,
      gitRootForIpc: gitRootForIpc2
    };
  }
});

// electron/git-worktree-ops.cjs
var require_git_worktree_ops = __commonJS({
  "electron/git-worktree-ops.cjs"(exports2, module2) {
    "use strict";
    var path2 = require("node:path");
    var fs2 = require("node:fs");
    var { execFile } = require("node:child_process");
    var { resolveRequestedPathForIpc: resolveRequestedPathForIpc2 } = require_hardening();
    function runGit2(gitBin, args, cwd) {
      return new Promise((resolve, reject) => {
        execFile(
          gitBin,
          args,
          { cwd, windowsHide: true, timeout: 3e4, maxBuffer: 8 * 1024 * 1024 },
          (err, stdout, stderr) => {
            if (err) {
              err.stderr = String(stderr || "");
              reject(err);
              return;
            }
            resolve(String(stdout || ""));
          }
        );
      });
    }
    function parseWorktrees(out) {
      const trees = [];
      let cur = null;
      for (const line of out.split("\n")) {
        if (line.startsWith("worktree ")) {
          if (cur) {
            trees.push(cur);
          }
          cur = { path: line.slice(9).trim(), branch: null, detached: false, bare: false, locked: false };
        } else if (!cur) {
          continue;
        } else if (line.startsWith("branch ")) {
          cur.branch = line.slice(7).trim().replace(/^refs\/heads\//, "");
        } else if (line === "detached") {
          cur.detached = true;
        } else if (line === "bare") {
          cur.bare = true;
        } else if (line.startsWith("locked")) {
          cur.locked = true;
        }
      }
      if (cur) {
        trees.push(cur);
      }
      return trees;
    }
    async function listWorktrees2(repoPath, gitBin) {
      let resolved;
      try {
        resolved = resolveRequestedPathForIpc2(repoPath, { purpose: "Worktree list" });
      } catch {
        return [];
      }
      try {
        const out = await runGit2(gitBin, ["worktree", "list", "--porcelain"], resolved);
        return parseWorktrees(out).map((tree, index) => ({
          path: tree.path,
          branch: tree.branch,
          isMain: index === 0,
          detached: tree.detached,
          locked: tree.locked
        }));
      } catch {
        return [];
      }
    }
    function sanitizeBranch(name) {
      return String(name || "").replace(/\s+/g, "-").replace(/[^\w./-]/g, "").replace(/-{2,}/g, "-").replace(/\/{2,}/g, "/").replace(/\.{2,}/g, ".").replace(/^[-./]+|[-./]+$/g, "");
    }
    function slugify(name) {
      const slug = String(name || "").trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40).replace(/-+$/g, "");
      return slug || "work";
    }
    var TRUNK_BRANCHES = ["main", "master"];
    async function gitLine(gitBin, args, cwd) {
      try {
        return (await runGit2(gitBin, args, cwd)).trim();
      } catch {
        return "";
      }
    }
    async function defaultBranch(gitBin, cwd) {
      const remote = (await gitLine(gitBin, ["symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD"], cwd)).replace(
        /^origin\//,
        ""
      );
      if (remote) {
        return remote;
      }
      const configured = await gitLine(gitBin, ["config", "--get", "init.defaultBranch"], cwd);
      if (configured) {
        return configured;
      }
      for (const branch of TRUNK_BRANCHES) {
        if (await gitLine(gitBin, ["show-ref", "--verify", `refs/heads/${branch}`], cwd)) {
          return branch;
        }
      }
      return "";
    }
    async function ensureGitRepo(gitBin, dir) {
      let needsRoot = false;
      try {
        const inside = (await runGit2(gitBin, ["rev-parse", "--is-inside-work-tree"], dir)).trim();
        if (inside !== "true") {
          await runGit2(gitBin, ["init"], dir);
          needsRoot = true;
        } else {
          try {
            await runGit2(gitBin, ["rev-parse", "--verify", "HEAD"], dir);
          } catch {
            needsRoot = true;
          }
        }
      } catch {
        await runGit2(gitBin, ["init"], dir);
        needsRoot = true;
      }
      if (needsRoot) {
        await runGit2(
          gitBin,
          ["-c", "user.email=hermes@localhost", "-c", "user.name=Hermes", "commit", "--allow-empty", "-m", "Initial commit"],
          dir
        );
      }
    }
    async function mainRoot(gitBin, cwd) {
      const list = await listWorktrees2(cwd, gitBin);
      const main = list.find((tree) => tree.isMain);
      return main ? main.path : cwd;
    }
    function uniqueDir(base) {
      let dir = base;
      let n = 1;
      while (fs2.existsSync(dir)) {
        n += 1;
        dir = `${base}-${n}`;
      }
      return dir;
    }
    async function addExistingBranchWorktree(gitBin, root, name) {
      const branch = sanitizeBranch(name);
      if (!branch) {
        throw new Error("Branch name is required.");
      }
      if (branch === await defaultBranch(gitBin, root)) {
        await runGit2(gitBin, ["switch", branch], root);
        return { path: root, branch, repoRoot: root };
      }
      const dir = uniqueDir(path2.join(root, ".worktrees", slugify(branch)));
      await runGit2(gitBin, ["worktree", "add", dir, branch], root);
      return { path: dir, branch, repoRoot: root };
    }
    async function addWorktree2(repoPath, options, gitBin) {
      const resolved = resolveRequestedPathForIpc2(repoPath, { purpose: "Worktree add" });
      await ensureGitRepo(gitBin, resolved);
      const root = await mainRoot(gitBin, resolved);
      const opts = options || {};
      if (opts.existingBranch) {
        return addExistingBranchWorktree(gitBin, root, opts.existingBranch);
      }
      const slug = slugify(opts.name || `work-${Date.now().toString(36)}`);
      const branch = sanitizeBranch(opts.branch) || `hermes/${slug}`;
      const dir = uniqueDir(path2.join(root, ".worktrees", slug));
      const args = ["worktree", "add", "-b", branch, dir];
      if (opts.base) {
        args.push(String(opts.base));
      }
      try {
        await runGit2(gitBin, args, root);
      } catch (err) {
        if (/already exists/i.test(err.stderr || "")) {
          await runGit2(gitBin, ["worktree", "add", dir, branch], root);
        } else {
          throw err;
        }
      }
      return { path: dir, branch, repoRoot: root };
    }
    async function removeWorktree2(repoPath, worktreePath, options, gitBin) {
      const resolvedRepo = resolveRequestedPathForIpc2(repoPath, { purpose: "Worktree remove (repo)" });
      const resolvedTree = resolveRequestedPathForIpc2(worktreePath, { purpose: "Worktree remove (tree)" });
      const root = await mainRoot(gitBin, resolvedRepo);
      const args = ["worktree", "remove"];
      if (options && options.force) {
        args.push("--force");
      }
      args.push(resolvedTree);
      await runGit2(gitBin, args, root);
      return { removed: resolvedTree };
    }
    async function listBranches2(repoPath, gitBin) {
      let resolved;
      try {
        resolved = resolveRequestedPathForIpc2(repoPath, { purpose: "Branch list" });
      } catch {
        return [];
      }
      try {
        const out = await runGit2(
          gitBin,
          ["for-each-ref", "--format=%(refname:short)", "--sort=-committerdate", "refs/heads"],
          resolved
        );
        const trees = await listWorktrees2(resolved, gitBin);
        const pathByBranch = new Map(trees.filter((tree) => tree.branch).map((tree) => [tree.branch, tree.path]));
        const trunk = await defaultBranch(gitBin, resolved);
        return out.split("\n").map((line) => line.trim()).filter(Boolean).map((name) => ({
          name,
          checkedOut: pathByBranch.has(name),
          isDefault: Boolean(trunk && name === trunk),
          worktreePath: pathByBranch.get(name) || null
        }));
      } catch {
        return [];
      }
    }
    async function switchBranch2(repoPath, branch, gitBin) {
      const resolved = resolveRequestedPathForIpc2(repoPath, { purpose: "Branch switch" });
      const target = sanitizeBranch(branch);
      if (!target) {
        throw new Error("Branch name is required.");
      }
      await runGit2(gitBin, ["switch", target], resolved);
      return { branch: target };
    }
    module2.exports = {
      addWorktree: addWorktree2,
      ensureGitRepo,
      listBranches: listBranches2,
      listWorktrees: listWorktrees2,
      parseWorktrees,
      removeWorktree: removeWorktree2,
      sanitizeBranch,
      switchBranch: switchBranch2
    };
  }
});

// ../../node_modules/ms/index.js
var require_ms = __commonJS({
  "../../node_modules/ms/index.js"(exports2, module2) {
    var s = 1e3;
    var m = s * 60;
    var h = m * 60;
    var d = h * 24;
    var w = d * 7;
    var y = d * 365.25;
    module2.exports = function(val, options) {
      options = options || {};
      var type = typeof val;
      if (type === "string" && val.length > 0) {
        return parse(val);
      } else if (type === "number" && isFinite(val)) {
        return options.long ? fmtLong(val) : fmtShort(val);
      }
      throw new Error(
        "val is not a non-empty string or a valid number. val=" + JSON.stringify(val)
      );
    };
    function parse(str) {
      str = String(str);
      if (str.length > 100) {
        return;
      }
      var match = /^(-?(?:\d+)?\.?\d+) *(milliseconds?|msecs?|ms|seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h|days?|d|weeks?|w|years?|yrs?|y)?$/i.exec(
        str
      );
      if (!match) {
        return;
      }
      var n = parseFloat(match[1]);
      var type = (match[2] || "ms").toLowerCase();
      switch (type) {
        case "years":
        case "year":
        case "yrs":
        case "yr":
        case "y":
          return n * y;
        case "weeks":
        case "week":
        case "w":
          return n * w;
        case "days":
        case "day":
        case "d":
          return n * d;
        case "hours":
        case "hour":
        case "hrs":
        case "hr":
        case "h":
          return n * h;
        case "minutes":
        case "minute":
        case "mins":
        case "min":
        case "m":
          return n * m;
        case "seconds":
        case "second":
        case "secs":
        case "sec":
        case "s":
          return n * s;
        case "milliseconds":
        case "millisecond":
        case "msecs":
        case "msec":
        case "ms":
          return n;
        default:
          return void 0;
      }
    }
    function fmtShort(ms) {
      var msAbs = Math.abs(ms);
      if (msAbs >= d) {
        return Math.round(ms / d) + "d";
      }
      if (msAbs >= h) {
        return Math.round(ms / h) + "h";
      }
      if (msAbs >= m) {
        return Math.round(ms / m) + "m";
      }
      if (msAbs >= s) {
        return Math.round(ms / s) + "s";
      }
      return ms + "ms";
    }
    function fmtLong(ms) {
      var msAbs = Math.abs(ms);
      if (msAbs >= d) {
        return plural(ms, msAbs, d, "day");
      }
      if (msAbs >= h) {
        return plural(ms, msAbs, h, "hour");
      }
      if (msAbs >= m) {
        return plural(ms, msAbs, m, "minute");
      }
      if (msAbs >= s) {
        return plural(ms, msAbs, s, "second");
      }
      return ms + " ms";
    }
    function plural(ms, msAbs, n, name) {
      var isPlural = msAbs >= n * 1.5;
      return Math.round(ms / n) + " " + name + (isPlural ? "s" : "");
    }
  }
});

// ../../node_modules/debug/src/common.js
var require_common = __commonJS({
  "../../node_modules/debug/src/common.js"(exports2, module2) {
    function setup(env2) {
      createDebug.debug = createDebug;
      createDebug.default = createDebug;
      createDebug.coerce = coerce;
      createDebug.disable = disable;
      createDebug.enable = enable;
      createDebug.enabled = enabled;
      createDebug.humanize = require_ms();
      createDebug.destroy = destroy;
      Object.keys(env2).forEach((key) => {
        createDebug[key] = env2[key];
      });
      createDebug.names = [];
      createDebug.skips = [];
      createDebug.formatters = {};
      function selectColor(namespace) {
        let hash = 0;
        for (let i = 0; i < namespace.length; i++) {
          hash = (hash << 5) - hash + namespace.charCodeAt(i);
          hash |= 0;
        }
        return createDebug.colors[Math.abs(hash) % createDebug.colors.length];
      }
      createDebug.selectColor = selectColor;
      function createDebug(namespace) {
        let prevTime;
        let enableOverride = null;
        let namespacesCache;
        let enabledCache;
        function debug(...args) {
          if (!debug.enabled) {
            return;
          }
          const self = debug;
          const curr = Number(/* @__PURE__ */ new Date());
          const ms = curr - (prevTime || curr);
          self.diff = ms;
          self.prev = prevTime;
          self.curr = curr;
          prevTime = curr;
          args[0] = createDebug.coerce(args[0]);
          if (typeof args[0] !== "string") {
            args.unshift("%O");
          }
          let index = 0;
          args[0] = args[0].replace(/%([a-zA-Z%])/g, (match, format) => {
            if (match === "%%") {
              return "%";
            }
            index++;
            const formatter = createDebug.formatters[format];
            if (typeof formatter === "function") {
              const val = args[index];
              match = formatter.call(self, val);
              args.splice(index, 1);
              index--;
            }
            return match;
          });
          createDebug.formatArgs.call(self, args);
          const logFn = self.log || createDebug.log;
          logFn.apply(self, args);
        }
        debug.namespace = namespace;
        debug.useColors = createDebug.useColors();
        debug.color = createDebug.selectColor(namespace);
        debug.extend = extend;
        debug.destroy = createDebug.destroy;
        Object.defineProperty(debug, "enabled", {
          enumerable: true,
          configurable: false,
          get: () => {
            if (enableOverride !== null) {
              return enableOverride;
            }
            if (namespacesCache !== createDebug.namespaces) {
              namespacesCache = createDebug.namespaces;
              enabledCache = createDebug.enabled(namespace);
            }
            return enabledCache;
          },
          set: (v) => {
            enableOverride = v;
          }
        });
        if (typeof createDebug.init === "function") {
          createDebug.init(debug);
        }
        return debug;
      }
      function extend(namespace, delimiter) {
        const newDebug = createDebug(this.namespace + (typeof delimiter === "undefined" ? ":" : delimiter) + namespace);
        newDebug.log = this.log;
        return newDebug;
      }
      function enable(namespaces) {
        createDebug.save(namespaces);
        createDebug.namespaces = namespaces;
        createDebug.names = [];
        createDebug.skips = [];
        const split = (typeof namespaces === "string" ? namespaces : "").trim().replace(/\s+/g, ",").split(",").filter(Boolean);
        for (const ns of split) {
          if (ns[0] === "-") {
            createDebug.skips.push(ns.slice(1));
          } else {
            createDebug.names.push(ns);
          }
        }
      }
      function matchesTemplate(search, template) {
        let searchIndex = 0;
        let templateIndex = 0;
        let starIndex = -1;
        let matchIndex = 0;
        while (searchIndex < search.length) {
          if (templateIndex < template.length && (template[templateIndex] === search[searchIndex] || template[templateIndex] === "*")) {
            if (template[templateIndex] === "*") {
              starIndex = templateIndex;
              matchIndex = searchIndex;
              templateIndex++;
            } else {
              searchIndex++;
              templateIndex++;
            }
          } else if (starIndex !== -1) {
            templateIndex = starIndex + 1;
            matchIndex++;
            searchIndex = matchIndex;
          } else {
            return false;
          }
        }
        while (templateIndex < template.length && template[templateIndex] === "*") {
          templateIndex++;
        }
        return templateIndex === template.length;
      }
      function disable() {
        const namespaces = [
          ...createDebug.names,
          ...createDebug.skips.map((namespace) => "-" + namespace)
        ].join(",");
        createDebug.enable("");
        return namespaces;
      }
      function enabled(name) {
        for (const skip of createDebug.skips) {
          if (matchesTemplate(name, skip)) {
            return false;
          }
        }
        for (const ns of createDebug.names) {
          if (matchesTemplate(name, ns)) {
            return true;
          }
        }
        return false;
      }
      function coerce(val) {
        if (val instanceof Error) {
          return val.stack || val.message;
        }
        return val;
      }
      function destroy() {
        console.warn("Instance method `debug.destroy()` is deprecated and no longer does anything. It will be removed in the next major version of `debug`.");
      }
      createDebug.enable(createDebug.load());
      return createDebug;
    }
    module2.exports = setup;
  }
});

// ../../node_modules/debug/src/browser.js
var require_browser = __commonJS({
  "../../node_modules/debug/src/browser.js"(exports2, module2) {
    exports2.formatArgs = formatArgs;
    exports2.save = save;
    exports2.load = load;
    exports2.useColors = useColors;
    exports2.storage = localstorage();
    exports2.destroy = /* @__PURE__ */ (() => {
      let warned = false;
      return () => {
        if (!warned) {
          warned = true;
          console.warn("Instance method `debug.destroy()` is deprecated and no longer does anything. It will be removed in the next major version of `debug`.");
        }
      };
    })();
    exports2.colors = [
      "#0000CC",
      "#0000FF",
      "#0033CC",
      "#0033FF",
      "#0066CC",
      "#0066FF",
      "#0099CC",
      "#0099FF",
      "#00CC00",
      "#00CC33",
      "#00CC66",
      "#00CC99",
      "#00CCCC",
      "#00CCFF",
      "#3300CC",
      "#3300FF",
      "#3333CC",
      "#3333FF",
      "#3366CC",
      "#3366FF",
      "#3399CC",
      "#3399FF",
      "#33CC00",
      "#33CC33",
      "#33CC66",
      "#33CC99",
      "#33CCCC",
      "#33CCFF",
      "#6600CC",
      "#6600FF",
      "#6633CC",
      "#6633FF",
      "#66CC00",
      "#66CC33",
      "#9900CC",
      "#9900FF",
      "#9933CC",
      "#9933FF",
      "#99CC00",
      "#99CC33",
      "#CC0000",
      "#CC0033",
      "#CC0066",
      "#CC0099",
      "#CC00CC",
      "#CC00FF",
      "#CC3300",
      "#CC3333",
      "#CC3366",
      "#CC3399",
      "#CC33CC",
      "#CC33FF",
      "#CC6600",
      "#CC6633",
      "#CC9900",
      "#CC9933",
      "#CCCC00",
      "#CCCC33",
      "#FF0000",
      "#FF0033",
      "#FF0066",
      "#FF0099",
      "#FF00CC",
      "#FF00FF",
      "#FF3300",
      "#FF3333",
      "#FF3366",
      "#FF3399",
      "#FF33CC",
      "#FF33FF",
      "#FF6600",
      "#FF6633",
      "#FF9900",
      "#FF9933",
      "#FFCC00",
      "#FFCC33"
    ];
    function useColors() {
      if (typeof window !== "undefined" && window.process && (window.process.type === "renderer" || window.process.__nwjs)) {
        return true;
      }
      if (typeof navigator !== "undefined" && navigator.userAgent && navigator.userAgent.toLowerCase().match(/(edge|trident)\/(\d+)/)) {
        return false;
      }
      let m;
      return typeof document !== "undefined" && document.documentElement && document.documentElement.style && document.documentElement.style.WebkitAppearance || // Is firebug? http://stackoverflow.com/a/398120/376773
      typeof window !== "undefined" && window.console && (window.console.firebug || window.console.exception && window.console.table) || // Is firefox >= v31?
      // https://developer.mozilla.org/en-US/docs/Tools/Web_Console#Styling_messages
      typeof navigator !== "undefined" && navigator.userAgent && (m = navigator.userAgent.toLowerCase().match(/firefox\/(\d+)/)) && parseInt(m[1], 10) >= 31 || // Double check webkit in userAgent just in case we are in a worker
      typeof navigator !== "undefined" && navigator.userAgent && navigator.userAgent.toLowerCase().match(/applewebkit\/(\d+)/);
    }
    function formatArgs(args) {
      args[0] = (this.useColors ? "%c" : "") + this.namespace + (this.useColors ? " %c" : " ") + args[0] + (this.useColors ? "%c " : " ") + "+" + module2.exports.humanize(this.diff);
      if (!this.useColors) {
        return;
      }
      const c = "color: " + this.color;
      args.splice(1, 0, c, "color: inherit");
      let index = 0;
      let lastC = 0;
      args[0].replace(/%[a-zA-Z%]/g, (match) => {
        if (match === "%%") {
          return;
        }
        index++;
        if (match === "%c") {
          lastC = index;
        }
      });
      args.splice(lastC, 0, c);
    }
    exports2.log = console.debug || console.log || (() => {
    });
    function save(namespaces) {
      try {
        if (namespaces) {
          exports2.storage.setItem("debug", namespaces);
        } else {
          exports2.storage.removeItem("debug");
        }
      } catch (error) {
      }
    }
    function load() {
      let r;
      try {
        r = exports2.storage.getItem("debug") || exports2.storage.getItem("DEBUG");
      } catch (error) {
      }
      if (!r && typeof process !== "undefined" && "env" in process) {
        r = process.env.DEBUG;
      }
      return r;
    }
    function localstorage() {
      try {
        return localStorage;
      } catch (error) {
      }
    }
    module2.exports = require_common()(exports2);
    var { formatters } = module2.exports;
    formatters.j = function(v) {
      try {
        return JSON.stringify(v);
      } catch (error) {
        return "[UnexpectedJSONParseError]: " + error.message;
      }
    };
  }
});

// ../../node_modules/supports-color/index.js
var supports_color_exports = {};
__export(supports_color_exports, {
  createSupportsColor: () => createSupportsColor,
  default: () => supports_color_default
});
function hasFlag(flag, argv = globalThis.Deno ? globalThis.Deno.args : import_node_process.default.argv) {
  const prefix = flag.startsWith("-") ? "" : flag.length === 1 ? "-" : "--";
  const position = argv.indexOf(prefix + flag);
  const terminatorPosition = argv.indexOf("--");
  return position !== -1 && (terminatorPosition === -1 || position < terminatorPosition);
}
function envForceColor() {
  if (!("FORCE_COLOR" in env)) {
    return;
  }
  if (env.FORCE_COLOR === "true") {
    return 1;
  }
  if (env.FORCE_COLOR === "false") {
    return 0;
  }
  if (env.FORCE_COLOR.length === 0) {
    return 1;
  }
  const level = Math.min(Number.parseInt(env.FORCE_COLOR, 10), 3);
  if (![0, 1, 2, 3].includes(level)) {
    return;
  }
  return level;
}
function translateLevel(level) {
  if (level === 0) {
    return false;
  }
  return {
    level,
    hasBasic: true,
    has256: level >= 2,
    has16m: level >= 3
  };
}
function _supportsColor(haveStream, { streamIsTTY, sniffFlags = true } = {}) {
  const noFlagForceColor = envForceColor();
  if (noFlagForceColor !== void 0) {
    flagForceColor = noFlagForceColor;
  }
  const forceColor = sniffFlags ? flagForceColor : noFlagForceColor;
  if (forceColor === 0) {
    return 0;
  }
  if (sniffFlags) {
    if (hasFlag("color=16m") || hasFlag("color=full") || hasFlag("color=truecolor")) {
      return 3;
    }
    if (hasFlag("color=256")) {
      return 2;
    }
  }
  if ("TF_BUILD" in env && "AGENT_NAME" in env) {
    return 1;
  }
  if (haveStream && !streamIsTTY && forceColor === void 0) {
    return 0;
  }
  const min = forceColor || 0;
  if (env.TERM === "dumb") {
    return min;
  }
  if (import_node_process.default.platform === "win32") {
    const osRelease = import_node_os.default.release().split(".");
    if (Number(osRelease[0]) >= 10 && Number(osRelease[2]) >= 10586) {
      return Number(osRelease[2]) >= 14931 ? 3 : 2;
    }
    return 1;
  }
  if ("CI" in env) {
    if (["GITHUB_ACTIONS", "GITEA_ACTIONS", "CIRCLECI"].some((key) => key in env)) {
      return 3;
    }
    if (["TRAVIS", "APPVEYOR", "GITLAB_CI", "BUILDKITE", "DRONE"].some((sign) => sign in env) || env.CI_NAME === "codeship") {
      return 1;
    }
    return min;
  }
  if ("TEAMCITY_VERSION" in env) {
    return /^(9\.(0*[1-9]\d*)\.|\d{2,}\.)/.test(env.TEAMCITY_VERSION) ? 1 : 0;
  }
  if (env.COLORTERM === "truecolor") {
    return 3;
  }
  if (env.TERM === "xterm-kitty") {
    return 3;
  }
  if (env.TERM === "xterm-ghostty") {
    return 3;
  }
  if (env.TERM === "wezterm") {
    return 3;
  }
  if ("TERM_PROGRAM" in env) {
    const version = Number.parseInt((env.TERM_PROGRAM_VERSION || "").split(".")[0], 10);
    switch (env.TERM_PROGRAM) {
      case "iTerm.app": {
        return version >= 3 ? 3 : 2;
      }
      case "Apple_Terminal": {
        return 2;
      }
    }
  }
  if (/-256(color)?$/i.test(env.TERM)) {
    return 2;
  }
  if (/^screen|^xterm|^vt100|^vt220|^rxvt|color|ansi|cygwin|linux/i.test(env.TERM)) {
    return 1;
  }
  if ("COLORTERM" in env) {
    return 1;
  }
  return min;
}
function createSupportsColor(stream, options = {}) {
  const level = _supportsColor(stream, {
    streamIsTTY: stream && stream.isTTY,
    ...options
  });
  return translateLevel(level);
}
var import_node_process, import_node_os, import_node_tty, env, flagForceColor, supportsColor, supports_color_default;
var init_supports_color = __esm({
  "../../node_modules/supports-color/index.js"() {
    import_node_process = __toESM(require("node:process"), 1);
    import_node_os = __toESM(require("node:os"), 1);
    import_node_tty = __toESM(require("node:tty"), 1);
    ({ env } = import_node_process.default);
    if (hasFlag("no-color") || hasFlag("no-colors") || hasFlag("color=false") || hasFlag("color=never")) {
      flagForceColor = 0;
    } else if (hasFlag("color") || hasFlag("colors") || hasFlag("color=true") || hasFlag("color=always")) {
      flagForceColor = 1;
    }
    supportsColor = {
      stdout: createSupportsColor({ isTTY: import_node_tty.default.isatty(1) }),
      stderr: createSupportsColor({ isTTY: import_node_tty.default.isatty(2) })
    };
    supports_color_default = supportsColor;
  }
});

// ../../node_modules/debug/src/node.js
var require_node = __commonJS({
  "../../node_modules/debug/src/node.js"(exports2, module2) {
    var tty2 = require("tty");
    var util = require("util");
    exports2.init = init;
    exports2.log = log;
    exports2.formatArgs = formatArgs;
    exports2.save = save;
    exports2.load = load;
    exports2.useColors = useColors;
    exports2.destroy = util.deprecate(
      () => {
      },
      "Instance method `debug.destroy()` is deprecated and no longer does anything. It will be removed in the next major version of `debug`."
    );
    exports2.colors = [6, 2, 3, 4, 5, 1];
    try {
      const supportsColor2 = (init_supports_color(), __toCommonJS(supports_color_exports));
      if (supportsColor2 && (supportsColor2.stderr || supportsColor2).level >= 2) {
        exports2.colors = [
          20,
          21,
          26,
          27,
          32,
          33,
          38,
          39,
          40,
          41,
          42,
          43,
          44,
          45,
          56,
          57,
          62,
          63,
          68,
          69,
          74,
          75,
          76,
          77,
          78,
          79,
          80,
          81,
          92,
          93,
          98,
          99,
          112,
          113,
          128,
          129,
          134,
          135,
          148,
          149,
          160,
          161,
          162,
          163,
          164,
          165,
          166,
          167,
          168,
          169,
          170,
          171,
          172,
          173,
          178,
          179,
          184,
          185,
          196,
          197,
          198,
          199,
          200,
          201,
          202,
          203,
          204,
          205,
          206,
          207,
          208,
          209,
          214,
          215,
          220,
          221
        ];
      }
    } catch (error) {
    }
    exports2.inspectOpts = Object.keys(process.env).filter((key) => {
      return /^debug_/i.test(key);
    }).reduce((obj, key) => {
      const prop = key.substring(6).toLowerCase().replace(/_([a-z])/g, (_, k) => {
        return k.toUpperCase();
      });
      let val = process.env[key];
      if (/^(yes|on|true|enabled)$/i.test(val)) {
        val = true;
      } else if (/^(no|off|false|disabled)$/i.test(val)) {
        val = false;
      } else if (val === "null") {
        val = null;
      } else {
        val = Number(val);
      }
      obj[prop] = val;
      return obj;
    }, {});
    function useColors() {
      return "colors" in exports2.inspectOpts ? Boolean(exports2.inspectOpts.colors) : tty2.isatty(process.stderr.fd);
    }
    function formatArgs(args) {
      const { namespace: name, useColors: useColors2 } = this;
      if (useColors2) {
        const c = this.color;
        const colorCode = "\x1B[3" + (c < 8 ? c : "8;5;" + c);
        const prefix = `  ${colorCode};1m${name} \x1B[0m`;
        args[0] = prefix + args[0].split("\n").join("\n" + prefix);
        args.push(colorCode + "m+" + module2.exports.humanize(this.diff) + "\x1B[0m");
      } else {
        args[0] = getDate() + name + " " + args[0];
      }
    }
    function getDate() {
      if (exports2.inspectOpts.hideDate) {
        return "";
      }
      return (/* @__PURE__ */ new Date()).toISOString() + " ";
    }
    function log(...args) {
      return process.stderr.write(util.formatWithOptions(exports2.inspectOpts, ...args) + "\n");
    }
    function save(namespaces) {
      if (namespaces) {
        process.env.DEBUG = namespaces;
      } else {
        delete process.env.DEBUG;
      }
    }
    function load() {
      return process.env.DEBUG;
    }
    function init(debug) {
      debug.inspectOpts = {};
      const keys = Object.keys(exports2.inspectOpts);
      for (let i = 0; i < keys.length; i++) {
        debug.inspectOpts[keys[i]] = exports2.inspectOpts[keys[i]];
      }
    }
    module2.exports = require_common()(exports2);
    var { formatters } = module2.exports;
    formatters.o = function(v) {
      this.inspectOpts.colors = this.useColors;
      return util.inspect(v, this.inspectOpts).split("\n").map((str) => str.trim()).join(" ");
    };
    formatters.O = function(v) {
      this.inspectOpts.colors = this.useColors;
      return util.inspect(v, this.inspectOpts);
    };
  }
});

// ../../node_modules/debug/src/index.js
var require_src = __commonJS({
  "../../node_modules/debug/src/index.js"(exports2, module2) {
    if (typeof process === "undefined" || process.type === "renderer" || process.browser === true || process.__nwjs) {
      module2.exports = require_browser();
    } else {
      module2.exports = require_node();
    }
  }
});

// ../../node_modules/@kwsites/file-exists/dist/src/index.js
var require_src2 = __commonJS({
  "../../node_modules/@kwsites/file-exists/dist/src/index.js"(exports2) {
    "use strict";
    var __importDefault = exports2 && exports2.__importDefault || function(mod) {
      return mod && mod.__esModule ? mod : { "default": mod };
    };
    Object.defineProperty(exports2, "__esModule", { value: true });
    var fs_1 = require("fs");
    var debug_1 = __importDefault(require_src());
    var log = debug_1.default("@kwsites/file-exists");
    function check(path2, isFile, isDirectory) {
      log(`checking %s`, path2);
      try {
        const stat = fs_1.statSync(path2);
        if (stat.isFile() && isFile) {
          log(`[OK] path represents a file`);
          return true;
        }
        if (stat.isDirectory() && isDirectory) {
          log(`[OK] path represents a directory`);
          return true;
        }
        log(`[FAIL] path represents something other than a file or directory`);
        return false;
      } catch (e) {
        if (e.code === "ENOENT") {
          log(`[FAIL] path is not accessible: %o`, e);
          return false;
        }
        log(`[FATAL] %o`, e);
        throw e;
      }
    }
    function exists(path2, type = exports2.READABLE) {
      return check(path2, (type & exports2.FILE) > 0, (type & exports2.FOLDER) > 0);
    }
    exports2.exists = exists;
    exports2.FILE = 1;
    exports2.FOLDER = 2;
    exports2.READABLE = exports2.FILE + exports2.FOLDER;
  }
});

// ../../node_modules/@kwsites/file-exists/dist/index.js
var require_dist = __commonJS({
  "../../node_modules/@kwsites/file-exists/dist/index.js"(exports2) {
    "use strict";
    function __export2(m) {
      for (var p in m) if (!exports2.hasOwnProperty(p)) exports2[p] = m[p];
    }
    Object.defineProperty(exports2, "__esModule", { value: true });
    __export2(require_src2());
  }
});

// ../../node_modules/@simple-git/args-pathspec/dist/index.cjs
var require_dist2 = __commonJS({
  "../../node_modules/@simple-git/args-pathspec/dist/index.cjs"(exports2) {
    "use strict";
    Object.defineProperty(exports2, Symbol.toStringTag, { value: "Module" });
    var e = /* @__PURE__ */ new WeakMap();
    function c(...t) {
      const n = new String(t);
      return e.set(n, t), n;
    }
    function o(t) {
      return t instanceof String && e.has(t);
    }
    function r(t) {
      return e.get(t) ?? [];
    }
    exports2.isPathSpec = o;
    exports2.pathspec = c;
    exports2.toPaths = r;
  }
});

// ../../node_modules/@simple-git/argv-parser/dist/index.cjs
var require_dist3 = __commonJS({
  "../../node_modules/@simple-git/argv-parser/dist/index.cjs"(exports2) {
    "use strict";
    Object.defineProperty(exports2, Symbol.toStringTag, { value: "Module" });
    var h = require_dist2();
    function* v(e, t) {
      const n = t === "global";
      for (const o of e) o.isGlobal === n && (yield o);
    }
    var S = /* @__PURE__ */ new Set(["--add", "--edit", "--remove-section", "--rename-section", "--replace-all", "--unset", "--unset-all", "-e"]);
    var P = /* @__PURE__ */ new Set(["--get", "--get-all", "--get-color", "--get-colorbool", "--get-regexp", "--get-urlmatch", "--list", "-l"]);
    var E = /* @__PURE__ */ new Set(["edit", "remove-section", "rename-section", "set", "unset"]);
    var A = /* @__PURE__ */ new Set(["get", "get-color", "get-colorbool", "list"]);
    function F(e, t) {
      for (const { name: o } of v(e, "task")) {
        if (S.has(o)) return p(true, t);
        if (P.has(o)) return p(false, t);
      }
      const n = t.at(0)?.toLowerCase();
      return n === void 0 ? null : E.has(n) ? p(true, t.slice(1)) : A.has(n) ? p(false, t.slice(1)) : t.length === 1 ? p(false, t) : p(true, t);
    }
    function p(e = false, t = []) {
      const n = t.at(0)?.toLowerCase();
      return n === void 0 ? null : { isWrite: e, isRead: !e, key: n, value: t.at(1) };
    }
    function M(e, t) {
      return t.isWrite && t.value !== void 0 ? { key: t.key, value: t.value, scope: e } : { key: t.key, scope: e };
    }
    function N(e) {
      const t = e?.indexOf("=") || -1;
      return !e || t < 0 ? null : { key: e.slice(0, t).trim().toLowerCase(), value: e.slice(t + 1) };
    }
    function O(e) {
      for (const { name: t } of v(e, "task")) switch (t) {
        case "--global":
          return "global";
        case "--system":
          return "system";
        case "--worktree":
          return "worktree";
        case "--local":
          return "local";
        case "--file":
        case "-f":
          return "file";
      }
      return "local";
    }
    function G({ name: e }) {
      if (e === "-c" || e === "--config") return "inline";
      if (e === "--config-env") return "env";
    }
    function* L(e) {
      for (const t of e) {
        const n = G(t), o = n && N(t.value);
        o && (yield { ...o, scope: n });
      }
    }
    function $(e, t, n) {
      const o = { read: [], write: [...L(t)] };
      return e === "config" && D(o, O(t), F(t, n)), o;
    }
    function D(e, t, n) {
      if (n === null) return;
      const o = M(t, n);
      n.isWrite ? e.write.push(o) : e.read.push(o);
    }
    var U = { short: /* @__PURE__ */ new Map([["c", true]]) };
    var T = { short: new Map([["C", true], ["P", false], ["h", false], ["p", false], ["v", false], ...U.short.entries()]), long: /* @__PURE__ */ new Set(["attr-source", "config-env", "exec-path", "git-dir", "list-cmds", "namespace", "super-prefix", "work-tree"]) };
    var R = { clone: { short: /* @__PURE__ */ new Map([["b", true], ["j", true], ["l", false], ["n", false], ["o", true], ["q", false], ["s", false], ["u", true]]), long: /* @__PURE__ */ new Set(["branch", "config", "jobs", "origin", "upload-pack", "u", "template"]) }, commit: { short: /* @__PURE__ */ new Map([["C", true], ["F", true], ["c", true], ["m", true], ["t", true]]), long: /* @__PURE__ */ new Set(["file", "message", "reedit-message", "reuse-message", "template"]) }, config: { short: /* @__PURE__ */ new Map([["e", false], ["f", true], ["l", false]]), long: /* @__PURE__ */ new Set(["blob", "comment", "default", "file", "type", "value"]) }, fetch: { short: /* @__PURE__ */ new Map(), long: /* @__PURE__ */ new Set(["upload-pack"]) }, init: { short: /* @__PURE__ */ new Map(), long: /* @__PURE__ */ new Set(["template"]) }, pull: { short: /* @__PURE__ */ new Map(), long: /* @__PURE__ */ new Set(["upload-pack"]) }, push: { short: /* @__PURE__ */ new Map(), long: /* @__PURE__ */ new Set(["exec", "receive-pack"]) } };
    var I = { short: /* @__PURE__ */ new Map(), long: /* @__PURE__ */ new Set() };
    function j(e) {
      const t = R[e ?? ""] ?? I;
      return { short: new Map([...U.short.entries(), ...t.short.entries()]), long: t.long };
    }
    function b(e, t = T) {
      if (e.startsWith("--")) {
        const n = e.indexOf("=");
        if (n > 2) return [{ name: e.slice(0, n), value: e.slice(n + 1), needsNext: false }];
        const o = e.slice(2);
        return [{ name: e, needsNext: t.long.has(o) }];
      }
      if (e.length === 2) {
        const n = e.charAt(1), o = t.short.get(n);
        return [{ name: e, needsNext: o === true }];
      }
      return W(e, t.short);
    }
    function W(e, t) {
      const n = e.slice(1).split(""), o = [];
      for (let s = 0; s < n.length; s++) {
        const r = n[s], a = t.get(r);
        if (a === void 0) return [{ name: e, needsNext: false }];
        if (a) {
          const l = n.slice(s + 1).join("");
          if (l && ![...l].every((m) => t.has(m))) return o.push({ name: `-${r}`, value: l, needsNext: false }), o;
        }
        o.push({ name: `-${r}`, needsNext: a });
      }
      return o;
    }
    function B(e, t = []) {
      let n = 0;
      for (; n < e.length; ) {
        const o = String(e[n]);
        if (!o.startsWith("-") || o.length < 2) break;
        const s = b(o);
        let r = n + 1;
        for (const a of s) {
          const l = { name: a.name, value: a.value, absorbedNext: false, isGlobal: true };
          a.needsNext && l.value === void 0 && r < e.length && (l.value = String(e[r]), l.absorbedNext = true, r++), t.push(l);
        }
        n = r;
      }
      return { flags: t, taskIndex: n };
    }
    function q(e, t, n = []) {
      const o = j(t), s = [], r = [];
      let a = 0;
      for (; a < e.length; ) {
        const l = e[a];
        if (h.isPathSpec(l)) {
          r.push(...h.toPaths(l)), a++;
          continue;
        }
        const f = String(l);
        if (f === "--") {
          for (let g = a + 1; g < e.length; g++) {
            const u = e[g];
            h.isPathSpec(u) ? r.push(...h.toPaths(u)) : r.push(String(u));
          }
          break;
        }
        if (!f.startsWith("-") || f.length < 2) {
          s.push(f), a++;
          continue;
        }
        const m = b(f, o);
        let d = a + 1;
        for (const g of m) {
          const u = { name: g.name, value: g.value, absorbedNext: false, isGlobal: false };
          g.needsNext && u.value === void 0 && d < e.length && !h.isPathSpec(e[d]) && (u.value = String(e[d]), u.absorbedNext = true, d++), n.push(u);
        }
        a = d;
      }
      return { flags: n, positionals: s, pathspecs: r };
    }
    function* V({ write: e }) {
      for (const t of e) for (const n of K) {
        const o = n(t.key);
        o && (yield o);
      }
    }
    function c(e, t, n = String(e)) {
      const o = typeof e == "string" ? new RegExp(`\\s*${e.toLowerCase()}`) : e;
      return function(r) {
        if (o.test(r)) return { category: t, message: `Configuring ${n} is not permitted without enabling ${t}` };
      };
    }
    function i(e, t) {
      const n = new RegExp(`\\s*${e.toLowerCase().replace(/\./g, "(..+)?.")}`);
      return c(n, t, e);
    }
    var K = [c("alias", "allowUnsafeAlias"), c("core.askPass", "allowUnsafeAskPass"), c("core.editor", "allowUnsafeEditor"), c("core.fsmonitor", "allowUnsafeFsMonitor"), c("core.gitProxy", "allowUnsafeGitProxy"), c("core.hooksPath", "allowUnsafeHooksPath"), c("core.pager", "allowUnsafePager"), c("core.sshCommand", "allowUnsafeSshCommand"), i("credential.helper", "allowUnsafeCredentialHelper"), i("diff.command", "allowUnsafeDiffExternal"), c("diff.external", "allowUnsafeDiffExternal"), i("diff.textconv", "allowUnsafeDiffTextConv"), i("filter.clean", "allowUnsafeFilter"), i("filter.smudge", "allowUnsafeFilter"), i("gpg.program", "allowUnsafeGpgProgram"), c("init.templateDir", "allowUnsafeTemplateDir"), i("merge.driver", "allowUnsafeMergeDriver"), i("mergetool.path", "allowUnsafeMergeDriver"), i("mergetool.cmd", "allowUnsafeMergeDriver"), i("protocol.allow", "allowUnsafeProtocolOverride"), i("remote.receivepack", "allowUnsafePack"), i("remote.uploadpack", "allowUnsafePack"), c("sequence.editor", "allowUnsafeEditor")];
    function* H(e, t) {
      for (const n of t) for (const o of Y) {
        const s = o(e, n.name);
        s && (yield s);
      }
    }
    function w(e, t, n, o = String(t)) {
      const s = typeof t == "string" ? new RegExp(`\\s*${t.toLowerCase()}`) : t, r = `Use of ${e ? `${e} with option ` : ""}${o} is not permitted without enabling ${n}`;
      return function(l, f) {
        if ((!e || l === e) && s.test(f)) return { category: n, message: r };
      };
    }
    var Y = [w(null, /--(upload|receive)-pack/, "allowUnsafePack", "--upload-pack or --receive-pack"), w("clone", /^-\w*u/, "allowUnsafePack"), w("clone", "--u", "allowUnsafePack"), w("push", "--exec", "allowUnsafePack"), w(null, "--template", "allowUnsafeTemplateDir")];
    function C(e, t, n) {
      return [...H(e, t), ...V(n)];
    }
    function x(...e) {
      const { flags: t, taskIndex: n } = B(e), o = n < e.length ? String(e[n]).toLowerCase() : null, s = o !== null ? e.slice(n + 1) : [], { positionals: r, pathspecs: a } = q(s, o, t), l = $(o, t, r);
      return { task: o, flags: t.map(J), paths: a, config: l, vulnerabilities: z(C(o, t, l)) };
    }
    function z(e) {
      return Object.defineProperty(e, "vulnerabilities", { value: e });
    }
    function J({ value: e, name: t }) {
      return e !== void 0 ? { name: t, value: e } : { name: t };
    }
    var y = { editor: "allowUnsafeEditor", git_askpass: "allowUnsafeAskPass", git_config_global: "allowUnsafeConfigPaths", git_config_system: "allowUnsafeConfigPaths", git_config_count: "allowUnsafeConfigEnvCount", git_config: "allowUnsafeConfigPaths", git_editor: "allowUnsafeEditor", git_exec_path: "allowUnsafeConfigPaths", git_external_diff: "allowUnsafeDiffExternal", git_pager: "allowUnsafePager", git_proxy_command: "allowUnsafeGitProxy", git_template_dir: "allowUnsafeTemplateDir", git_sequence_editor: "allowUnsafeEditor", git_ssh: "allowUnsafeSshCommand", git_ssh_command: "allowUnsafeSshCommand", pager: "allowUnsafePager", prefix: "allowUnsafeConfigPaths", ssh_askpass: "allowUnsafeAskPass" };
    function* Q(e) {
      const t = parseInt(e.git_config_count ?? "0", 10);
      for (let n = 0; n < t; n++) {
        const o = e[`git_config_key_${n}`], s = e[`git_config_value_${n}`];
        o !== void 0 && (yield { key: o.toLowerCase().trim(), value: s, scope: "env" });
      }
    }
    function* X(e) {
      for (const t of Object.keys(e)) if (k(t)) {
        const n = y[t];
        yield { category: n, message: `Use of "${t.toUpperCase()}" is not permitted without enabling ${n}` };
      }
    }
    function k(e) {
      return Object.hasOwn(y, e);
    }
    function Z(e) {
      const t = {};
      for (const [n, o] of Object.entries(e)) {
        const s = n.toLowerCase().trim();
        (k(s) || s.startsWith("git")) && (t[s] = String(o));
      }
      return t;
    }
    function _(e) {
      const t = Z(e), n = { read: [], write: [...Q(t)] }, o = [...X(t), ...C(null, [], n)];
      return { config: n, vulnerabilities: o };
    }
    function ee(e, t) {
      return [...x(...e).vulnerabilities, ..._(t).vulnerabilities];
    }
    exports2.parseArgv = x;
    exports2.parseEnv = _;
    exports2.vulnerabilityCheck = ee;
  }
});

// ../../node_modules/@kwsites/promise-deferred/dist/index.js
var require_dist4 = __commonJS({
  "../../node_modules/@kwsites/promise-deferred/dist/index.js"(exports2) {
    "use strict";
    Object.defineProperty(exports2, "__esModule", { value: true });
    exports2.createDeferred = exports2.deferred = void 0;
    function deferred() {
      let done;
      let fail;
      let status = "pending";
      const promise = new Promise((_done, _fail) => {
        done = _done;
        fail = _fail;
      });
      return {
        promise,
        done(result) {
          if (status === "pending") {
            status = "resolved";
            done(result);
          }
        },
        fail(error) {
          if (status === "pending") {
            status = "rejected";
            fail(error);
          }
        },
        get fulfilled() {
          return status !== "pending";
        },
        get status() {
          return status;
        }
      };
    }
    exports2.deferred = deferred;
    exports2.createDeferred = deferred;
    exports2.default = deferred;
  }
});

// ../../node_modules/simple-git/dist/cjs/index.js
var require_cjs = __commonJS({
  "../../node_modules/simple-git/dist/cjs/index.js"(exports2, module2) {
    "use strict";
    var __create2 = Object.create;
    var __defProp2 = Object.defineProperty;
    var __getOwnPropDesc2 = Object.getOwnPropertyDescriptor;
    var __getOwnPropNames2 = Object.getOwnPropertyNames;
    var __getProtoOf2 = Object.getPrototypeOf;
    var __hasOwnProp2 = Object.prototype.hasOwnProperty;
    var __esm2 = (fn, res) => function __init() {
      return fn && (res = (0, fn[__getOwnPropNames2(fn)[0]])(fn = 0)), res;
    };
    var __commonJS2 = (cb, mod) => function __require() {
      return mod || (0, cb[__getOwnPropNames2(cb)[0]])((mod = { exports: {} }).exports, mod), mod.exports;
    };
    var __export2 = (target, all) => {
      for (var name in all)
        __defProp2(target, name, { get: all[name], enumerable: true });
    };
    var __copyProps2 = (to, from, except, desc) => {
      if (from && typeof from === "object" || typeof from === "function") {
        for (let key of __getOwnPropNames2(from))
          if (!__hasOwnProp2.call(to, key) && key !== except)
            __defProp2(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc2(from, key)) || desc.enumerable });
      }
      return to;
    };
    var __toESM2 = (mod, isNodeMode, target) => (target = mod != null ? __create2(__getProtoOf2(mod)) : {}, __copyProps2(
      // If the importer is in node compatibility mode or this is not an ESM
      // file that has been converted to a CommonJS file using a Babel-
      // compatible transform (i.e. "__esModule" has not been set), then set
      // "default" to the CommonJS "module.exports" for node compatibility.
      isNodeMode || !mod || !mod.__esModule ? __defProp2(target, "default", { value: mod, enumerable: true }) : target,
      mod
    ));
    var __toCommonJS2 = (mod) => __copyProps2(__defProp2({}, "__esModule", { value: true }), mod);
    var GitError;
    var init_git_error = __esm2({
      "src/lib/errors/git-error.ts"() {
        "use strict";
        GitError = class extends Error {
          constructor(task, message) {
            super(message);
            this.task = task;
            Object.setPrototypeOf(this, new.target.prototype);
          }
        };
      }
    });
    var GitResponseError;
    var init_git_response_error = __esm2({
      "src/lib/errors/git-response-error.ts"() {
        "use strict";
        init_git_error();
        GitResponseError = class extends GitError {
          constructor(git, message) {
            super(void 0, message || String(git));
            this.git = git;
          }
        };
      }
    });
    var GitConstructError;
    var init_git_construct_error = __esm2({
      "src/lib/errors/git-construct-error.ts"() {
        "use strict";
        init_git_error();
        GitConstructError = class extends GitError {
          constructor(config, message) {
            super(void 0, message);
            this.config = config;
          }
        };
      }
    });
    var GitPluginError;
    var init_git_plugin_error = __esm2({
      "src/lib/errors/git-plugin-error.ts"() {
        "use strict";
        init_git_error();
        GitPluginError = class extends GitError {
          constructor(task, plugin, message) {
            super(task, message);
            this.task = task;
            this.plugin = plugin;
            Object.setPrototypeOf(this, new.target.prototype);
          }
        };
      }
    });
    var TaskConfigurationError;
    var init_task_configuration_error = __esm2({
      "src/lib/errors/task-configuration-error.ts"() {
        "use strict";
        init_git_error();
        TaskConfigurationError = class extends GitError {
          constructor(message) {
            super(void 0, message);
          }
        };
      }
    });
    function asFunction(source) {
      if (typeof source !== "function") {
        return NOOP;
      }
      return source;
    }
    function isUserFunction(source) {
      return typeof source === "function" && source !== NOOP;
    }
    function splitOn(input, char) {
      const index = input.indexOf(char);
      if (index <= 0) {
        return [input, ""];
      }
      return [input.substr(0, index), input.substr(index + 1)];
    }
    function first(input, offset = 0) {
      return isArrayLike(input) && input.length > offset ? input[offset] : void 0;
    }
    function last(input, offset = 0) {
      if (isArrayLike(input) && input.length > offset) {
        return input[input.length - 1 - offset];
      }
    }
    function isArrayLike(input) {
      return filterHasLength(input);
    }
    function toLinesWithContent(input = "", trimmed2 = true, separator = "\n") {
      return input.split(separator).reduce((output, line) => {
        const lineContent = trimmed2 ? line.trim() : line;
        if (lineContent) {
          output.push(lineContent);
        }
        return output;
      }, []);
    }
    function forEachLineWithContent(input, callback) {
      return toLinesWithContent(input, true).map((line) => callback(line));
    }
    function folderExists(path2) {
      return (0, import_file_exists.exists)(path2, import_file_exists.FOLDER);
    }
    function append(target, item) {
      if (Array.isArray(target)) {
        if (!target.includes(item)) {
          target.push(item);
        }
      } else {
        target.add(item);
      }
      return item;
    }
    function including(target, item) {
      if (Array.isArray(target) && !target.includes(item)) {
        target.push(item);
      }
      return target;
    }
    function remove(target, item) {
      if (Array.isArray(target)) {
        const index = target.indexOf(item);
        if (index >= 0) {
          target.splice(index, 1);
        }
      } else {
        target.delete(item);
      }
      return item;
    }
    function asArray(source) {
      return Array.isArray(source) ? source : [source];
    }
    function asCamelCase(str) {
      return str.replace(/[\s-]+(.)/g, (_all, chr) => {
        return chr.toUpperCase();
      });
    }
    function asStringArray(source) {
      return asArray(source).map((item) => {
        return item instanceof String ? item : String(item);
      });
    }
    function asNumber(source, onNaN = 0) {
      if (source == null) {
        return onNaN;
      }
      const num = parseInt(source, 10);
      return Number.isNaN(num) ? onNaN : num;
    }
    function prefixedArray(input, prefix) {
      const output = [];
      for (let i = 0, max = input.length; i < max; i++) {
        output.push(prefix, input[i]);
      }
      return output;
    }
    function bufferToString(input) {
      return (Array.isArray(input) ? Buffer.concat(input) : input).toString("utf-8");
    }
    function pick(source, properties) {
      const out = {};
      properties.forEach((key) => {
        if (source[key] !== void 0) {
          out[key] = source[key];
        }
      });
      return out;
    }
    function delay(duration = 0) {
      return new Promise((done) => setTimeout(done, duration));
    }
    function orVoid(input) {
      if (input === false) {
        return void 0;
      }
      return input;
    }
    var import_file_exists;
    var NULL;
    var NOOP;
    var objectToString;
    var init_util = __esm2({
      "src/lib/utils/util.ts"() {
        "use strict";
        import_file_exists = require_dist();
        init_argument_filters();
        NULL = "\0";
        NOOP = () => {
        };
        objectToString = Object.prototype.toString.call.bind(Object.prototype.toString);
      }
    });
    function filterType(input, filter, def) {
      if (filter(input)) {
        return input;
      }
      return arguments.length > 2 ? def : void 0;
    }
    function filterPrimitives(input, omit) {
      const type = (0, import_args_pathspec.isPathSpec)(input) ? "string" : typeof input;
      return /number|string|boolean/.test(type) && (!omit || !omit.includes(type));
    }
    function filterPlainObject(input) {
      return !!input && objectToString(input) === "[object Object]";
    }
    function filterFunction(input) {
      return typeof input === "function";
    }
    var import_args_pathspec;
    var filterArray;
    var filterNumber;
    var filterString;
    var filterStringOrStringArray;
    var filterHasLength;
    var init_argument_filters = __esm2({
      "src/lib/utils/argument-filters.ts"() {
        "use strict";
        import_args_pathspec = require_dist2();
        init_util();
        filterArray = (input) => {
          return Array.isArray(input);
        };
        filterNumber = (input) => {
          return typeof input === "number";
        };
        filterString = (input) => {
          return typeof input === "string" || (0, import_args_pathspec.isPathSpec)(input);
        };
        filterStringOrStringArray = (input) => {
          return filterString(input) || Array.isArray(input) && input.every(filterString);
        };
        filterHasLength = (input) => {
          if (input == null || "number|boolean|function".includes(typeof input)) {
            return false;
          }
          return typeof input.length === "number";
        };
      }
    });
    var ExitCodes;
    var init_exit_codes = __esm2({
      "src/lib/utils/exit-codes.ts"() {
        "use strict";
        ExitCodes = /* @__PURE__ */ ((ExitCodes2) => {
          ExitCodes2[ExitCodes2["SUCCESS"] = 0] = "SUCCESS";
          ExitCodes2[ExitCodes2["ERROR"] = 1] = "ERROR";
          ExitCodes2[ExitCodes2["NOT_FOUND"] = -2] = "NOT_FOUND";
          ExitCodes2[ExitCodes2["UNCLEAN"] = 128] = "UNCLEAN";
          return ExitCodes2;
        })(ExitCodes || {});
      }
    });
    var GitOutputStreams;
    var init_git_output_streams = __esm2({
      "src/lib/utils/git-output-streams.ts"() {
        "use strict";
        GitOutputStreams = class _GitOutputStreams {
          constructor(stdOut, stdErr) {
            this.stdOut = stdOut;
            this.stdErr = stdErr;
          }
          asStrings() {
            return new _GitOutputStreams(this.stdOut.toString("utf8"), this.stdErr.toString("utf8"));
          }
        };
      }
    });
    function useMatchesDefault() {
      throw new Error(`LineParser:useMatches not implemented`);
    }
    var LineParser;
    var RemoteLineParser;
    var init_line_parser = __esm2({
      "src/lib/utils/line-parser.ts"() {
        "use strict";
        LineParser = class {
          constructor(regExp, useMatches) {
            this.matches = [];
            this.useMatches = useMatchesDefault;
            this.parse = (line, target) => {
              this.resetMatches();
              if (!this._regExp.every((reg, index) => this.addMatch(reg, index, line(index)))) {
                return false;
              }
              return this.useMatches(target, this.prepareMatches()) !== false;
            };
            this._regExp = Array.isArray(regExp) ? regExp : [regExp];
            if (useMatches) {
              this.useMatches = useMatches;
            }
          }
          resetMatches() {
            this.matches.length = 0;
          }
          prepareMatches() {
            return this.matches;
          }
          addMatch(reg, index, line) {
            const matched = line && reg.exec(line);
            if (matched) {
              this.pushMatch(index, matched);
            }
            return !!matched;
          }
          pushMatch(_index, matched) {
            this.matches.push(...matched.slice(1));
          }
        };
        RemoteLineParser = class extends LineParser {
          addMatch(reg, index, line) {
            return /^remote:\s/.test(String(line)) && super.addMatch(reg, index, line);
          }
          pushMatch(index, matched) {
            if (index > 0 || matched.length > 1) {
              super.pushMatch(index, matched);
            }
          }
        };
      }
    });
    function createInstanceConfig(...options) {
      const baseDir = process.cwd();
      const config = Object.assign(
        { baseDir, ...defaultOptions },
        ...options.filter((o) => typeof o === "object" && o)
      );
      config.baseDir = config.baseDir || baseDir;
      config.trimmed = config.trimmed === true;
      return config;
    }
    var defaultOptions;
    var init_simple_git_options = __esm2({
      "src/lib/utils/simple-git-options.ts"() {
        "use strict";
        defaultOptions = {
          binary: "git",
          maxConcurrentProcesses: 5,
          config: [],
          trimmed: false
        };
      }
    });
    function appendTaskOptions(options, commands = []) {
      if (!filterPlainObject(options)) {
        return commands;
      }
      return Object.keys(options).reduce((commands2, key) => {
        const value = options[key];
        if ((0, import_args_pathspec2.isPathSpec)(value)) {
          commands2.push(value);
        } else if (filterPrimitives(value, ["boolean"])) {
          commands2.push(key + "=" + value);
        } else if (Array.isArray(value)) {
          for (const v of value) {
            if (!filterPrimitives(v, ["string", "number"])) {
              commands2.push(key + "=" + v);
            }
          }
        } else {
          commands2.push(key);
        }
        return commands2;
      }, commands);
    }
    function getTrailingOptions(args, initialPrimitive = 0, objectOnly = false) {
      const command = [];
      for (let i = 0, max = initialPrimitive < 0 ? args.length : initialPrimitive; i < max; i++) {
        if ("string|number".includes(typeof args[i])) {
          command.push(String(args[i]));
        }
      }
      appendTaskOptions(trailingOptionsArgument(args), command);
      if (!objectOnly) {
        command.push(...trailingArrayArgument(args));
      }
      return command;
    }
    function trailingArrayArgument(args) {
      const hasTrailingCallback = typeof last(args) === "function";
      return asStringArray(filterType(last(args, hasTrailingCallback ? 1 : 0), filterArray, []));
    }
    function trailingOptionsArgument(args) {
      const hasTrailingCallback = filterFunction(last(args));
      return filterType(last(args, hasTrailingCallback ? 1 : 0), filterPlainObject);
    }
    function trailingFunctionArgument(args, includeNoop = true) {
      const callback = asFunction(last(args));
      return includeNoop || isUserFunction(callback) ? callback : void 0;
    }
    var import_args_pathspec2;
    var init_task_options = __esm2({
      "src/lib/utils/task-options.ts"() {
        "use strict";
        init_argument_filters();
        init_util();
        import_args_pathspec2 = require_dist2();
      }
    });
    function callTaskParser(parser4, streams) {
      return parser4(streams.stdOut, streams.stdErr);
    }
    function parseStringResponse(result, parsers12, texts, trim = true) {
      asArray(texts).forEach((text) => {
        for (let lines = toLinesWithContent(text, trim), i = 0, max = lines.length; i < max; i++) {
          const line = (offset = 0) => {
            if (i + offset >= max) {
              return;
            }
            return lines[i + offset];
          };
          parsers12.some(({ parse }) => parse(line, result));
        }
      });
      return result;
    }
    var init_task_parser = __esm2({
      "src/lib/utils/task-parser.ts"() {
        "use strict";
        init_util();
      }
    });
    var utils_exports = {};
    __export2(utils_exports, {
      ExitCodes: () => ExitCodes,
      GitOutputStreams: () => GitOutputStreams,
      LineParser: () => LineParser,
      NOOP: () => NOOP,
      NULL: () => NULL,
      RemoteLineParser: () => RemoteLineParser,
      append: () => append,
      appendTaskOptions: () => appendTaskOptions,
      asArray: () => asArray,
      asCamelCase: () => asCamelCase,
      asFunction: () => asFunction,
      asNumber: () => asNumber,
      asStringArray: () => asStringArray,
      bufferToString: () => bufferToString,
      callTaskParser: () => callTaskParser,
      createInstanceConfig: () => createInstanceConfig,
      delay: () => delay,
      filterArray: () => filterArray,
      filterFunction: () => filterFunction,
      filterHasLength: () => filterHasLength,
      filterNumber: () => filterNumber,
      filterPlainObject: () => filterPlainObject,
      filterPrimitives: () => filterPrimitives,
      filterString: () => filterString,
      filterStringOrStringArray: () => filterStringOrStringArray,
      filterType: () => filterType,
      first: () => first,
      folderExists: () => folderExists,
      forEachLineWithContent: () => forEachLineWithContent,
      getTrailingOptions: () => getTrailingOptions,
      including: () => including,
      isUserFunction: () => isUserFunction,
      last: () => last,
      objectToString: () => objectToString,
      orVoid: () => orVoid,
      parseStringResponse: () => parseStringResponse,
      pick: () => pick,
      prefixedArray: () => prefixedArray,
      remove: () => remove,
      splitOn: () => splitOn,
      toLinesWithContent: () => toLinesWithContent,
      trailingFunctionArgument: () => trailingFunctionArgument,
      trailingOptionsArgument: () => trailingOptionsArgument
    });
    var init_utils = __esm2({
      "src/lib/utils/index.ts"() {
        "use strict";
        init_argument_filters();
        init_exit_codes();
        init_git_output_streams();
        init_line_parser();
        init_simple_git_options();
        init_task_options();
        init_task_parser();
        init_util();
      }
    });
    var check_is_repo_exports = {};
    __export2(check_is_repo_exports, {
      CheckRepoActions: () => CheckRepoActions,
      checkIsBareRepoTask: () => checkIsBareRepoTask,
      checkIsRepoRootTask: () => checkIsRepoRootTask,
      checkIsRepoTask: () => checkIsRepoTask
    });
    function checkIsRepoTask(action) {
      switch (action) {
        case "bare":
          return checkIsBareRepoTask();
        case "root":
          return checkIsRepoRootTask();
      }
      const commands = ["rev-parse", "--is-inside-work-tree"];
      return {
        commands,
        format: "utf-8",
        onError,
        parser
      };
    }
    function checkIsRepoRootTask() {
      const commands = ["rev-parse", "--git-dir"];
      return {
        commands,
        format: "utf-8",
        onError,
        parser(path2) {
          return /^\.(git)?$/.test(path2.trim());
        }
      };
    }
    function checkIsBareRepoTask() {
      const commands = ["rev-parse", "--is-bare-repository"];
      return {
        commands,
        format: "utf-8",
        onError,
        parser
      };
    }
    function isNotRepoMessage(error) {
      return /(Not a git repository|Kein Git-Repository)/i.test(String(error));
    }
    var CheckRepoActions;
    var onError;
    var parser;
    var init_check_is_repo = __esm2({
      "src/lib/tasks/check-is-repo.ts"() {
        "use strict";
        init_utils();
        CheckRepoActions = /* @__PURE__ */ ((CheckRepoActions2) => {
          CheckRepoActions2["BARE"] = "bare";
          CheckRepoActions2["IN_TREE"] = "tree";
          CheckRepoActions2["IS_REPO_ROOT"] = "root";
          return CheckRepoActions2;
        })(CheckRepoActions || {});
        onError = ({ exitCode }, error, done, fail) => {
          if (exitCode === 128 && isNotRepoMessage(error)) {
            return done(Buffer.from("false"));
          }
          fail(error);
        };
        parser = (text) => {
          return text.trim() === "true";
        };
      }
    });
    function cleanSummaryParser(dryRun, text) {
      const summary = new CleanResponse(dryRun);
      const regexp = dryRun ? dryRunRemovalRegexp : removalRegexp;
      toLinesWithContent(text).forEach((line) => {
        const removed = line.replace(regexp, "");
        summary.paths.push(removed);
        (isFolderRegexp.test(removed) ? summary.folders : summary.files).push(removed);
      });
      return summary;
    }
    var CleanResponse;
    var removalRegexp;
    var dryRunRemovalRegexp;
    var isFolderRegexp;
    var init_CleanSummary = __esm2({
      "src/lib/responses/CleanSummary.ts"() {
        "use strict";
        init_utils();
        CleanResponse = class {
          constructor(dryRun) {
            this.dryRun = dryRun;
            this.paths = [];
            this.files = [];
            this.folders = [];
          }
        };
        removalRegexp = /^[a-z]+\s*/i;
        dryRunRemovalRegexp = /^[a-z]+\s+[a-z]+\s*/i;
        isFolderRegexp = /\/$/;
      }
    });
    var task_exports = {};
    __export2(task_exports, {
      EMPTY_COMMANDS: () => EMPTY_COMMANDS,
      adhocExecTask: () => adhocExecTask,
      configurationErrorTask: () => configurationErrorTask,
      isBufferTask: () => isBufferTask,
      isEmptyTask: () => isEmptyTask,
      straightThroughBufferTask: () => straightThroughBufferTask,
      straightThroughStringTask: () => straightThroughStringTask
    });
    function adhocExecTask(parser4) {
      return {
        commands: EMPTY_COMMANDS,
        format: "empty",
        parser: parser4
      };
    }
    function configurationErrorTask(error) {
      return {
        commands: EMPTY_COMMANDS,
        format: "empty",
        parser() {
          throw typeof error === "string" ? new TaskConfigurationError(error) : error;
        }
      };
    }
    function straightThroughStringTask(commands, trimmed2 = false) {
      return {
        commands,
        format: "utf-8",
        parser(text) {
          return trimmed2 ? String(text).trim() : text;
        }
      };
    }
    function straightThroughBufferTask(commands) {
      return {
        commands,
        format: "buffer",
        parser(buffer) {
          return buffer;
        }
      };
    }
    function isBufferTask(task) {
      return task.format === "buffer";
    }
    function isEmptyTask(task) {
      return task.format === "empty" || !task.commands.length;
    }
    var EMPTY_COMMANDS;
    var init_task = __esm2({
      "src/lib/tasks/task.ts"() {
        "use strict";
        init_task_configuration_error();
        EMPTY_COMMANDS = [];
      }
    });
    var clean_exports = {};
    __export2(clean_exports, {
      CONFIG_ERROR_INTERACTIVE_MODE: () => CONFIG_ERROR_INTERACTIVE_MODE,
      CONFIG_ERROR_MODE_REQUIRED: () => CONFIG_ERROR_MODE_REQUIRED,
      CONFIG_ERROR_UNKNOWN_OPTION: () => CONFIG_ERROR_UNKNOWN_OPTION,
      CleanOptions: () => CleanOptions,
      cleanTask: () => cleanTask,
      cleanWithOptionsTask: () => cleanWithOptionsTask,
      isCleanOptionsArray: () => isCleanOptionsArray
    });
    function cleanWithOptionsTask(mode, customArgs) {
      const { cleanMode, options, valid } = getCleanOptions(mode);
      if (!cleanMode) {
        return configurationErrorTask(CONFIG_ERROR_MODE_REQUIRED);
      }
      if (!valid.options) {
        return configurationErrorTask(CONFIG_ERROR_UNKNOWN_OPTION + JSON.stringify(mode));
      }
      options.push(...customArgs);
      if (options.some(isInteractiveMode)) {
        return configurationErrorTask(CONFIG_ERROR_INTERACTIVE_MODE);
      }
      return cleanTask(cleanMode, options);
    }
    function cleanTask(mode, customArgs) {
      const commands = ["clean", `-${mode}`, ...customArgs];
      return {
        commands,
        format: "utf-8",
        parser(text) {
          return cleanSummaryParser(mode === "n", text);
        }
      };
    }
    function isCleanOptionsArray(input) {
      return Array.isArray(input) && input.every((test) => CleanOptionValues.has(test));
    }
    function getCleanOptions(input) {
      let cleanMode;
      let options = [];
      let valid = { cleanMode: false, options: true };
      input.replace(/[^a-z]i/g, "").split("").forEach((char) => {
        if (isCleanMode(char)) {
          cleanMode = char;
          valid.cleanMode = true;
        } else {
          valid.options = valid.options && isKnownOption(options[options.length] = `-${char}`);
        }
      });
      return {
        cleanMode,
        options,
        valid
      };
    }
    function isCleanMode(cleanMode) {
      return cleanMode === "f" || cleanMode === "n";
    }
    function isKnownOption(option) {
      return /^-[a-z]$/i.test(option) && CleanOptionValues.has(option.charAt(1));
    }
    function isInteractiveMode(option) {
      if (/^-[^\-]/.test(option)) {
        return option.indexOf("i") > 0;
      }
      return option === "--interactive";
    }
    var CONFIG_ERROR_INTERACTIVE_MODE;
    var CONFIG_ERROR_MODE_REQUIRED;
    var CONFIG_ERROR_UNKNOWN_OPTION;
    var CleanOptions;
    var CleanOptionValues;
    var init_clean = __esm2({
      "src/lib/tasks/clean.ts"() {
        "use strict";
        init_CleanSummary();
        init_utils();
        init_task();
        CONFIG_ERROR_INTERACTIVE_MODE = "Git clean interactive mode is not supported";
        CONFIG_ERROR_MODE_REQUIRED = 'Git clean mode parameter ("n" or "f") is required';
        CONFIG_ERROR_UNKNOWN_OPTION = "Git clean unknown option found in: ";
        CleanOptions = /* @__PURE__ */ ((CleanOptions2) => {
          CleanOptions2["DRY_RUN"] = "n";
          CleanOptions2["FORCE"] = "f";
          CleanOptions2["IGNORED_INCLUDED"] = "x";
          CleanOptions2["IGNORED_ONLY"] = "X";
          CleanOptions2["EXCLUDING"] = "e";
          CleanOptions2["QUIET"] = "q";
          CleanOptions2["RECURSIVE"] = "d";
          return CleanOptions2;
        })(CleanOptions || {});
        CleanOptionValues = /* @__PURE__ */ new Set([
          "i",
          ...asStringArray(Object.values(CleanOptions))
        ]);
      }
    });
    function configListParser(text) {
      const config = new ConfigList();
      for (const item of configParser(text)) {
        config.addValue(item.file, String(item.key), item.value);
      }
      return config;
    }
    function configGetParser(text, key) {
      let value = null;
      const values = [];
      const scopes = /* @__PURE__ */ new Map();
      for (const item of configParser(text, key)) {
        if (item.key !== key) {
          continue;
        }
        values.push(value = item.value);
        if (!scopes.has(item.file)) {
          scopes.set(item.file, []);
        }
        scopes.get(item.file).push(value);
      }
      return {
        key,
        paths: Array.from(scopes.keys()),
        scopes,
        value,
        values
      };
    }
    function configFilePath(filePath) {
      return filePath.replace(/^(file):/, "");
    }
    function* configParser(text, requestedKey = null) {
      const lines = text.split("\0");
      for (let i = 0, max = lines.length - 1; i < max; ) {
        const file = configFilePath(lines[i++]);
        let value = lines[i++];
        let key = requestedKey;
        if (value.includes("\n")) {
          const line = splitOn(value, "\n");
          key = line[0];
          value = line[1];
        }
        yield { file, key, value };
      }
    }
    var ConfigList;
    var init_ConfigList = __esm2({
      "src/lib/responses/ConfigList.ts"() {
        "use strict";
        init_utils();
        ConfigList = class {
          constructor() {
            this.files = [];
            this.values = /* @__PURE__ */ Object.create(null);
          }
          get all() {
            if (!this._all) {
              this._all = this.files.reduce((all, file) => {
                return Object.assign(all, this.values[file]);
              }, {});
            }
            return this._all;
          }
          addFile(file) {
            if (!(file in this.values)) {
              const latest = last(this.files);
              this.values[file] = latest ? Object.create(this.values[latest]) : {};
              this.files.push(file);
            }
            return this.values[file];
          }
          addValue(file, key, value) {
            const values = this.addFile(file);
            if (!Object.hasOwn(values, key)) {
              values[key] = value;
            } else if (Array.isArray(values[key])) {
              values[key].push(value);
            } else {
              values[key] = [values[key], value];
            }
            this._all = void 0;
          }
        };
      }
    });
    function asConfigScope(scope, fallback) {
      if (typeof scope === "string" && Object.hasOwn(GitConfigScope, scope)) {
        return scope;
      }
      return fallback;
    }
    function addConfigTask(key, value, append2, scope) {
      const commands = ["config", `--${scope}`];
      if (append2) {
        commands.push("--add");
      }
      commands.push(key, value);
      return {
        commands,
        format: "utf-8",
        parser(text) {
          return text;
        }
      };
    }
    function getConfigTask(key, scope) {
      const commands = ["config", "--null", "--show-origin", "--get-all", key];
      if (scope) {
        commands.splice(1, 0, `--${scope}`);
      }
      return {
        commands,
        format: "utf-8",
        parser(text) {
          return configGetParser(text, key);
        }
      };
    }
    function listConfigTask(scope) {
      const commands = ["config", "--list", "--show-origin", "--null"];
      if (scope) {
        commands.push(`--${scope}`);
      }
      return {
        commands,
        format: "utf-8",
        parser(text) {
          return configListParser(text);
        }
      };
    }
    function config_default() {
      return {
        addConfig(key, value, ...rest) {
          return this._runTask(
            addConfigTask(
              key,
              value,
              rest[0] === true,
              asConfigScope(
                rest[1],
                "local"
                /* local */
              )
            ),
            trailingFunctionArgument(arguments)
          );
        },
        getConfig(key, scope) {
          return this._runTask(
            getConfigTask(key, asConfigScope(scope, void 0)),
            trailingFunctionArgument(arguments)
          );
        },
        listConfig(...rest) {
          return this._runTask(
            listConfigTask(asConfigScope(rest[0], void 0)),
            trailingFunctionArgument(arguments)
          );
        }
      };
    }
    var GitConfigScope;
    var init_config = __esm2({
      "src/lib/tasks/config.ts"() {
        "use strict";
        init_ConfigList();
        init_utils();
        GitConfigScope = /* @__PURE__ */ ((GitConfigScope2) => {
          GitConfigScope2["system"] = "system";
          GitConfigScope2["global"] = "global";
          GitConfigScope2["local"] = "local";
          GitConfigScope2["worktree"] = "worktree";
          return GitConfigScope2;
        })(GitConfigScope || {});
      }
    });
    function isDiffNameStatus(input) {
      return diffNameStatus.has(input);
    }
    var DiffNameStatus;
    var diffNameStatus;
    var init_diff_name_status = __esm2({
      "src/lib/tasks/diff-name-status.ts"() {
        "use strict";
        DiffNameStatus = /* @__PURE__ */ ((DiffNameStatus2) => {
          DiffNameStatus2["ADDED"] = "A";
          DiffNameStatus2["COPIED"] = "C";
          DiffNameStatus2["DELETED"] = "D";
          DiffNameStatus2["MODIFIED"] = "M";
          DiffNameStatus2["RENAMED"] = "R";
          DiffNameStatus2["CHANGED"] = "T";
          DiffNameStatus2["UNMERGED"] = "U";
          DiffNameStatus2["UNKNOWN"] = "X";
          DiffNameStatus2["BROKEN"] = "B";
          return DiffNameStatus2;
        })(DiffNameStatus || {});
        diffNameStatus = new Set(Object.values(DiffNameStatus));
      }
    });
    function grepQueryBuilder(...params) {
      return new GrepQuery().param(...params);
    }
    function parseGrep(grep) {
      const paths = /* @__PURE__ */ new Set();
      const results = {};
      forEachLineWithContent(grep, (input) => {
        const [path2, line, preview] = input.split(NULL);
        paths.add(path2);
        (results[path2] = results[path2] || []).push({
          line: asNumber(line),
          path: path2,
          preview
        });
      });
      return {
        paths,
        results
      };
    }
    function grep_default() {
      return {
        grep(searchTerm) {
          const then = trailingFunctionArgument(arguments);
          const options = getTrailingOptions(arguments);
          for (const option of disallowedOptions) {
            if (options.includes(option)) {
              return this._runTask(
                configurationErrorTask(`git.grep: use of "${option}" is not supported.`),
                then
              );
            }
          }
          if (typeof searchTerm === "string") {
            searchTerm = grepQueryBuilder().param(searchTerm);
          }
          const commands = ["grep", "--null", "-n", "--full-name", ...options, ...searchTerm];
          return this._runTask(
            {
              commands,
              format: "utf-8",
              parser(stdOut) {
                return parseGrep(stdOut);
              }
            },
            then
          );
        }
      };
    }
    var disallowedOptions;
    var Query;
    var _a;
    var GrepQuery;
    var init_grep = __esm2({
      "src/lib/tasks/grep.ts"() {
        "use strict";
        init_utils();
        init_task();
        disallowedOptions = ["-h"];
        Query = /* @__PURE__ */ Symbol("grepQuery");
        GrepQuery = class {
          constructor() {
            this[_a] = [];
          }
          *[(_a = Query, Symbol.iterator)]() {
            for (const query of this[Query]) {
              yield query;
            }
          }
          and(...and) {
            and.length && this[Query].push("--and", "(", ...prefixedArray(and, "-e"), ")");
            return this;
          }
          param(...param) {
            this[Query].push(...prefixedArray(param, "-e"));
            return this;
          }
        };
      }
    });
    var reset_exports = {};
    __export2(reset_exports, {
      ResetMode: () => ResetMode,
      getResetMode: () => getResetMode,
      resetTask: () => resetTask
    });
    function resetTask(mode, customArgs) {
      const commands = ["reset"];
      if (isValidResetMode(mode)) {
        commands.push(`--${mode}`);
      }
      commands.push(...customArgs);
      return straightThroughStringTask(commands);
    }
    function getResetMode(mode) {
      if (isValidResetMode(mode)) {
        return mode;
      }
      switch (typeof mode) {
        case "string":
        case "undefined":
          return "soft";
      }
      return;
    }
    function isValidResetMode(mode) {
      return typeof mode === "string" && validResetModes.includes(mode);
    }
    var ResetMode;
    var validResetModes;
    var init_reset = __esm2({
      "src/lib/tasks/reset.ts"() {
        "use strict";
        init_utils();
        init_task();
        ResetMode = /* @__PURE__ */ ((ResetMode2) => {
          ResetMode2["MIXED"] = "mixed";
          ResetMode2["SOFT"] = "soft";
          ResetMode2["HARD"] = "hard";
          ResetMode2["MERGE"] = "merge";
          ResetMode2["KEEP"] = "keep";
          return ResetMode2;
        })(ResetMode || {});
        validResetModes = asStringArray(Object.values(ResetMode));
      }
    });
    var api_exports = {};
    __export2(api_exports, {
      CheckRepoActions: () => CheckRepoActions,
      CleanOptions: () => CleanOptions,
      DiffNameStatus: () => DiffNameStatus,
      GitConfigScope: () => GitConfigScope,
      GitConstructError: () => GitConstructError,
      GitError: () => GitError,
      GitPluginError: () => GitPluginError,
      GitResponseError: () => GitResponseError,
      ResetMode: () => ResetMode,
      TaskConfigurationError: () => TaskConfigurationError,
      grepQueryBuilder: () => grepQueryBuilder,
      pathspec: () => import_args_pathspec3.pathspec
    });
    var import_args_pathspec3;
    var init_api = __esm2({
      "src/lib/api.ts"() {
        "use strict";
        import_args_pathspec3 = require_dist2();
        init_git_construct_error();
        init_git_error();
        init_git_plugin_error();
        init_git_response_error();
        init_task_configuration_error();
        init_check_is_repo();
        init_clean();
        init_config();
        init_diff_name_status();
        init_grep();
        init_reset();
      }
    });
    function abortPlugin(signal) {
      if (!signal) {
        return;
      }
      const onSpawnAfter = {
        type: "spawn.after",
        action(_data, context) {
          function kill() {
            context.kill(new GitPluginError(void 0, "abort", "Abort signal received"));
          }
          signal.addEventListener("abort", kill);
          context.spawned.on("close", () => signal.removeEventListener("abort", kill));
        }
      };
      const onSpawnBefore = {
        type: "spawn.before",
        action(_data, context) {
          if (signal.aborted) {
            context.kill(new GitPluginError(void 0, "abort", "Abort already signaled"));
          }
        }
      };
      return [onSpawnBefore, onSpawnAfter];
    }
    var init_abort_plugin = __esm2({
      "src/lib/plugins/abort-plugin.ts"() {
        "use strict";
        init_git_plugin_error();
      }
    });
    function blockUnsafeOperationsPlugin(options = {}) {
      return {
        type: "spawn.args",
        action(args, { env: env2 }) {
          for (const vulnerability of (0, import_argv_parser.vulnerabilityCheck)(args, env2)) {
            if (options[vulnerability.category] !== true) {
              throw new GitPluginError(void 0, "unsafe", vulnerability.message);
            }
          }
          return args;
        }
      };
    }
    var import_argv_parser;
    var init_block_unsafe_operations_plugin = __esm2({
      "src/lib/plugins/block-unsafe-operations-plugin.ts"() {
        "use strict";
        import_argv_parser = require_dist3();
        init_git_plugin_error();
      }
    });
    function commandConfigPrefixingPlugin(configuration) {
      const prefix = prefixedArray(configuration, "-c");
      return {
        type: "spawn.args",
        action(data) {
          return [...prefix, ...data];
        }
      };
    }
    var init_command_config_prefixing_plugin = __esm2({
      "src/lib/plugins/command-config-prefixing-plugin.ts"() {
        "use strict";
        init_utils();
      }
    });
    function completionDetectionPlugin({
      onClose = true,
      onExit = 50
    } = {}) {
      function createEvents() {
        let exitCode = -1;
        const events = {
          close: (0, import_promise_deferred.deferred)(),
          closeTimeout: (0, import_promise_deferred.deferred)(),
          exit: (0, import_promise_deferred.deferred)(),
          exitTimeout: (0, import_promise_deferred.deferred)()
        };
        const result = Promise.race([
          onClose === false ? never : events.closeTimeout.promise,
          onExit === false ? never : events.exitTimeout.promise
        ]);
        configureTimeout(onClose, events.close, events.closeTimeout);
        configureTimeout(onExit, events.exit, events.exitTimeout);
        return {
          close(code) {
            exitCode = code;
            events.close.done();
          },
          exit(code) {
            exitCode = code;
            events.exit.done();
          },
          get exitCode() {
            return exitCode;
          },
          result
        };
      }
      function configureTimeout(flag, event, timeout) {
        if (flag === false) {
          return;
        }
        (flag === true ? event.promise : event.promise.then(() => delay(flag))).then(timeout.done);
      }
      return {
        type: "spawn.after",
        async action(_data, { spawned, close }) {
          const events = createEvents();
          let deferClose = true;
          let quickClose = () => void (deferClose = false);
          spawned.stdout?.on("data", quickClose);
          spawned.stderr?.on("data", quickClose);
          spawned.on("error", quickClose);
          spawned.on("close", (code) => events.close(code));
          spawned.on("exit", (code) => events.exit(code));
          try {
            await events.result;
            if (deferClose) {
              await delay(50);
            }
            close(events.exitCode);
          } catch (err) {
            close(events.exitCode, err);
          }
        }
      };
    }
    var import_promise_deferred;
    var never;
    var init_completion_detection_plugin = __esm2({
      "src/lib/plugins/completion-detection.plugin.ts"() {
        "use strict";
        import_promise_deferred = require_dist4();
        init_utils();
        never = (0, import_promise_deferred.deferred)().promise;
      }
    });
    function isBadArgument(arg) {
      return !arg || !/^([a-z]:)?([a-z0-9/.\\_~-]+)$/i.test(arg);
    }
    function toBinaryConfig(input, allowUnsafe) {
      if (input.length < 1 || input.length > 2) {
        throw new GitPluginError(void 0, "binary", WRONG_NUMBER_ERR);
      }
      const isBad = input.some(isBadArgument);
      if (isBad) {
        if (allowUnsafe) {
          console.warn(WRONG_CHARS_ERR);
        } else {
          throw new GitPluginError(void 0, "binary", WRONG_CHARS_ERR);
        }
      }
      const [binary, prefix] = input;
      return {
        binary,
        prefix
      };
    }
    function customBinaryPlugin(plugins, input = ["git"], allowUnsafe = false) {
      let config = toBinaryConfig(asArray(input), allowUnsafe);
      plugins.on("binary", (input2) => {
        config = toBinaryConfig(asArray(input2), allowUnsafe);
      });
      plugins.append("spawn.binary", () => {
        return config.binary;
      });
      plugins.append("spawn.args", (data) => {
        return config.prefix ? [config.prefix, ...data] : data;
      });
    }
    var WRONG_NUMBER_ERR;
    var WRONG_CHARS_ERR;
    var init_custom_binary_plugin = __esm2({
      "src/lib/plugins/custom-binary.plugin.ts"() {
        "use strict";
        init_git_plugin_error();
        init_utils();
        WRONG_NUMBER_ERR = `Invalid value supplied for custom binary, requires a single string or an array containing either one or two strings`;
        WRONG_CHARS_ERR = `Invalid value supplied for custom binary, restricted characters must be removed or supply the unsafe.allowUnsafeCustomBinary option`;
      }
    });
    function isTaskError(result) {
      return !!(result.exitCode && result.stdErr.length);
    }
    function getErrorMessage(result) {
      return Buffer.concat([...result.stdOut, ...result.stdErr]);
    }
    function errorDetectionHandler(overwrite = false, isError = isTaskError, errorMessage = getErrorMessage) {
      return (error, result) => {
        if (!overwrite && error || !isError(result)) {
          return error;
        }
        return errorMessage(result);
      };
    }
    function errorDetectionPlugin(config) {
      return {
        type: "task.error",
        action(data, context) {
          const error = config(data.error, {
            stdErr: context.stdErr,
            stdOut: context.stdOut,
            exitCode: context.exitCode
          });
          if (Buffer.isBuffer(error)) {
            return { error: new GitError(void 0, error.toString("utf-8")) };
          }
          return {
            error
          };
        }
      };
    }
    var init_error_detection_plugin = __esm2({
      "src/lib/plugins/error-detection.plugin.ts"() {
        "use strict";
        init_git_error();
      }
    });
    var import_node_events;
    var PluginStore;
    var init_plugin_store = __esm2({
      "src/lib/plugins/plugin-store.ts"() {
        "use strict";
        import_node_events = require("node:events");
        init_utils();
        PluginStore = class {
          constructor() {
            this.plugins = /* @__PURE__ */ new Set();
            this.events = new import_node_events.EventEmitter();
          }
          on(type, listener) {
            this.events.on(type, listener);
          }
          reconfigure(type, data) {
            this.events.emit(type, data);
          }
          append(type, action) {
            const plugin = append(this.plugins, { type, action });
            return () => this.plugins.delete(plugin);
          }
          add(plugin) {
            const plugins = [];
            asArray(plugin).forEach((plugin2) => plugin2 && this.plugins.add(append(plugins, plugin2)));
            return () => {
              plugins.forEach((plugin2) => this.plugins.delete(plugin2));
            };
          }
          exec(type, data, context) {
            let output = data;
            const contextual = Object.freeze(Object.create(context));
            for (const plugin of this.plugins) {
              if (plugin.type === type) {
                output = plugin.action(output, contextual);
              }
            }
            return output;
          }
        };
      }
    });
    function progressMonitorPlugin(progress) {
      const progressCommand = "--progress";
      const progressMethods = ["checkout", "clone", "fetch", "pull", "push"];
      const onProgress = {
        type: "spawn.after",
        action(_data, context) {
          if (!context.commands.includes(progressCommand)) {
            return;
          }
          context.spawned.stderr?.on("data", (chunk) => {
            const message = /^([\s\S]+?):\s*(\d+)% \((\d+)\/(\d+)\)/.exec(chunk.toString("utf8"));
            if (!message) {
              return;
            }
            progress({
              method: context.method,
              stage: progressEventStage(message[1]),
              progress: asNumber(message[2]),
              processed: asNumber(message[3]),
              total: asNumber(message[4])
            });
          });
        }
      };
      const onArgs = {
        type: "spawn.args",
        action(args, context) {
          if (!progressMethods.includes(context.method)) {
            return args;
          }
          return including(args, progressCommand);
        }
      };
      return [onArgs, onProgress];
    }
    function progressEventStage(input) {
      return String(input.toLowerCase().split(" ", 1)) || "unknown";
    }
    var init_progress_monitor_plugin = __esm2({
      "src/lib/plugins/progress-monitor-plugin.ts"() {
        "use strict";
        init_utils();
      }
    });
    var init_simple_git_plugin = __esm2({
      "src/lib/plugins/simple-git-plugin.ts"() {
        "use strict";
      }
    });
    function spawnOptionsPlugin(spawnOptions) {
      const options = pick(spawnOptions, ["uid", "gid"]);
      return {
        type: "spawn.options",
        action(data) {
          return { ...options, ...data };
        }
      };
    }
    var init_spawn_options_plugin = __esm2({
      "src/lib/plugins/spawn-options-plugin.ts"() {
        "use strict";
        init_utils();
      }
    });
    function timeoutPlugin({
      block,
      stdErr = true,
      stdOut = true
    }) {
      if (block > 0) {
        return {
          type: "spawn.after",
          action(_data, context) {
            let timeout;
            function wait() {
              timeout && clearTimeout(timeout);
              timeout = setTimeout(kill, block);
            }
            function stop() {
              context.spawned.stdout?.off("data", wait);
              context.spawned.stderr?.off("data", wait);
              context.spawned.off("exit", stop);
              context.spawned.off("close", stop);
              timeout && clearTimeout(timeout);
            }
            function kill() {
              stop();
              context.kill(new GitPluginError(void 0, "timeout", `block timeout reached`));
            }
            stdOut && context.spawned.stdout?.on("data", wait);
            stdErr && context.spawned.stderr?.on("data", wait);
            context.spawned.on("exit", stop);
            context.spawned.on("close", stop);
            wait();
          }
        };
      }
    }
    var init_timout_plugin = __esm2({
      "src/lib/plugins/timout-plugin.ts"() {
        "use strict";
        init_git_plugin_error();
      }
    });
    var init_plugins = __esm2({
      "src/lib/plugins/index.ts"() {
        "use strict";
        init_abort_plugin();
        init_block_unsafe_operations_plugin();
        init_command_config_prefixing_plugin();
        init_completion_detection_plugin();
        init_custom_binary_plugin();
        init_error_detection_plugin();
        init_plugin_store();
        init_progress_monitor_plugin();
        init_simple_git_plugin();
        init_spawn_options_plugin();
        init_timout_plugin();
      }
    });
    function suffixPathsPlugin() {
      return {
        type: "spawn.args",
        action(data) {
          const prefix = [];
          let suffix;
          function append2(args) {
            (suffix = suffix || []).push(...args);
          }
          for (let i = 0; i < data.length; i++) {
            const param = data[i];
            if ((0, import_args_pathspec4.isPathSpec)(param)) {
              append2((0, import_args_pathspec4.toPaths)(param));
              continue;
            }
            if (param === "--") {
              append2(
                data.slice(i + 1).flatMap((item) => (0, import_args_pathspec4.isPathSpec)(item) && (0, import_args_pathspec4.toPaths)(item) || item)
              );
              break;
            }
            prefix.push(param);
          }
          return !suffix ? prefix : [...prefix, "--", ...suffix.map(String)];
        }
      };
    }
    var import_args_pathspec4;
    var init_suffix_paths_plugin = __esm2({
      "src/lib/plugins/suffix-paths.plugin.ts"() {
        "use strict";
        import_args_pathspec4 = require_dist2();
      }
    });
    function createLog() {
      return (0, import_debug.default)("simple-git");
    }
    function prefixedLogger(to, prefix, forward) {
      if (!prefix || !String(prefix).replace(/\s*/, "")) {
        return !forward ? to : (message, ...args) => {
          to(message, ...args);
          forward(message, ...args);
        };
      }
      return (message, ...args) => {
        to(`%s ${message}`, prefix, ...args);
        if (forward) {
          forward(message, ...args);
        }
      };
    }
    function childLoggerName(name, childDebugger, { namespace: parentNamespace }) {
      if (typeof name === "string") {
        return name;
      }
      const childNamespace = childDebugger && childDebugger.namespace || "";
      if (childNamespace.startsWith(parentNamespace)) {
        return childNamespace.substr(parentNamespace.length + 1);
      }
      return childNamespace || parentNamespace;
    }
    function createLogger(label, verbose, initialStep, infoDebugger = createLog()) {
      const labelPrefix = label && `[${label}]` || "";
      const spawned = [];
      const debugDebugger = typeof verbose === "string" ? infoDebugger.extend(verbose) : verbose;
      const key = childLoggerName(filterType(verbose, filterString), debugDebugger, infoDebugger);
      return step(initialStep);
      function sibling(name, initial) {
        return append(
          spawned,
          createLogger(label, key.replace(/^[^:]+/, name), initial, infoDebugger)
        );
      }
      function step(phase) {
        const stepPrefix = phase && `[${phase}]` || "";
        const debug2 = debugDebugger && prefixedLogger(debugDebugger, stepPrefix) || NOOP;
        const info = prefixedLogger(infoDebugger, `${labelPrefix} ${stepPrefix}`, debug2);
        return Object.assign(debugDebugger ? debug2 : info, {
          label,
          sibling,
          info,
          step
        });
      }
    }
    var import_debug;
    var init_git_logger = __esm2({
      "src/lib/git-logger.ts"() {
        "use strict";
        import_debug = __toESM2(require_src());
        init_utils();
        import_debug.default.formatters.L = (value) => String(filterHasLength(value) ? value.length : "-");
        import_debug.default.formatters.B = (value) => {
          if (Buffer.isBuffer(value)) {
            return value.toString("utf8");
          }
          return objectToString(value);
        };
      }
    });
    var TasksPendingQueue;
    var init_tasks_pending_queue = __esm2({
      "src/lib/runners/tasks-pending-queue.ts"() {
        "use strict";
        init_git_error();
        init_git_logger();
        TasksPendingQueue = class _TasksPendingQueue {
          constructor(logLabel = "GitExecutor") {
            this.logLabel = logLabel;
            this._queue = /* @__PURE__ */ new Map();
          }
          withProgress(task) {
            return this._queue.get(task);
          }
          createProgress(task) {
            const name = _TasksPendingQueue.getName(task.commands[0]);
            const logger = createLogger(this.logLabel, name);
            return {
              task,
              logger,
              name
            };
          }
          push(task) {
            const progress = this.createProgress(task);
            progress.logger("Adding task to the queue, commands = %o", task.commands);
            this._queue.set(task, progress);
            return progress;
          }
          fatal(err) {
            for (const [task, { logger }] of Array.from(this._queue.entries())) {
              if (task === err.task) {
                logger.info(`Failed %o`, err);
                logger(
                  `Fatal exception, any as-yet un-started tasks run through this executor will not be attempted`
                );
              } else {
                logger.info(
                  `A fatal exception occurred in a previous task, the queue has been purged: %o`,
                  err.message
                );
              }
              this.complete(task);
            }
            if (this._queue.size !== 0) {
              throw new Error(`Queue size should be zero after fatal: ${this._queue.size}`);
            }
          }
          complete(task) {
            const progress = this.withProgress(task);
            if (progress) {
              this._queue.delete(task);
            }
          }
          attempt(task) {
            const progress = this.withProgress(task);
            if (!progress) {
              throw new GitError(void 0, "TasksPendingQueue: attempt called for an unknown task");
            }
            progress.logger("Starting task");
            return progress;
          }
          static getName(name = "empty") {
            return `task:${name}:${++_TasksPendingQueue.counter}`;
          }
          static {
            this.counter = 0;
          }
        };
      }
    });
    function pluginContext(task, commands) {
      return {
        method: first(task.commands) || "",
        commands
      };
    }
    function onErrorReceived(target, logger) {
      return (err) => {
        logger(`[ERROR] child process exception %o`, err);
        target.push(Buffer.from(String(err.stack), "ascii"));
      };
    }
    function onDataReceived(target, name, logger, output) {
      return (buffer) => {
        logger(`%s received %L bytes`, name, buffer);
        output(`%B`, buffer);
        target.push(buffer);
      };
    }
    var import_child_process;
    var GitExecutorChain;
    var init_git_executor_chain = __esm2({
      "src/lib/runners/git-executor-chain.ts"() {
        "use strict";
        import_child_process = require("child_process");
        init_git_error();
        init_task();
        init_utils();
        init_tasks_pending_queue();
        GitExecutorChain = class {
          constructor(_executor, _scheduler, _plugins) {
            this._executor = _executor;
            this._scheduler = _scheduler;
            this._plugins = _plugins;
            this._chain = Promise.resolve();
            this._queue = new TasksPendingQueue();
          }
          get cwd() {
            return this._cwd || this._executor.cwd;
          }
          set cwd(cwd) {
            this._cwd = cwd;
          }
          get env() {
            return this._executor.env;
          }
          get outputHandler() {
            return this._executor.outputHandler;
          }
          chain() {
            return this;
          }
          push(task) {
            this._queue.push(task);
            return this._chain = this._chain.then(() => this.attemptTask(task));
          }
          async attemptTask(task) {
            const onScheduleComplete = await this._scheduler.next();
            const onQueueComplete = () => this._queue.complete(task);
            try {
              const { logger } = this._queue.attempt(task);
              return await (isEmptyTask(task) ? this.attemptEmptyTask(task, logger) : this.attemptRemoteTask(task, logger));
            } catch (e) {
              throw this.onFatalException(task, e);
            } finally {
              onQueueComplete();
              onScheduleComplete();
            }
          }
          onFatalException(task, e) {
            const gitError = e instanceof GitError ? Object.assign(e, { task }) : new GitError(task, e && String(e));
            this._chain = Promise.resolve();
            this._queue.fatal(gitError);
            return gitError;
          }
          async attemptRemoteTask(task, logger) {
            const binary = this._plugins.exec("spawn.binary", "", pluginContext(task, task.commands));
            const args = this._plugins.exec("spawn.args", [...task.commands], {
              ...pluginContext(task, task.commands),
              env: { ...this.env }
            });
            const raw = await this.gitResponse(
              task,
              binary,
              args,
              this.outputHandler,
              logger.step("SPAWN")
            );
            const outputStreams = await this.handleTaskData(task, args, raw, logger.step("HANDLE"));
            logger(`passing response to task's parser as a %s`, task.format);
            if (isBufferTask(task)) {
              return callTaskParser(task.parser, outputStreams);
            }
            return callTaskParser(task.parser, outputStreams.asStrings());
          }
          async attemptEmptyTask(task, logger) {
            logger(`empty task bypassing child process to call to task's parser`);
            return task.parser(this);
          }
          handleTaskData(task, args, result, logger) {
            const { exitCode, rejection, stdOut, stdErr } = result;
            return new Promise((done, fail) => {
              logger(`Preparing to handle process response exitCode=%d stdOut=`, exitCode);
              const { error } = this._plugins.exec(
                "task.error",
                { error: rejection },
                {
                  ...pluginContext(task, args),
                  ...result
                }
              );
              if (error && task.onError) {
                logger.info(`exitCode=%s handling with custom error handler`);
                return task.onError(
                  result,
                  error,
                  (newStdOut) => {
                    logger.info(`custom error handler treated as success`);
                    logger(`custom error returned a %s`, objectToString(newStdOut));
                    done(
                      new GitOutputStreams(
                        Array.isArray(newStdOut) ? Buffer.concat(newStdOut) : newStdOut,
                        Buffer.concat(stdErr)
                      )
                    );
                  },
                  fail
                );
              }
              if (error) {
                logger.info(
                  `handling as error: exitCode=%s stdErr=%s rejection=%o`,
                  exitCode,
                  stdErr.length,
                  rejection
                );
                return fail(error);
              }
              logger.info(`retrieving task output complete`);
              done(new GitOutputStreams(Buffer.concat(stdOut), Buffer.concat(stdErr)));
            });
          }
          async gitResponse(task, command, args, outputHandler, logger) {
            const outputLogger = logger.sibling("output");
            const spawnOptions = this._plugins.exec(
              "spawn.options",
              {
                cwd: this.cwd,
                env: this.env,
                windowsHide: true
              },
              pluginContext(task, task.commands)
            );
            return new Promise((done) => {
              const stdOut = [];
              const stdErr = [];
              logger.info(`%s %o`, command, args);
              logger("%O", spawnOptions);
              let rejection = this._beforeSpawn(task, args);
              if (rejection) {
                return done({
                  stdOut,
                  stdErr,
                  exitCode: 9901,
                  rejection
                });
              }
              this._plugins.exec("spawn.before", void 0, {
                ...pluginContext(task, args),
                kill(reason) {
                  rejection = reason || rejection;
                }
              });
              const spawned = (0, import_child_process.spawn)(command, args, spawnOptions);
              spawned.stdout.on(
                "data",
                onDataReceived(stdOut, "stdOut", logger, outputLogger.step("stdOut"))
              );
              spawned.stderr.on(
                "data",
                onDataReceived(stdErr, "stdErr", logger, outputLogger.step("stdErr"))
              );
              spawned.on("error", onErrorReceived(stdErr, logger));
              if (outputHandler) {
                logger(`Passing child process stdOut/stdErr to custom outputHandler`);
                outputHandler(command, spawned.stdout, spawned.stderr, [...args]);
              }
              this._plugins.exec("spawn.after", void 0, {
                ...pluginContext(task, args),
                spawned,
                close(exitCode, reason) {
                  done({
                    stdOut,
                    stdErr,
                    exitCode,
                    rejection: rejection || reason
                  });
                },
                kill(reason) {
                  if (spawned.killed) {
                    return;
                  }
                  rejection = reason;
                  spawned.kill("SIGINT");
                }
              });
            });
          }
          _beforeSpawn(task, args) {
            let rejection;
            this._plugins.exec("spawn.before", void 0, {
              ...pluginContext(task, args),
              kill(reason) {
                rejection = reason || rejection;
              }
            });
            return rejection;
          }
        };
      }
    });
    var git_executor_exports = {};
    __export2(git_executor_exports, {
      GitExecutor: () => GitExecutor
    });
    var GitExecutor;
    var init_git_executor = __esm2({
      "src/lib/runners/git-executor.ts"() {
        "use strict";
        init_git_executor_chain();
        GitExecutor = class {
          constructor(cwd, _scheduler, _plugins) {
            this.cwd = cwd;
            this._scheduler = _scheduler;
            this._plugins = _plugins;
            this._chain = new GitExecutorChain(this, this._scheduler, this._plugins);
          }
          chain() {
            return new GitExecutorChain(this, this._scheduler, this._plugins);
          }
          push(task) {
            return this._chain.push(task);
          }
        };
      }
    });
    function taskCallback(task, response, callback = NOOP) {
      const onSuccess = (data) => {
        callback(null, data);
      };
      const onError2 = (err) => {
        if (err?.task === task) {
          callback(
            err instanceof GitResponseError ? addDeprecationNoticeToError(err) : err,
            void 0
          );
        }
      };
      response.then(onSuccess, onError2);
    }
    function addDeprecationNoticeToError(err) {
      let log = (name) => {
        console.warn(
          `simple-git deprecation notice: accessing GitResponseError.${name} should be GitResponseError.git.${name}, this will no longer be available in version 3`
        );
        log = NOOP;
      };
      return Object.create(err, Object.getOwnPropertyNames(err.git).reduce(descriptorReducer, {}));
      function descriptorReducer(all, name) {
        if (name in err) {
          return all;
        }
        all[name] = {
          enumerable: false,
          configurable: false,
          get() {
            log(name);
            return err.git[name];
          }
        };
        return all;
      }
    }
    var init_task_callback = __esm2({
      "src/lib/task-callback.ts"() {
        "use strict";
        init_git_response_error();
        init_utils();
      }
    });
    function changeWorkingDirectoryTask(directory, root) {
      return adhocExecTask((instance) => {
        if (!folderExists(directory)) {
          throw new Error(`Git.cwd: cannot change to non-directory "${directory}"`);
        }
        return (root || instance).cwd = directory;
      });
    }
    var init_change_working_directory = __esm2({
      "src/lib/tasks/change-working-directory.ts"() {
        "use strict";
        init_utils();
        init_task();
      }
    });
    function checkoutTask(args) {
      const commands = ["checkout", ...args];
      if (commands[1] === "-b" && commands.includes("-B")) {
        commands[1] = remove(commands, "-B");
      }
      return straightThroughStringTask(commands);
    }
    function checkout_default() {
      return {
        checkout() {
          return this._runTask(
            checkoutTask(getTrailingOptions(arguments, 1)),
            trailingFunctionArgument(arguments)
          );
        },
        checkoutBranch(branchName, startPoint) {
          return this._runTask(
            checkoutTask(["-b", branchName, startPoint, ...getTrailingOptions(arguments)]),
            trailingFunctionArgument(arguments)
          );
        },
        checkoutLocalBranch(branchName) {
          return this._runTask(
            checkoutTask(["-b", branchName, ...getTrailingOptions(arguments)]),
            trailingFunctionArgument(arguments)
          );
        }
      };
    }
    var init_checkout = __esm2({
      "src/lib/tasks/checkout.ts"() {
        "use strict";
        init_utils();
        init_task();
      }
    });
    function countObjectsResponse() {
      return {
        count: 0,
        garbage: 0,
        inPack: 0,
        packs: 0,
        prunePackable: 0,
        size: 0,
        sizeGarbage: 0,
        sizePack: 0
      };
    }
    function count_objects_default() {
      return {
        countObjects() {
          return this._runTask({
            commands: ["count-objects", "--verbose"],
            format: "utf-8",
            parser(stdOut) {
              return parseStringResponse(countObjectsResponse(), [parser2], stdOut);
            }
          });
        }
      };
    }
    var parser2;
    var init_count_objects = __esm2({
      "src/lib/tasks/count-objects.ts"() {
        "use strict";
        init_utils();
        parser2 = new LineParser(
          /([a-z-]+): (\d+)$/,
          (result, [key, value]) => {
            const property = asCamelCase(key);
            if (Object.hasOwn(result, property)) {
              result[property] = asNumber(value);
            }
          }
        );
      }
    });
    function parseCommitResult(stdOut) {
      const result = {
        author: null,
        branch: "",
        commit: "",
        root: false,
        summary: {
          changes: 0,
          insertions: 0,
          deletions: 0
        }
      };
      return parseStringResponse(result, parsers, stdOut);
    }
    var parsers;
    var init_parse_commit = __esm2({
      "src/lib/parsers/parse-commit.ts"() {
        "use strict";
        init_utils();
        parsers = [
          new LineParser(/^\[([^\s]+)( \([^)]+\))? ([^\]]+)/, (result, [branch, root, commit]) => {
            result.branch = branch;
            result.commit = commit;
            result.root = !!root;
          }),
          new LineParser(/\s*Author:\s(.+)/i, (result, [author]) => {
            const parts = author.split("<");
            const email = parts.pop();
            if (!email || !email.includes("@")) {
              return;
            }
            result.author = {
              email: email.substr(0, email.length - 1),
              name: parts.join("<").trim()
            };
          }),
          new LineParser(
            /(\d+)[^,]*(?:,\s*(\d+)[^,]*)(?:,\s*(\d+))/g,
            (result, [changes, insertions, deletions]) => {
              result.summary.changes = parseInt(changes, 10) || 0;
              result.summary.insertions = parseInt(insertions, 10) || 0;
              result.summary.deletions = parseInt(deletions, 10) || 0;
            }
          ),
          new LineParser(
            /^(\d+)[^,]*(?:,\s*(\d+)[^(]+\(([+-]))?/,
            (result, [changes, lines, direction]) => {
              result.summary.changes = parseInt(changes, 10) || 0;
              const count = parseInt(lines, 10) || 0;
              if (direction === "-") {
                result.summary.deletions = count;
              } else if (direction === "+") {
                result.summary.insertions = count;
              }
            }
          )
        ];
      }
    });
    function commitTask(message, files, customArgs) {
      const commands = [
        "-c",
        "core.abbrev=40",
        "commit",
        ...prefixedArray(message, "-m"),
        ...files,
        ...customArgs
      ];
      return {
        commands,
        format: "utf-8",
        parser: parseCommitResult
      };
    }
    function commit_default() {
      return {
        commit(message, ...rest) {
          const next = trailingFunctionArgument(arguments);
          const task = rejectDeprecatedSignatures(message) || commitTask(
            asArray(message),
            asArray(filterType(rest[0], filterStringOrStringArray, [])),
            [
              ...asStringArray(filterType(rest[1], filterArray, [])),
              ...getTrailingOptions(arguments, 0, true)
            ]
          );
          return this._runTask(task, next);
        }
      };
      function rejectDeprecatedSignatures(message) {
        return !filterStringOrStringArray(message) && configurationErrorTask(
          `git.commit: requires the commit message to be supplied as a string/string[]`
        );
      }
    }
    var init_commit = __esm2({
      "src/lib/tasks/commit.ts"() {
        "use strict";
        init_parse_commit();
        init_utils();
        init_task();
      }
    });
    function first_commit_default() {
      return {
        firstCommit() {
          return this._runTask(
            straightThroughStringTask(["rev-list", "--max-parents=0", "HEAD"], true),
            trailingFunctionArgument(arguments)
          );
        }
      };
    }
    var init_first_commit = __esm2({
      "src/lib/tasks/first-commit.ts"() {
        "use strict";
        init_utils();
        init_task();
      }
    });
    function hashObjectTask(filePath, write) {
      const commands = ["hash-object", filePath];
      if (write) {
        commands.push("-w");
      }
      return straightThroughStringTask(commands, true);
    }
    var init_hash_object = __esm2({
      "src/lib/tasks/hash-object.ts"() {
        "use strict";
        init_task();
      }
    });
    function parseInit(bare, path2, text) {
      const response = String(text).trim();
      let result;
      if (result = initResponseRegex.exec(response)) {
        return new InitSummary(bare, path2, false, result[1]);
      }
      if (result = reInitResponseRegex.exec(response)) {
        return new InitSummary(bare, path2, true, result[1]);
      }
      let gitDir = "";
      const tokens = response.split(" ");
      while (tokens.length) {
        const token = tokens.shift();
        if (token === "in") {
          gitDir = tokens.join(" ");
          break;
        }
      }
      return new InitSummary(bare, path2, /^re/i.test(response), gitDir);
    }
    var InitSummary;
    var initResponseRegex;
    var reInitResponseRegex;
    var init_InitSummary = __esm2({
      "src/lib/responses/InitSummary.ts"() {
        "use strict";
        InitSummary = class {
          constructor(bare, path2, existing, gitDir) {
            this.bare = bare;
            this.path = path2;
            this.existing = existing;
            this.gitDir = gitDir;
          }
        };
        initResponseRegex = /^Init.+ repository in (.+)$/;
        reInitResponseRegex = /^Rein.+ in (.+)$/;
      }
    });
    function hasBareCommand(command) {
      return command.includes(bareCommand);
    }
    function initTask(bare = false, path2, customArgs) {
      const commands = ["init", ...customArgs];
      if (bare && !hasBareCommand(commands)) {
        commands.splice(1, 0, bareCommand);
      }
      return {
        commands,
        format: "utf-8",
        parser(text) {
          return parseInit(commands.includes("--bare"), path2, text);
        }
      };
    }
    var bareCommand;
    var init_init = __esm2({
      "src/lib/tasks/init.ts"() {
        "use strict";
        init_InitSummary();
        bareCommand = "--bare";
      }
    });
    function logFormatFromCommand(customArgs) {
      for (let i = 0; i < customArgs.length; i++) {
        const format = logFormatRegex.exec(customArgs[i]);
        if (format) {
          return `--${format[1]}`;
        }
      }
      return "";
    }
    function isLogFormat(customArg) {
      return logFormatRegex.test(customArg);
    }
    var logFormatRegex;
    var init_log_format = __esm2({
      "src/lib/args/log-format.ts"() {
        "use strict";
        logFormatRegex = /^--(stat|numstat|name-only|name-status)(=|$)/;
      }
    });
    var DiffSummary;
    var init_DiffSummary = __esm2({
      "src/lib/responses/DiffSummary.ts"() {
        "use strict";
        DiffSummary = class {
          constructor() {
            this.changed = 0;
            this.deletions = 0;
            this.insertions = 0;
            this.files = [];
          }
        };
      }
    });
    function getDiffParser(format = "") {
      const parser4 = diffSummaryParsers[format];
      return (stdOut) => parseStringResponse(new DiffSummary(), parser4, stdOut, false);
    }
    var statParser;
    var numStatParser;
    var nameOnlyParser;
    var nameStatusParser;
    var diffSummaryParsers;
    var init_parse_diff_summary = __esm2({
      "src/lib/parsers/parse-diff-summary.ts"() {
        "use strict";
        init_log_format();
        init_DiffSummary();
        init_diff_name_status();
        init_utils();
        statParser = [
          new LineParser(
            /^(.+)\s+\|\s+(\d+)(\s+[+\-]+)?$/,
            (result, [file, changes, alterations = ""]) => {
              result.files.push({
                file: file.trim(),
                changes: asNumber(changes),
                insertions: alterations.replace(/[^+]/g, "").length,
                deletions: alterations.replace(/[^-]/g, "").length,
                binary: false
              });
            }
          ),
          new LineParser(
            /^(.+) \|\s+Bin ([0-9.]+) -> ([0-9.]+) ([a-z]+)/,
            (result, [file, before, after]) => {
              result.files.push({
                file: file.trim(),
                before: asNumber(before),
                after: asNumber(after),
                binary: true
              });
            }
          ),
          new LineParser(
            /(\d+) files? changed\s*((?:, \d+ [^,]+){0,2})/,
            (result, [changed, summary]) => {
              const inserted = /(\d+) i/.exec(summary);
              const deleted = /(\d+) d/.exec(summary);
              result.changed = asNumber(changed);
              result.insertions = asNumber(inserted?.[1]);
              result.deletions = asNumber(deleted?.[1]);
            }
          )
        ];
        numStatParser = [
          new LineParser(
            /(\d+)\t(\d+)\t(.+)$/,
            (result, [changesInsert, changesDelete, file]) => {
              const insertions = asNumber(changesInsert);
              const deletions = asNumber(changesDelete);
              result.changed++;
              result.insertions += insertions;
              result.deletions += deletions;
              result.files.push({
                file,
                changes: insertions + deletions,
                insertions,
                deletions,
                binary: false
              });
            }
          ),
          new LineParser(/-\t-\t(.+)$/, (result, [file]) => {
            result.changed++;
            result.files.push({
              file,
              after: 0,
              before: 0,
              binary: true
            });
          })
        ];
        nameOnlyParser = [
          new LineParser(/(.+)$/, (result, [file]) => {
            result.changed++;
            result.files.push({
              file,
              changes: 0,
              insertions: 0,
              deletions: 0,
              binary: false
            });
          })
        ];
        nameStatusParser = [
          new LineParser(
            /([ACDMRTUXB])([0-9]{0,3})\t(.[^\t]*)(\t(.[^\t]*))?$/,
            (result, [status, similarity, from, _to, to]) => {
              result.changed++;
              result.files.push({
                file: to ?? from,
                changes: 0,
                insertions: 0,
                deletions: 0,
                binary: false,
                status: orVoid(isDiffNameStatus(status) && status),
                from: orVoid(!!to && from !== to && from),
                similarity: asNumber(similarity)
              });
            }
          )
        ];
        diffSummaryParsers = {
          [
            ""
            /* NONE */
          ]: statParser,
          [
            "--stat"
            /* STAT */
          ]: statParser,
          [
            "--numstat"
            /* NUM_STAT */
          ]: numStatParser,
          [
            "--name-status"
            /* NAME_STATUS */
          ]: nameStatusParser,
          [
            "--name-only"
            /* NAME_ONLY */
          ]: nameOnlyParser
        };
      }
    });
    function lineBuilder(tokens, fields) {
      return fields.reduce(
        (line, field, index) => {
          line[field] = tokens[index] || "";
          return line;
        },
        /* @__PURE__ */ Object.create({ diff: null })
      );
    }
    function createListLogSummaryParser(splitter = SPLITTER, fields = defaultFieldNames, logFormat = "") {
      const parseDiffResult = getDiffParser(logFormat);
      return function(stdOut) {
        const all = toLinesWithContent(
          stdOut.trim(),
          false,
          START_BOUNDARY
        ).map(function(item) {
          const lineDetail = item.split(COMMIT_BOUNDARY);
          const listLogLine = lineBuilder(lineDetail[0].split(splitter), fields);
          if (lineDetail.length > 1 && !!lineDetail[1].trim()) {
            listLogLine.diff = parseDiffResult(lineDetail[1]);
          }
          return listLogLine;
        });
        return {
          all,
          latest: all.length && all[0] || null,
          total: all.length
        };
      };
    }
    var START_BOUNDARY;
    var COMMIT_BOUNDARY;
    var SPLITTER;
    var defaultFieldNames;
    var init_parse_list_log_summary = __esm2({
      "src/lib/parsers/parse-list-log-summary.ts"() {
        "use strict";
        init_utils();
        init_parse_diff_summary();
        init_log_format();
        START_BOUNDARY = "\xF2\xF2\xF2\xF2\xF2\xF2 ";
        COMMIT_BOUNDARY = " \xF2\xF2";
        SPLITTER = " \xF2 ";
        defaultFieldNames = ["hash", "date", "message", "refs", "author_name", "author_email"];
      }
    });
    var diff_exports = {};
    __export2(diff_exports, {
      diffSummaryTask: () => diffSummaryTask,
      validateLogFormatConfig: () => validateLogFormatConfig
    });
    function diffSummaryTask(customArgs) {
      let logFormat = logFormatFromCommand(customArgs);
      const commands = ["diff"];
      if (logFormat === "") {
        logFormat = "--stat";
        commands.push("--stat=4096");
      }
      commands.push(...customArgs);
      return validateLogFormatConfig(commands) || {
        commands,
        format: "utf-8",
        parser: getDiffParser(logFormat)
      };
    }
    function validateLogFormatConfig(customArgs) {
      const flags = customArgs.filter(isLogFormat);
      if (flags.length > 1) {
        return configurationErrorTask(
          `Summary flags are mutually exclusive - pick one of ${flags.join(",")}`
        );
      }
      if (flags.length && customArgs.includes("-z")) {
        return configurationErrorTask(
          `Summary flag ${flags} parsing is not compatible with null termination option '-z'`
        );
      }
    }
    var init_diff = __esm2({
      "src/lib/tasks/diff.ts"() {
        "use strict";
        init_log_format();
        init_parse_diff_summary();
        init_task();
      }
    });
    function prettyFormat(format, splitter) {
      const fields = [];
      const formatStr = [];
      Object.keys(format).forEach((field) => {
        fields.push(field);
        formatStr.push(String(format[field]));
      });
      return [fields, formatStr.join(splitter)];
    }
    function userOptions(input) {
      return Object.keys(input).reduce((out, key) => {
        if (!(key in excludeOptions)) {
          out[key] = input[key];
        }
        return out;
      }, {});
    }
    function parseLogOptions(opt = {}, customArgs = []) {
      const splitter = filterType(opt.splitter, filterString, SPLITTER);
      const format = filterPlainObject(opt.format) ? opt.format : {
        hash: "%H",
        date: opt.strictDate === false ? "%ai" : "%aI",
        message: "%s",
        refs: "%D",
        body: opt.multiLine ? "%B" : "%b",
        author_name: opt.mailMap !== false ? "%aN" : "%an",
        author_email: opt.mailMap !== false ? "%aE" : "%ae"
      };
      const [fields, formatStr] = prettyFormat(format, splitter);
      const suffix = [];
      const command = [
        `--pretty=format:${START_BOUNDARY}${formatStr}${COMMIT_BOUNDARY}`,
        ...customArgs
      ];
      const maxCount = opt.n || opt["max-count"] || opt.maxCount;
      if (maxCount) {
        command.push(`--max-count=${maxCount}`);
      }
      if (opt.from || opt.to) {
        const rangeOperator = opt.symmetric !== false ? "..." : "..";
        suffix.push(`${opt.from || ""}${rangeOperator}${opt.to || ""}`);
      }
      if (filterString(opt.file)) {
        command.push("--follow", (0, import_args_pathspec5.pathspec)(opt.file));
      }
      appendTaskOptions(userOptions(opt), command);
      return {
        fields,
        splitter,
        commands: [...command, ...suffix]
      };
    }
    function logTask(splitter, fields, customArgs) {
      const parser4 = createListLogSummaryParser(splitter, fields, logFormatFromCommand(customArgs));
      return {
        commands: ["log", ...customArgs],
        format: "utf-8",
        parser: parser4
      };
    }
    function log_default() {
      return {
        log(...rest) {
          const next = trailingFunctionArgument(arguments);
          const options = parseLogOptions(
            trailingOptionsArgument(arguments),
            asStringArray(filterType(arguments[0], filterArray, []))
          );
          const task = rejectDeprecatedSignatures(...rest) || validateLogFormatConfig(options.commands) || createLogTask(options);
          return this._runTask(task, next);
        }
      };
      function createLogTask(options) {
        return logTask(options.splitter, options.fields, options.commands);
      }
      function rejectDeprecatedSignatures(from, to) {
        return filterString(from) && filterString(to) && configurationErrorTask(
          `git.log(string, string) should be replaced with git.log({ from: string, to: string })`
        );
      }
    }
    var import_args_pathspec5;
    var excludeOptions;
    var init_log = __esm2({
      "src/lib/tasks/log.ts"() {
        "use strict";
        init_log_format();
        import_args_pathspec5 = require_dist2();
        init_parse_list_log_summary();
        init_utils();
        init_task();
        init_diff();
        excludeOptions = /* @__PURE__ */ ((excludeOptions2) => {
          excludeOptions2[excludeOptions2["--pretty"] = 0] = "--pretty";
          excludeOptions2[excludeOptions2["max-count"] = 1] = "max-count";
          excludeOptions2[excludeOptions2["maxCount"] = 2] = "maxCount";
          excludeOptions2[excludeOptions2["n"] = 3] = "n";
          excludeOptions2[excludeOptions2["file"] = 4] = "file";
          excludeOptions2[excludeOptions2["format"] = 5] = "format";
          excludeOptions2[excludeOptions2["from"] = 6] = "from";
          excludeOptions2[excludeOptions2["to"] = 7] = "to";
          excludeOptions2[excludeOptions2["splitter"] = 8] = "splitter";
          excludeOptions2[excludeOptions2["symmetric"] = 9] = "symmetric";
          excludeOptions2[excludeOptions2["mailMap"] = 10] = "mailMap";
          excludeOptions2[excludeOptions2["multiLine"] = 11] = "multiLine";
          excludeOptions2[excludeOptions2["strictDate"] = 12] = "strictDate";
          return excludeOptions2;
        })(excludeOptions || {});
      }
    });
    var MergeSummaryConflict;
    var MergeSummaryDetail;
    var init_MergeSummary = __esm2({
      "src/lib/responses/MergeSummary.ts"() {
        "use strict";
        MergeSummaryConflict = class {
          constructor(reason, file = null, meta) {
            this.reason = reason;
            this.file = file;
            this.meta = meta;
          }
          toString() {
            return `${this.file}:${this.reason}`;
          }
        };
        MergeSummaryDetail = class {
          constructor() {
            this.conflicts = [];
            this.merges = [];
            this.result = "success";
          }
          get failed() {
            return this.conflicts.length > 0;
          }
          get reason() {
            return this.result;
          }
          toString() {
            if (this.conflicts.length) {
              return `CONFLICTS: ${this.conflicts.join(", ")}`;
            }
            return "OK";
          }
        };
      }
    });
    var PullSummary;
    var PullFailedSummary;
    var init_PullSummary = __esm2({
      "src/lib/responses/PullSummary.ts"() {
        "use strict";
        PullSummary = class {
          constructor() {
            this.remoteMessages = {
              all: []
            };
            this.created = [];
            this.deleted = [];
            this.files = [];
            this.deletions = {};
            this.insertions = {};
            this.summary = {
              changes: 0,
              deletions: 0,
              insertions: 0
            };
          }
        };
        PullFailedSummary = class {
          constructor() {
            this.remote = "";
            this.hash = {
              local: "",
              remote: ""
            };
            this.branch = {
              local: "",
              remote: ""
            };
            this.message = "";
          }
          toString() {
            return this.message;
          }
        };
      }
    });
    function objectEnumerationResult(remoteMessages) {
      return remoteMessages.objects = remoteMessages.objects || {
        compressing: 0,
        counting: 0,
        enumerating: 0,
        packReused: 0,
        reused: { count: 0, delta: 0 },
        total: { count: 0, delta: 0 }
      };
    }
    function asObjectCount(source) {
      const count = /^\s*(\d+)/.exec(source);
      const delta = /delta (\d+)/i.exec(source);
      return {
        count: asNumber(count && count[1] || "0"),
        delta: asNumber(delta && delta[1] || "0")
      };
    }
    var remoteMessagesObjectParsers;
    var init_parse_remote_objects = __esm2({
      "src/lib/parsers/parse-remote-objects.ts"() {
        "use strict";
        init_utils();
        remoteMessagesObjectParsers = [
          new RemoteLineParser(
            /^remote:\s*(enumerating|counting|compressing) objects: (\d+),/i,
            (result, [action, count]) => {
              const key = action.toLowerCase();
              const enumeration = objectEnumerationResult(result.remoteMessages);
              Object.assign(enumeration, { [key]: asNumber(count) });
            }
          ),
          new RemoteLineParser(
            /^remote:\s*(enumerating|counting|compressing) objects: \d+% \(\d+\/(\d+)\),/i,
            (result, [action, count]) => {
              const key = action.toLowerCase();
              const enumeration = objectEnumerationResult(result.remoteMessages);
              Object.assign(enumeration, { [key]: asNumber(count) });
            }
          ),
          new RemoteLineParser(
            /total ([^,]+), reused ([^,]+), pack-reused (\d+)/i,
            (result, [total, reused, packReused]) => {
              const objects = objectEnumerationResult(result.remoteMessages);
              objects.total = asObjectCount(total);
              objects.reused = asObjectCount(reused);
              objects.packReused = asNumber(packReused);
            }
          )
        ];
      }
    });
    function parseRemoteMessages(_stdOut, stdErr) {
      return parseStringResponse({ remoteMessages: new RemoteMessageSummary() }, parsers2, stdErr);
    }
    var parsers2;
    var RemoteMessageSummary;
    var init_parse_remote_messages = __esm2({
      "src/lib/parsers/parse-remote-messages.ts"() {
        "use strict";
        init_utils();
        init_parse_remote_objects();
        parsers2 = [
          new RemoteLineParser(/^remote:\s*(.+)$/, (result, [text]) => {
            result.remoteMessages.all.push(text.trim());
            return false;
          }),
          ...remoteMessagesObjectParsers,
          new RemoteLineParser(
            [/create a (?:pull|merge) request/i, /\s(https?:\/\/\S+)$/],
            (result, [pullRequestUrl]) => {
              result.remoteMessages.pullRequestUrl = pullRequestUrl;
            }
          ),
          new RemoteLineParser(
            [/found (\d+) vulnerabilities.+\(([^)]+)\)/i, /\s(https?:\/\/\S+)$/],
            (result, [count, summary, url]) => {
              result.remoteMessages.vulnerabilities = {
                count: asNumber(count),
                summary,
                url
              };
            }
          )
        ];
        RemoteMessageSummary = class {
          constructor() {
            this.all = [];
          }
        };
      }
    });
    function parsePullErrorResult(stdOut, stdErr) {
      const pullError = parseStringResponse(new PullFailedSummary(), errorParsers, [stdOut, stdErr]);
      return pullError.message && pullError;
    }
    var FILE_UPDATE_REGEX;
    var SUMMARY_REGEX;
    var ACTION_REGEX;
    var parsers3;
    var errorParsers;
    var parsePullDetail;
    var parsePullResult;
    var init_parse_pull = __esm2({
      "src/lib/parsers/parse-pull.ts"() {
        "use strict";
        init_PullSummary();
        init_utils();
        init_parse_remote_messages();
        FILE_UPDATE_REGEX = /^\s*(.+?)\s+\|\s+\d+\s*(\+*)(-*)/;
        SUMMARY_REGEX = /(\d+)\D+((\d+)\D+\(\+\))?(\D+(\d+)\D+\(-\))?/;
        ACTION_REGEX = /^(create|delete) mode \d+ (.+)/;
        parsers3 = [
          new LineParser(FILE_UPDATE_REGEX, (result, [file, insertions, deletions]) => {
            result.files.push(file);
            if (insertions) {
              result.insertions[file] = insertions.length;
            }
            if (deletions) {
              result.deletions[file] = deletions.length;
            }
          }),
          new LineParser(SUMMARY_REGEX, (result, [changes, , insertions, , deletions]) => {
            if (insertions !== void 0 || deletions !== void 0) {
              result.summary.changes = +changes || 0;
              result.summary.insertions = +insertions || 0;
              result.summary.deletions = +deletions || 0;
              return true;
            }
            return false;
          }),
          new LineParser(ACTION_REGEX, (result, [action, file]) => {
            append(result.files, file);
            append(action === "create" ? result.created : result.deleted, file);
          })
        ];
        errorParsers = [
          new LineParser(/^from\s(.+)$/i, (result, [remote]) => void (result.remote = remote)),
          new LineParser(/^fatal:\s(.+)$/, (result, [message]) => void (result.message = message)),
          new LineParser(
            /([a-z0-9]+)\.\.([a-z0-9]+)\s+(\S+)\s+->\s+(\S+)$/,
            (result, [hashLocal, hashRemote, branchLocal, branchRemote]) => {
              result.branch.local = branchLocal;
              result.hash.local = hashLocal;
              result.branch.remote = branchRemote;
              result.hash.remote = hashRemote;
            }
          )
        ];
        parsePullDetail = (stdOut, stdErr) => {
          return parseStringResponse(new PullSummary(), parsers3, [stdOut, stdErr]);
        };
        parsePullResult = (stdOut, stdErr) => {
          return Object.assign(
            new PullSummary(),
            parsePullDetail(stdOut, stdErr),
            parseRemoteMessages(stdOut, stdErr)
          );
        };
      }
    });
    var parsers4;
    var parseMergeResult;
    var parseMergeDetail;
    var init_parse_merge = __esm2({
      "src/lib/parsers/parse-merge.ts"() {
        "use strict";
        init_MergeSummary();
        init_utils();
        init_parse_pull();
        parsers4 = [
          new LineParser(/^Auto-merging\s+(.+)$/, (summary, [autoMerge]) => {
            summary.merges.push(autoMerge);
          }),
          new LineParser(/^CONFLICT\s+\((.+)\): Merge conflict in (.+)$/, (summary, [reason, file]) => {
            summary.conflicts.push(new MergeSummaryConflict(reason, file));
          }),
          new LineParser(
            /^CONFLICT\s+\((.+\/delete)\): (.+) deleted in (.+) and/,
            (summary, [reason, file, deleteRef]) => {
              summary.conflicts.push(new MergeSummaryConflict(reason, file, { deleteRef }));
            }
          ),
          new LineParser(/^CONFLICT\s+\((.+)\):/, (summary, [reason]) => {
            summary.conflicts.push(new MergeSummaryConflict(reason, null));
          }),
          new LineParser(/^Automatic merge failed;\s+(.+)$/, (summary, [result]) => {
            summary.result = result;
          })
        ];
        parseMergeResult = (stdOut, stdErr) => {
          return Object.assign(parseMergeDetail(stdOut, stdErr), parsePullResult(stdOut, stdErr));
        };
        parseMergeDetail = (stdOut) => {
          return parseStringResponse(new MergeSummaryDetail(), parsers4, stdOut);
        };
      }
    });
    function mergeTask(customArgs) {
      if (!customArgs.length) {
        return configurationErrorTask("Git.merge requires at least one option");
      }
      return {
        commands: ["merge", ...customArgs],
        format: "utf-8",
        parser(stdOut, stdErr) {
          const merge = parseMergeResult(stdOut, stdErr);
          if (merge.failed) {
            throw new GitResponseError(merge);
          }
          return merge;
        }
      };
    }
    var init_merge = __esm2({
      "src/lib/tasks/merge.ts"() {
        "use strict";
        init_git_response_error();
        init_parse_merge();
        init_task();
      }
    });
    function pushResultPushedItem(local, remote, status) {
      const deleted = status.includes("deleted");
      const tag = status.includes("tag") || /^refs\/tags/.test(local);
      const alreadyUpdated = !status.includes("new");
      return {
        deleted,
        tag,
        branch: !tag,
        new: !alreadyUpdated,
        alreadyUpdated,
        local,
        remote
      };
    }
    var parsers5;
    var parsePushResult;
    var parsePushDetail;
    var init_parse_push = __esm2({
      "src/lib/parsers/parse-push.ts"() {
        "use strict";
        init_utils();
        init_parse_remote_messages();
        parsers5 = [
          new LineParser(/^Pushing to (.+)$/, (result, [repo]) => {
            result.repo = repo;
          }),
          new LineParser(/^updating local tracking ref '(.+)'/, (result, [local]) => {
            result.ref = {
              ...result.ref || {},
              local
            };
          }),
          new LineParser(/^[=*-]\s+([^:]+):(\S+)\s+\[(.+)]$/, (result, [local, remote, type]) => {
            result.pushed.push(pushResultPushedItem(local, remote, type));
          }),
          new LineParser(
            /^Branch '([^']+)' set up to track remote branch '([^']+)' from '([^']+)'/,
            (result, [local, remote, remoteName]) => {
              result.branch = {
                ...result.branch || {},
                local,
                remote,
                remoteName
              };
            }
          ),
          new LineParser(
            /^([^:]+):(\S+)\s+([a-z0-9]+)\.\.([a-z0-9]+)$/,
            (result, [local, remote, from, to]) => {
              result.update = {
                head: {
                  local,
                  remote
                },
                hash: {
                  from,
                  to
                }
              };
            }
          )
        ];
        parsePushResult = (stdOut, stdErr) => {
          const pushDetail = parsePushDetail(stdOut, stdErr);
          const responseDetail = parseRemoteMessages(stdOut, stdErr);
          return {
            ...pushDetail,
            ...responseDetail
          };
        };
        parsePushDetail = (stdOut, stdErr) => {
          return parseStringResponse({ pushed: [] }, parsers5, [stdOut, stdErr]);
        };
      }
    });
    var push_exports = {};
    __export2(push_exports, {
      pushTagsTask: () => pushTagsTask,
      pushTask: () => pushTask
    });
    function pushTagsTask(ref = {}, customArgs) {
      append(customArgs, "--tags");
      return pushTask(ref, customArgs);
    }
    function pushTask(ref = {}, customArgs) {
      const commands = ["push", ...customArgs];
      if (ref.branch) {
        commands.splice(1, 0, ref.branch);
      }
      if (ref.remote) {
        commands.splice(1, 0, ref.remote);
      }
      remove(commands, "-v");
      append(commands, "--verbose");
      append(commands, "--porcelain");
      return {
        commands,
        format: "utf-8",
        parser: parsePushResult
      };
    }
    var init_push = __esm2({
      "src/lib/tasks/push.ts"() {
        "use strict";
        init_parse_push();
        init_utils();
      }
    });
    function show_default() {
      return {
        showBuffer() {
          const commands = ["show", ...getTrailingOptions(arguments, 1)];
          if (!commands.includes("--binary")) {
            commands.splice(1, 0, "--binary");
          }
          return this._runTask(
            straightThroughBufferTask(commands),
            trailingFunctionArgument(arguments)
          );
        },
        show() {
          const commands = ["show", ...getTrailingOptions(arguments, 1)];
          return this._runTask(
            straightThroughStringTask(commands),
            trailingFunctionArgument(arguments)
          );
        }
      };
    }
    var init_show = __esm2({
      "src/lib/tasks/show.ts"() {
        "use strict";
        init_utils();
        init_task();
      }
    });
    var fromPathRegex;
    var FileStatusSummary;
    var init_FileStatusSummary = __esm2({
      "src/lib/responses/FileStatusSummary.ts"() {
        "use strict";
        fromPathRegex = /^(.+)\0(.+)$/;
        FileStatusSummary = class {
          constructor(path2, index, working_dir) {
            this.path = path2;
            this.index = index;
            this.working_dir = working_dir;
            if (index === "R" || working_dir === "R") {
              const detail = fromPathRegex.exec(path2) || [null, path2, path2];
              this.from = detail[2] || "";
              this.path = detail[1] || "";
            }
          }
        };
      }
    });
    function renamedFile(line) {
      const [to, from] = line.split(NULL);
      return {
        from: from || to,
        to
      };
    }
    function parser3(indexX, indexY, handler) {
      return [`${indexX}${indexY}`, handler];
    }
    function conflicts(indexX, ...indexY) {
      return indexY.map((y) => parser3(indexX, y, (result, file) => result.conflicted.push(file)));
    }
    function splitLine(result, lineStr) {
      const trimmed2 = lineStr.trim();
      switch (" ") {
        case trimmed2.charAt(2):
          return data(trimmed2.charAt(0), trimmed2.charAt(1), trimmed2.slice(3));
        case trimmed2.charAt(1):
          return data(" ", trimmed2.charAt(0), trimmed2.slice(2));
        default:
          return;
      }
      function data(index, workingDir, path2) {
        const raw = `${index}${workingDir}`;
        const handler = parsers6.get(raw);
        if (handler) {
          handler(result, path2);
        }
        if (raw !== "##" && raw !== "!!") {
          result.files.push(new FileStatusSummary(path2, index, workingDir));
        }
      }
    }
    var StatusSummary;
    var parsers6;
    var parseStatusSummary;
    var init_StatusSummary = __esm2({
      "src/lib/responses/StatusSummary.ts"() {
        "use strict";
        init_utils();
        init_FileStatusSummary();
        StatusSummary = class {
          constructor() {
            this.not_added = [];
            this.conflicted = [];
            this.created = [];
            this.deleted = [];
            this.ignored = void 0;
            this.modified = [];
            this.renamed = [];
            this.files = [];
            this.staged = [];
            this.ahead = 0;
            this.behind = 0;
            this.current = null;
            this.tracking = null;
            this.detached = false;
            this.isClean = () => {
              return !this.files.length;
            };
          }
        };
        parsers6 = new Map([
          parser3(
            " ",
            "A",
            (result, file) => result.created.push(file)
          ),
          parser3(
            " ",
            "D",
            (result, file) => result.deleted.push(file)
          ),
          parser3(
            " ",
            "M",
            (result, file) => result.modified.push(file)
          ),
          parser3("A", " ", (result, file) => {
            result.created.push(file);
            result.staged.push(file);
          }),
          parser3("A", "M", (result, file) => {
            result.created.push(file);
            result.staged.push(file);
            result.modified.push(file);
          }),
          parser3("D", " ", (result, file) => {
            result.deleted.push(file);
            result.staged.push(file);
          }),
          parser3("M", " ", (result, file) => {
            result.modified.push(file);
            result.staged.push(file);
          }),
          parser3("M", "M", (result, file) => {
            result.modified.push(file);
            result.staged.push(file);
          }),
          parser3("R", " ", (result, file) => {
            result.renamed.push(renamedFile(file));
          }),
          parser3("R", "M", (result, file) => {
            const renamed = renamedFile(file);
            result.renamed.push(renamed);
            result.modified.push(renamed.to);
          }),
          parser3("!", "!", (_result, _file) => {
            (_result.ignored = _result.ignored || []).push(_file);
          }),
          parser3(
            "?",
            "?",
            (result, file) => result.not_added.push(file)
          ),
          ...conflicts(
            "A",
            "A",
            "U"
            /* UNMERGED */
          ),
          ...conflicts(
            "D",
            "D",
            "U"
            /* UNMERGED */
          ),
          ...conflicts(
            "U",
            "A",
            "D",
            "U"
            /* UNMERGED */
          ),
          [
            "##",
            (result, line) => {
              const aheadReg = /ahead (\d+)/;
              const behindReg = /behind (\d+)/;
              const currentReg = /^(.+?(?=(?:\.{3}|\s|$)))/;
              const trackingReg = /\.{3}(\S*)/;
              const onEmptyBranchReg = /\son\s(\S+?)(?=\.{3}|$)/;
              let regexResult = aheadReg.exec(line);
              result.ahead = regexResult && +regexResult[1] || 0;
              regexResult = behindReg.exec(line);
              result.behind = regexResult && +regexResult[1] || 0;
              regexResult = currentReg.exec(line);
              result.current = filterType(regexResult?.[1], filterString, null);
              regexResult = trackingReg.exec(line);
              result.tracking = filterType(regexResult?.[1], filterString, null);
              regexResult = onEmptyBranchReg.exec(line);
              if (regexResult) {
                result.current = filterType(regexResult?.[1], filterString, result.current);
              }
              result.detached = /\(no branch\)/.test(line);
            }
          ]
        ]);
        parseStatusSummary = function(text) {
          const lines = text.split(NULL);
          const status = new StatusSummary();
          for (let i = 0, l = lines.length; i < l; ) {
            let line = lines[i++].trim();
            if (!line) {
              continue;
            }
            if (line.charAt(0) === "R") {
              line += NULL + (lines[i++] || "");
            }
            splitLine(status, line);
          }
          return status;
        };
      }
    });
    function statusTask(customArgs) {
      const commands = [
        "status",
        "--porcelain",
        "-b",
        "-u",
        "--null",
        ...customArgs.filter((arg) => !ignoredOptions.includes(arg))
      ];
      return {
        format: "utf-8",
        commands,
        parser(text) {
          return parseStatusSummary(text);
        }
      };
    }
    var ignoredOptions;
    var init_status = __esm2({
      "src/lib/tasks/status.ts"() {
        "use strict";
        init_StatusSummary();
        ignoredOptions = ["--null", "-z"];
      }
    });
    function versionResponse(major = 0, minor = 0, patch = 0, agent = "", installed = true) {
      return Object.defineProperty(
        {
          major,
          minor,
          patch,
          agent,
          installed
        },
        "toString",
        {
          value() {
            return `${this.major}.${this.minor}.${this.patch}`;
          },
          configurable: false,
          enumerable: false
        }
      );
    }
    function notInstalledResponse() {
      return versionResponse(0, 0, 0, "", false);
    }
    function version_default() {
      return {
        version() {
          return this._runTask({
            commands: ["--version"],
            format: "utf-8",
            parser: versionParser,
            onError(result, error, done, fail) {
              if (result.exitCode === -2) {
                return done(Buffer.from(NOT_INSTALLED));
              }
              fail(error);
            }
          });
        }
      };
    }
    function versionParser(stdOut) {
      if (stdOut === NOT_INSTALLED) {
        return notInstalledResponse();
      }
      return parseStringResponse(versionResponse(0, 0, 0, stdOut), parsers7, stdOut);
    }
    var NOT_INSTALLED;
    var parsers7;
    var init_version = __esm2({
      "src/lib/tasks/version.ts"() {
        "use strict";
        init_utils();
        NOT_INSTALLED = "installed=false";
        parsers7 = [
          new LineParser(
            /version (\d+)\.(\d+)\.(\d+)(?:\s*\((.+)\))?/,
            (result, [major, minor, patch, agent = ""]) => {
              Object.assign(
                result,
                versionResponse(asNumber(major), asNumber(minor), asNumber(patch), agent)
              );
            }
          ),
          new LineParser(
            /version (\d+)\.(\d+)\.(\D+)(.+)?$/,
            (result, [major, minor, patch, agent = ""]) => {
              Object.assign(result, versionResponse(asNumber(major), asNumber(minor), patch, agent));
            }
          )
        ];
      }
    });
    function createCloneTask(api, task, repoPath, ...args) {
      if (!filterString(repoPath)) {
        return configurationErrorTask(`git.${api}() requires a string 'repoPath'`);
      }
      return task(repoPath, filterType(args[0], filterString), getTrailingOptions(arguments));
    }
    function clone_default() {
      return {
        clone(repo, ...rest) {
          return this._runTask(
            createCloneTask("clone", cloneTask, filterType(repo, filterString), ...rest),
            trailingFunctionArgument(arguments)
          );
        },
        mirror(repo, ...rest) {
          return this._runTask(
            createCloneTask("mirror", cloneMirrorTask, filterType(repo, filterString), ...rest),
            trailingFunctionArgument(arguments)
          );
        }
      };
    }
    var import_args_pathspec6;
    var cloneTask;
    var cloneMirrorTask;
    var init_clone = __esm2({
      "src/lib/tasks/clone.ts"() {
        "use strict";
        init_task();
        init_utils();
        import_args_pathspec6 = require_dist2();
        cloneTask = (repo, directory, customArgs) => {
          const commands = ["clone", ...customArgs];
          filterString(repo) && commands.push((0, import_args_pathspec6.pathspec)(repo));
          filterString(directory) && commands.push((0, import_args_pathspec6.pathspec)(directory));
          return straightThroughStringTask(commands);
        };
        cloneMirrorTask = (repo, directory, customArgs) => {
          append(customArgs, "--mirror");
          return cloneTask(repo, directory, customArgs);
        };
      }
    });
    var simple_git_api_exports = {};
    __export2(simple_git_api_exports, {
      SimpleGitApi: () => SimpleGitApi
    });
    var SimpleGitApi;
    var init_simple_git_api = __esm2({
      "src/lib/simple-git-api.ts"() {
        "use strict";
        init_task_callback();
        init_change_working_directory();
        init_checkout();
        init_count_objects();
        init_commit();
        init_config();
        init_first_commit();
        init_grep();
        init_hash_object();
        init_init();
        init_log();
        init_merge();
        init_push();
        init_show();
        init_status();
        init_task();
        init_version();
        init_utils();
        init_clone();
        SimpleGitApi = class {
          constructor(_executor) {
            this._executor = _executor;
          }
          _runTask(task, then) {
            const chain = this._executor.chain();
            const promise = chain.push(task);
            if (then) {
              taskCallback(task, promise, then);
            }
            return Object.create(this, {
              then: { value: promise.then.bind(promise) },
              catch: { value: promise.catch.bind(promise) },
              _executor: { value: chain }
            });
          }
          add(files) {
            return this._runTask(
              straightThroughStringTask(["add", ...asArray(files)]),
              trailingFunctionArgument(arguments)
            );
          }
          cwd(directory) {
            const next = trailingFunctionArgument(arguments);
            if (typeof directory === "string") {
              return this._runTask(changeWorkingDirectoryTask(directory, this._executor), next);
            }
            if (typeof directory?.path === "string") {
              return this._runTask(
                changeWorkingDirectoryTask(
                  directory.path,
                  directory.root && this._executor || void 0
                ),
                next
              );
            }
            return this._runTask(
              configurationErrorTask("Git.cwd: workingDirectory must be supplied as a string"),
              next
            );
          }
          hashObject(path2, write) {
            return this._runTask(
              hashObjectTask(path2, write === true),
              trailingFunctionArgument(arguments)
            );
          }
          init(bare) {
            return this._runTask(
              initTask(bare === true, this._executor.cwd, getTrailingOptions(arguments)),
              trailingFunctionArgument(arguments)
            );
          }
          merge() {
            return this._runTask(
              mergeTask(getTrailingOptions(arguments)),
              trailingFunctionArgument(arguments)
            );
          }
          mergeFromTo(remote, branch) {
            if (!(filterString(remote) && filterString(branch))) {
              return this._runTask(
                configurationErrorTask(
                  `Git.mergeFromTo requires that the 'remote' and 'branch' arguments are supplied as strings`
                )
              );
            }
            return this._runTask(
              mergeTask([remote, branch, ...getTrailingOptions(arguments)]),
              trailingFunctionArgument(arguments, false)
            );
          }
          outputHandler(handler) {
            this._executor.outputHandler = handler;
            return this;
          }
          push() {
            const task = pushTask(
              {
                remote: filterType(arguments[0], filterString),
                branch: filterType(arguments[1], filterString)
              },
              getTrailingOptions(arguments)
            );
            return this._runTask(task, trailingFunctionArgument(arguments));
          }
          stash() {
            return this._runTask(
              straightThroughStringTask(["stash", ...getTrailingOptions(arguments)]),
              trailingFunctionArgument(arguments)
            );
          }
          status() {
            return this._runTask(
              statusTask(getTrailingOptions(arguments)),
              trailingFunctionArgument(arguments)
            );
          }
        };
        Object.assign(
          SimpleGitApi.prototype,
          checkout_default(),
          clone_default(),
          commit_default(),
          config_default(),
          count_objects_default(),
          first_commit_default(),
          grep_default(),
          log_default(),
          show_default(),
          version_default()
        );
      }
    });
    var scheduler_exports = {};
    __export2(scheduler_exports, {
      Scheduler: () => Scheduler
    });
    var import_promise_deferred2;
    var createScheduledTask;
    var Scheduler;
    var init_scheduler = __esm2({
      "src/lib/runners/scheduler.ts"() {
        "use strict";
        init_utils();
        import_promise_deferred2 = require_dist4();
        init_git_logger();
        createScheduledTask = /* @__PURE__ */ (() => {
          let id = 0;
          return () => {
            id++;
            const { promise, done } = (0, import_promise_deferred2.createDeferred)();
            return {
              promise,
              done,
              id
            };
          };
        })();
        Scheduler = class {
          constructor(concurrency = 2) {
            this.concurrency = concurrency;
            this.logger = createLogger("", "scheduler");
            this.pending = [];
            this.running = [];
            this.logger(`Constructed, concurrency=%s`, concurrency);
          }
          schedule() {
            if (!this.pending.length || this.running.length >= this.concurrency) {
              this.logger(
                `Schedule attempt ignored, pending=%s running=%s concurrency=%s`,
                this.pending.length,
                this.running.length,
                this.concurrency
              );
              return;
            }
            const task = append(this.running, this.pending.shift());
            this.logger(`Attempting id=%s`, task.id);
            task.done(() => {
              this.logger(`Completing id=`, task.id);
              remove(this.running, task);
              this.schedule();
            });
          }
          next() {
            const { promise, id } = append(this.pending, createScheduledTask());
            this.logger(`Scheduling id=%s`, id);
            this.schedule();
            return promise;
          }
        };
      }
    });
    var apply_patch_exports = {};
    __export2(apply_patch_exports, {
      applyPatchTask: () => applyPatchTask
    });
    function applyPatchTask(patches, customArgs) {
      return straightThroughStringTask(["apply", ...customArgs, ...patches]);
    }
    var init_apply_patch = __esm2({
      "src/lib/tasks/apply-patch.ts"() {
        "use strict";
        init_task();
      }
    });
    function branchDeletionSuccess(branch, hash) {
      return {
        branch,
        hash,
        success: true
      };
    }
    function branchDeletionFailure(branch) {
      return {
        branch,
        hash: null,
        success: false
      };
    }
    var BranchDeletionBatch;
    var init_BranchDeleteSummary = __esm2({
      "src/lib/responses/BranchDeleteSummary.ts"() {
        "use strict";
        BranchDeletionBatch = class {
          constructor() {
            this.all = [];
            this.branches = {};
            this.errors = [];
          }
          get success() {
            return !this.errors.length;
          }
        };
      }
    });
    function hasBranchDeletionError(data, processExitCode) {
      return processExitCode === 1 && deleteErrorRegex.test(data);
    }
    var deleteSuccessRegex;
    var deleteErrorRegex;
    var parsers8;
    var parseBranchDeletions;
    var init_parse_branch_delete = __esm2({
      "src/lib/parsers/parse-branch-delete.ts"() {
        "use strict";
        init_BranchDeleteSummary();
        init_utils();
        deleteSuccessRegex = /(\S+)\s+\(\S+\s([^)]+)\)/;
        deleteErrorRegex = /^error[^']+'([^']+)'/m;
        parsers8 = [
          new LineParser(deleteSuccessRegex, (result, [branch, hash]) => {
            const deletion = branchDeletionSuccess(branch, hash);
            result.all.push(deletion);
            result.branches[branch] = deletion;
          }),
          new LineParser(deleteErrorRegex, (result, [branch]) => {
            const deletion = branchDeletionFailure(branch);
            result.errors.push(deletion);
            result.all.push(deletion);
            result.branches[branch] = deletion;
          })
        ];
        parseBranchDeletions = (stdOut, stdErr) => {
          return parseStringResponse(new BranchDeletionBatch(), parsers8, [stdOut, stdErr]);
        };
      }
    });
    var BranchSummaryResult;
    var init_BranchSummary = __esm2({
      "src/lib/responses/BranchSummary.ts"() {
        "use strict";
        BranchSummaryResult = class {
          constructor() {
            this.all = [];
            this.branches = {};
            this.current = "";
            this.detached = false;
          }
          push(status, detached, name, commit, label) {
            if (status === "*") {
              this.detached = detached;
              this.current = name;
            }
            this.all.push(name);
            this.branches[name] = {
              current: status === "*",
              linkedWorkTree: status === "+",
              name,
              commit,
              label
            };
          }
        };
      }
    });
    function branchStatus(input) {
      return input ? input.charAt(0) : "";
    }
    function parseBranchSummary(stdOut, currentOnly = false) {
      return parseStringResponse(
        new BranchSummaryResult(),
        currentOnly ? [currentBranchParser] : parsers9,
        stdOut
      );
    }
    var parsers9;
    var currentBranchParser;
    var init_parse_branch = __esm2({
      "src/lib/parsers/parse-branch.ts"() {
        "use strict";
        init_BranchSummary();
        init_utils();
        parsers9 = [
          new LineParser(
            /^([*+]\s)?\((?:HEAD )?detached (?:from|at) (\S+)\)\s+([a-z0-9]+)\s(.*)$/,
            (result, [current, name, commit, label]) => {
              result.push(branchStatus(current), true, name, commit, label);
            }
          ),
          new LineParser(
            /^([*+]\s)?(\S+)\s+([a-z0-9]+)\s?(.*)$/s,
            (result, [current, name, commit, label]) => {
              result.push(branchStatus(current), false, name, commit, label);
            }
          )
        ];
        currentBranchParser = new LineParser(/^(\S+)$/s, (result, [name]) => {
          result.push("*", false, name, "", "");
        });
      }
    });
    var branch_exports = {};
    __export2(branch_exports, {
      branchLocalTask: () => branchLocalTask,
      branchTask: () => branchTask,
      containsDeleteBranchCommand: () => containsDeleteBranchCommand,
      deleteBranchTask: () => deleteBranchTask,
      deleteBranchesTask: () => deleteBranchesTask
    });
    function containsDeleteBranchCommand(commands) {
      const deleteCommands = ["-d", "-D", "--delete"];
      return commands.some((command) => deleteCommands.includes(command));
    }
    function branchTask(customArgs) {
      const isDelete = containsDeleteBranchCommand(customArgs);
      const isCurrentOnly = customArgs.includes("--show-current");
      const commands = ["branch", ...customArgs];
      if (commands.length === 1) {
        commands.push("-a");
      }
      if (!commands.includes("-v")) {
        commands.splice(1, 0, "-v");
      }
      return {
        format: "utf-8",
        commands,
        parser(stdOut, stdErr) {
          if (isDelete) {
            return parseBranchDeletions(stdOut, stdErr).all[0];
          }
          return parseBranchSummary(stdOut, isCurrentOnly);
        }
      };
    }
    function branchLocalTask() {
      return {
        format: "utf-8",
        commands: ["branch", "-v"],
        parser(stdOut) {
          return parseBranchSummary(stdOut);
        }
      };
    }
    function deleteBranchesTask(branches, forceDelete = false) {
      return {
        format: "utf-8",
        commands: ["branch", "-v", forceDelete ? "-D" : "-d", ...branches],
        parser(stdOut, stdErr) {
          return parseBranchDeletions(stdOut, stdErr);
        },
        onError({ exitCode, stdOut }, error, done, fail) {
          if (!hasBranchDeletionError(String(error), exitCode)) {
            return fail(error);
          }
          done(stdOut);
        }
      };
    }
    function deleteBranchTask(branch, forceDelete = false) {
      const task = {
        format: "utf-8",
        commands: ["branch", "-v", forceDelete ? "-D" : "-d", branch],
        parser(stdOut, stdErr) {
          return parseBranchDeletions(stdOut, stdErr).branches[branch];
        },
        onError({ exitCode, stdErr, stdOut }, error, _, fail) {
          if (!hasBranchDeletionError(String(error), exitCode)) {
            return fail(error);
          }
          throw new GitResponseError(
            task.parser(bufferToString(stdOut), bufferToString(stdErr)),
            String(error)
          );
        }
      };
      return task;
    }
    var init_branch = __esm2({
      "src/lib/tasks/branch.ts"() {
        "use strict";
        init_git_response_error();
        init_parse_branch_delete();
        init_parse_branch();
        init_utils();
      }
    });
    function toPath(input) {
      const path2 = input.trim().replace(/^["']|["']$/g, "");
      return path2 && (0, import_node_path.normalize)(path2);
    }
    var import_node_path;
    var parseCheckIgnore;
    var init_CheckIgnore = __esm2({
      "src/lib/responses/CheckIgnore.ts"() {
        "use strict";
        import_node_path = require("node:path");
        parseCheckIgnore = (text) => {
          return text.split(/\n/g).map(toPath).filter(Boolean);
        };
      }
    });
    var check_ignore_exports = {};
    __export2(check_ignore_exports, {
      checkIgnoreTask: () => checkIgnoreTask
    });
    function checkIgnoreTask(paths) {
      return {
        commands: ["check-ignore", ...paths],
        format: "utf-8",
        parser: parseCheckIgnore
      };
    }
    var init_check_ignore = __esm2({
      "src/lib/tasks/check-ignore.ts"() {
        "use strict";
        init_CheckIgnore();
      }
    });
    function parseFetchResult(stdOut, stdErr) {
      const result = {
        raw: stdOut,
        remote: null,
        branches: [],
        tags: [],
        updated: [],
        deleted: []
      };
      return parseStringResponse(result, parsers10, [stdOut, stdErr]);
    }
    var parsers10;
    var init_parse_fetch = __esm2({
      "src/lib/parsers/parse-fetch.ts"() {
        "use strict";
        init_utils();
        parsers10 = [
          new LineParser(/From (.+)$/, (result, [remote]) => {
            result.remote = remote;
          }),
          new LineParser(/\* \[new branch]\s+(\S+)\s*-> (.+)$/, (result, [name, tracking]) => {
            result.branches.push({
              name,
              tracking
            });
          }),
          new LineParser(/\* \[new tag]\s+(\S+)\s*-> (.+)$/, (result, [name, tracking]) => {
            result.tags.push({
              name,
              tracking
            });
          }),
          new LineParser(/- \[deleted]\s+\S+\s*-> (.+)$/, (result, [tracking]) => {
            result.deleted.push({
              tracking
            });
          }),
          new LineParser(
            /\s*([^.]+)\.\.(\S+)\s+(\S+)\s*-> (.+)$/,
            (result, [from, to, name, tracking]) => {
              result.updated.push({
                name,
                tracking,
                to,
                from
              });
            }
          )
        ];
      }
    });
    var fetch_exports = {};
    __export2(fetch_exports, {
      fetchTask: () => fetchTask
    });
    function disallowedCommand(command) {
      return /^--upload-pack(=|$)/.test(command);
    }
    function fetchTask(remote, branch, customArgs) {
      const commands = ["fetch", ...customArgs];
      if (remote && branch) {
        commands.push(remote, branch);
      }
      const banned = commands.find(disallowedCommand);
      if (banned) {
        return configurationErrorTask(`git.fetch: potential exploit argument blocked.`);
      }
      return {
        commands,
        format: "utf-8",
        parser: parseFetchResult
      };
    }
    var init_fetch = __esm2({
      "src/lib/tasks/fetch.ts"() {
        "use strict";
        init_parse_fetch();
        init_task();
      }
    });
    function parseMoveResult(stdOut) {
      return parseStringResponse({ moves: [] }, parsers11, stdOut);
    }
    var parsers11;
    var init_parse_move = __esm2({
      "src/lib/parsers/parse-move.ts"() {
        "use strict";
        init_utils();
        parsers11 = [
          new LineParser(/^Renaming (.+) to (.+)$/, (result, [from, to]) => {
            result.moves.push({ from, to });
          })
        ];
      }
    });
    var move_exports = {};
    __export2(move_exports, {
      moveTask: () => moveTask
    });
    function moveTask(from, to) {
      return {
        commands: ["mv", "-v", ...asArray(from), to],
        format: "utf-8",
        parser: parseMoveResult
      };
    }
    var init_move = __esm2({
      "src/lib/tasks/move.ts"() {
        "use strict";
        init_parse_move();
        init_utils();
      }
    });
    var pull_exports = {};
    __export2(pull_exports, {
      pullTask: () => pullTask
    });
    function pullTask(remote, branch, customArgs) {
      const commands = ["pull", ...customArgs];
      if (remote && branch) {
        commands.splice(1, 0, remote, branch);
      }
      return {
        commands,
        format: "utf-8",
        parser(stdOut, stdErr) {
          return parsePullResult(stdOut, stdErr);
        },
        onError(result, _error, _done, fail) {
          const pullError = parsePullErrorResult(
            bufferToString(result.stdOut),
            bufferToString(result.stdErr)
          );
          if (pullError) {
            return fail(new GitResponseError(pullError));
          }
          fail(_error);
        }
      };
    }
    var init_pull = __esm2({
      "src/lib/tasks/pull.ts"() {
        "use strict";
        init_git_response_error();
        init_parse_pull();
        init_utils();
      }
    });
    function parseGetRemotes(text) {
      const remotes = {};
      forEach(text, ([name]) => remotes[name] = { name });
      return Object.values(remotes);
    }
    function parseGetRemotesVerbose(text) {
      const remotes = {};
      forEach(text, ([name, url, purpose]) => {
        if (!Object.hasOwn(remotes, name)) {
          remotes[name] = {
            name,
            refs: { fetch: "", push: "" }
          };
        }
        if (purpose && url) {
          remotes[name].refs[purpose.replace(/[^a-z]/g, "")] = url;
        }
      });
      return Object.values(remotes);
    }
    function forEach(text, handler) {
      forEachLineWithContent(text, (line) => handler(line.split(/\s+/)));
    }
    var init_GetRemoteSummary = __esm2({
      "src/lib/responses/GetRemoteSummary.ts"() {
        "use strict";
        init_utils();
      }
    });
    var remote_exports = {};
    __export2(remote_exports, {
      addRemoteTask: () => addRemoteTask,
      getRemotesTask: () => getRemotesTask,
      listRemotesTask: () => listRemotesTask,
      remoteTask: () => remoteTask,
      removeRemoteTask: () => removeRemoteTask
    });
    function addRemoteTask(remoteName, remoteRepo, customArgs) {
      return straightThroughStringTask(["remote", "add", ...customArgs, remoteName, remoteRepo]);
    }
    function getRemotesTask(verbose) {
      const commands = ["remote"];
      if (verbose) {
        commands.push("-v");
      }
      return {
        commands,
        format: "utf-8",
        parser: verbose ? parseGetRemotesVerbose : parseGetRemotes
      };
    }
    function listRemotesTask(customArgs) {
      const commands = [...customArgs];
      if (commands[0] !== "ls-remote") {
        commands.unshift("ls-remote");
      }
      return straightThroughStringTask(commands);
    }
    function remoteTask(customArgs) {
      const commands = [...customArgs];
      if (commands[0] !== "remote") {
        commands.unshift("remote");
      }
      return straightThroughStringTask(commands);
    }
    function removeRemoteTask(remoteName) {
      return straightThroughStringTask(["remote", "remove", remoteName]);
    }
    var init_remote = __esm2({
      "src/lib/tasks/remote.ts"() {
        "use strict";
        init_GetRemoteSummary();
        init_task();
      }
    });
    var stash_list_exports = {};
    __export2(stash_list_exports, {
      stashListTask: () => stashListTask
    });
    function stashListTask(opt = {}, customArgs) {
      const options = parseLogOptions(opt);
      const commands = ["stash", "list", ...options.commands, ...customArgs];
      const parser4 = createListLogSummaryParser(
        options.splitter,
        options.fields,
        logFormatFromCommand(commands)
      );
      return validateLogFormatConfig(commands) || {
        commands,
        format: "utf-8",
        parser: parser4
      };
    }
    var init_stash_list = __esm2({
      "src/lib/tasks/stash-list.ts"() {
        "use strict";
        init_log_format();
        init_parse_list_log_summary();
        init_diff();
        init_log();
      }
    });
    var sub_module_exports = {};
    __export2(sub_module_exports, {
      addSubModuleTask: () => addSubModuleTask,
      initSubModuleTask: () => initSubModuleTask,
      subModuleTask: () => subModuleTask,
      updateSubModuleTask: () => updateSubModuleTask
    });
    function addSubModuleTask(repo, path2) {
      return subModuleTask(["add", repo, path2]);
    }
    function initSubModuleTask(customArgs) {
      return subModuleTask(["init", ...customArgs]);
    }
    function subModuleTask(customArgs) {
      const commands = [...customArgs];
      if (commands[0] !== "submodule") {
        commands.unshift("submodule");
      }
      return straightThroughStringTask(commands);
    }
    function updateSubModuleTask(customArgs) {
      return subModuleTask(["update", ...customArgs]);
    }
    var init_sub_module = __esm2({
      "src/lib/tasks/sub-module.ts"() {
        "use strict";
        init_task();
      }
    });
    function singleSorted(a, b) {
      const aIsNum = Number.isNaN(a);
      const bIsNum = Number.isNaN(b);
      if (aIsNum !== bIsNum) {
        return aIsNum ? 1 : -1;
      }
      return aIsNum ? sorted(a, b) : 0;
    }
    function sorted(a, b) {
      return a === b ? 0 : a > b ? 1 : -1;
    }
    function trimmed(input) {
      return input.trim();
    }
    function toNumber(input) {
      if (typeof input === "string") {
        return parseInt(input.replace(/^\D+/g, ""), 10) || 0;
      }
      return 0;
    }
    var TagList;
    var parseTagList;
    var init_TagList = __esm2({
      "src/lib/responses/TagList.ts"() {
        "use strict";
        TagList = class {
          constructor(all, latest) {
            this.all = all;
            this.latest = latest;
          }
        };
        parseTagList = function(data, customSort = false) {
          const tags = data.split("\n").map(trimmed).filter(Boolean);
          if (!customSort) {
            tags.sort(function(tagA, tagB) {
              const partsA = tagA.split(".");
              const partsB = tagB.split(".");
              if (partsA.length === 1 || partsB.length === 1) {
                return singleSorted(toNumber(partsA[0]), toNumber(partsB[0]));
              }
              for (let i = 0, l = Math.max(partsA.length, partsB.length); i < l; i++) {
                const diff = sorted(toNumber(partsA[i]), toNumber(partsB[i]));
                if (diff) {
                  return diff;
                }
              }
              return 0;
            });
          }
          const latest = customSort ? tags[0] : [...tags].reverse().find((tag) => tag.indexOf(".") >= 0);
          return new TagList(tags, latest);
        };
      }
    });
    var tag_exports = {};
    __export2(tag_exports, {
      addAnnotatedTagTask: () => addAnnotatedTagTask,
      addTagTask: () => addTagTask,
      tagListTask: () => tagListTask
    });
    function tagListTask(customArgs = []) {
      const hasCustomSort = customArgs.some((option) => /^--sort=/.test(option));
      return {
        format: "utf-8",
        commands: ["tag", "-l", ...customArgs],
        parser(text) {
          return parseTagList(text, hasCustomSort);
        }
      };
    }
    function addTagTask(name) {
      return {
        format: "utf-8",
        commands: ["tag", name],
        parser() {
          return { name };
        }
      };
    }
    function addAnnotatedTagTask(name, tagMessage) {
      return {
        format: "utf-8",
        commands: ["tag", "-a", "-m", tagMessage, name],
        parser() {
          return { name };
        }
      };
    }
    var init_tag = __esm2({
      "src/lib/tasks/tag.ts"() {
        "use strict";
        init_TagList();
      }
    });
    var require_git = __commonJS2({
      "src/git.js"(exports22, module22) {
        "use strict";
        var { GitExecutor: GitExecutor2 } = (init_git_executor(), __toCommonJS2(git_executor_exports));
        var { SimpleGitApi: SimpleGitApi2 } = (init_simple_git_api(), __toCommonJS2(simple_git_api_exports));
        var { Scheduler: Scheduler2 } = (init_scheduler(), __toCommonJS2(scheduler_exports));
        var { adhocExecTask: adhocExecTask2, configurationErrorTask: configurationErrorTask2 } = (init_task(), __toCommonJS2(task_exports));
        var {
          asArray: asArray2,
          filterArray: filterArray2,
          filterPrimitives: filterPrimitives2,
          filterString: filterString2,
          filterStringOrStringArray: filterStringOrStringArray2,
          filterType: filterType2,
          getTrailingOptions: getTrailingOptions2,
          trailingFunctionArgument: trailingFunctionArgument2,
          trailingOptionsArgument: trailingOptionsArgument2
        } = (init_utils(), __toCommonJS2(utils_exports));
        var { applyPatchTask: applyPatchTask2 } = (init_apply_patch(), __toCommonJS2(apply_patch_exports));
        var {
          branchTask: branchTask2,
          branchLocalTask: branchLocalTask2,
          deleteBranchesTask: deleteBranchesTask2,
          deleteBranchTask: deleteBranchTask2
        } = (init_branch(), __toCommonJS2(branch_exports));
        var { checkIgnoreTask: checkIgnoreTask2 } = (init_check_ignore(), __toCommonJS2(check_ignore_exports));
        var { checkIsRepoTask: checkIsRepoTask2 } = (init_check_is_repo(), __toCommonJS2(check_is_repo_exports));
        var { cleanWithOptionsTask: cleanWithOptionsTask2, isCleanOptionsArray: isCleanOptionsArray2 } = (init_clean(), __toCommonJS2(clean_exports));
        var { diffSummaryTask: diffSummaryTask2 } = (init_diff(), __toCommonJS2(diff_exports));
        var { fetchTask: fetchTask2 } = (init_fetch(), __toCommonJS2(fetch_exports));
        var { moveTask: moveTask2 } = (init_move(), __toCommonJS2(move_exports));
        var { pullTask: pullTask2 } = (init_pull(), __toCommonJS2(pull_exports));
        var { pushTagsTask: pushTagsTask2 } = (init_push(), __toCommonJS2(push_exports));
        var {
          addRemoteTask: addRemoteTask2,
          getRemotesTask: getRemotesTask2,
          listRemotesTask: listRemotesTask2,
          remoteTask: remoteTask2,
          removeRemoteTask: removeRemoteTask2
        } = (init_remote(), __toCommonJS2(remote_exports));
        var { getResetMode: getResetMode2, resetTask: resetTask2 } = (init_reset(), __toCommonJS2(reset_exports));
        var { stashListTask: stashListTask2 } = (init_stash_list(), __toCommonJS2(stash_list_exports));
        var {
          addSubModuleTask: addSubModuleTask2,
          initSubModuleTask: initSubModuleTask2,
          subModuleTask: subModuleTask2,
          updateSubModuleTask: updateSubModuleTask2
        } = (init_sub_module(), __toCommonJS2(sub_module_exports));
        var { addAnnotatedTagTask: addAnnotatedTagTask2, addTagTask: addTagTask2, tagListTask: tagListTask2 } = (init_tag(), __toCommonJS2(tag_exports));
        var { straightThroughBufferTask: straightThroughBufferTask2, straightThroughStringTask: straightThroughStringTask2 } = (init_task(), __toCommonJS2(task_exports));
        function Git2(options, plugins) {
          this._plugins = plugins;
          this._executor = new GitExecutor2(
            options.baseDir,
            new Scheduler2(options.maxConcurrentProcesses),
            plugins
          );
          this._trimmed = options.trimmed;
        }
        (Git2.prototype = Object.create(SimpleGitApi2.prototype)).constructor = Git2;
        Git2.prototype.customBinary = function(command) {
          this._plugins.reconfigure("binary", command);
          return this;
        };
        Git2.prototype.env = function(name, value) {
          if (arguments.length === 1 && typeof name === "object") {
            this._executor.env = name;
          } else {
            (this._executor.env = this._executor.env || {})[name] = value;
          }
          return this;
        };
        Git2.prototype.stashList = function(options) {
          return this._runTask(
            stashListTask2(
              trailingOptionsArgument2(arguments) || {},
              filterArray2(options) && options || []
            ),
            trailingFunctionArgument2(arguments)
          );
        };
        Git2.prototype.mv = function(from, to) {
          return this._runTask(moveTask2(from, to), trailingFunctionArgument2(arguments));
        };
        Git2.prototype.checkoutLatestTag = function(then) {
          var git = this;
          return this.pull(function() {
            git.tags(function(err, tags) {
              git.checkout(tags.latest, then);
            });
          });
        };
        Git2.prototype.pull = function(remote, branch, options, then) {
          return this._runTask(
            pullTask2(
              filterType2(remote, filterString2),
              filterType2(branch, filterString2),
              getTrailingOptions2(arguments)
            ),
            trailingFunctionArgument2(arguments)
          );
        };
        Git2.prototype.fetch = function(remote, branch) {
          return this._runTask(
            fetchTask2(
              filterType2(remote, filterString2),
              filterType2(branch, filterString2),
              getTrailingOptions2(arguments)
            ),
            trailingFunctionArgument2(arguments)
          );
        };
        Git2.prototype.silent = function(silence) {
          return this._runTask(
            adhocExecTask2(
              () => console.warn(
                "simple-git deprecation notice: git.silent: logging should be configured using the `debug` library / `DEBUG` environment variable, this method will be removed."
              )
            )
          );
        };
        Git2.prototype.tags = function(options, then) {
          return this._runTask(
            tagListTask2(getTrailingOptions2(arguments)),
            trailingFunctionArgument2(arguments)
          );
        };
        Git2.prototype.rebase = function() {
          return this._runTask(
            straightThroughStringTask2(["rebase", ...getTrailingOptions2(arguments)]),
            trailingFunctionArgument2(arguments)
          );
        };
        Git2.prototype.reset = function(mode) {
          return this._runTask(
            resetTask2(getResetMode2(mode), getTrailingOptions2(arguments)),
            trailingFunctionArgument2(arguments)
          );
        };
        Git2.prototype.revert = function(commit) {
          const next = trailingFunctionArgument2(arguments);
          if (typeof commit !== "string") {
            return this._runTask(configurationErrorTask2("Commit must be a string"), next);
          }
          return this._runTask(
            straightThroughStringTask2(["revert", ...getTrailingOptions2(arguments, 0, true), commit]),
            next
          );
        };
        Git2.prototype.addTag = function(name) {
          const task = typeof name === "string" ? addTagTask2(name) : configurationErrorTask2("Git.addTag requires a tag name");
          return this._runTask(task, trailingFunctionArgument2(arguments));
        };
        Git2.prototype.addAnnotatedTag = function(tagName, tagMessage) {
          return this._runTask(
            addAnnotatedTagTask2(tagName, tagMessage),
            trailingFunctionArgument2(arguments)
          );
        };
        Git2.prototype.deleteLocalBranch = function(branchName, forceDelete, then) {
          return this._runTask(
            deleteBranchTask2(branchName, typeof forceDelete === "boolean" ? forceDelete : false),
            trailingFunctionArgument2(arguments)
          );
        };
        Git2.prototype.deleteLocalBranches = function(branchNames, forceDelete, then) {
          return this._runTask(
            deleteBranchesTask2(branchNames, typeof forceDelete === "boolean" ? forceDelete : false),
            trailingFunctionArgument2(arguments)
          );
        };
        Git2.prototype.branch = function(options, then) {
          return this._runTask(
            branchTask2(getTrailingOptions2(arguments)),
            trailingFunctionArgument2(arguments)
          );
        };
        Git2.prototype.branchLocal = function(then) {
          return this._runTask(branchLocalTask2(), trailingFunctionArgument2(arguments));
        };
        Git2.prototype.raw = function(commands) {
          const createRestCommands = !Array.isArray(commands);
          const command = [].slice.call(createRestCommands ? arguments : commands, 0);
          for (let i = 0; i < command.length && createRestCommands; i++) {
            if (!filterPrimitives2(command[i])) {
              command.splice(i, command.length - i);
              break;
            }
          }
          command.push(...getTrailingOptions2(arguments, 0, true));
          var next = trailingFunctionArgument2(arguments);
          if (!command.length) {
            return this._runTask(
              configurationErrorTask2("Raw: must supply one or more command to execute"),
              next
            );
          }
          return this._runTask(straightThroughStringTask2(command, this._trimmed), next);
        };
        Git2.prototype.submoduleAdd = function(repo, path2, then) {
          return this._runTask(addSubModuleTask2(repo, path2), trailingFunctionArgument2(arguments));
        };
        Git2.prototype.submoduleUpdate = function(args, then) {
          return this._runTask(
            updateSubModuleTask2(getTrailingOptions2(arguments, true)),
            trailingFunctionArgument2(arguments)
          );
        };
        Git2.prototype.submoduleInit = function(args, then) {
          return this._runTask(
            initSubModuleTask2(getTrailingOptions2(arguments, true)),
            trailingFunctionArgument2(arguments)
          );
        };
        Git2.prototype.subModule = function(options, then) {
          return this._runTask(
            subModuleTask2(getTrailingOptions2(arguments)),
            trailingFunctionArgument2(arguments)
          );
        };
        Git2.prototype.listRemote = function() {
          return this._runTask(
            listRemotesTask2(getTrailingOptions2(arguments)),
            trailingFunctionArgument2(arguments)
          );
        };
        Git2.prototype.addRemote = function(remoteName, remoteRepo, then) {
          return this._runTask(
            addRemoteTask2(remoteName, remoteRepo, getTrailingOptions2(arguments)),
            trailingFunctionArgument2(arguments)
          );
        };
        Git2.prototype.removeRemote = function(remoteName, then) {
          return this._runTask(removeRemoteTask2(remoteName), trailingFunctionArgument2(arguments));
        };
        Git2.prototype.getRemotes = function(verbose, then) {
          return this._runTask(getRemotesTask2(verbose === true), trailingFunctionArgument2(arguments));
        };
        Git2.prototype.remote = function(options, then) {
          return this._runTask(
            remoteTask2(getTrailingOptions2(arguments)),
            trailingFunctionArgument2(arguments)
          );
        };
        Git2.prototype.tag = function(options, then) {
          const command = getTrailingOptions2(arguments);
          if (command[0] !== "tag") {
            command.unshift("tag");
          }
          return this._runTask(straightThroughStringTask2(command), trailingFunctionArgument2(arguments));
        };
        Git2.prototype.updateServerInfo = function(then) {
          return this._runTask(
            straightThroughStringTask2(["update-server-info"]),
            trailingFunctionArgument2(arguments)
          );
        };
        Git2.prototype.pushTags = function(remote, then) {
          const task = pushTagsTask2(
            { remote: filterType2(remote, filterString2) },
            getTrailingOptions2(arguments)
          );
          return this._runTask(task, trailingFunctionArgument2(arguments));
        };
        Git2.prototype.rm = function(files) {
          return this._runTask(
            straightThroughStringTask2(["rm", "-f", ...asArray2(files)]),
            trailingFunctionArgument2(arguments)
          );
        };
        Git2.prototype.rmKeepLocal = function(files) {
          return this._runTask(
            straightThroughStringTask2(["rm", "--cached", ...asArray2(files)]),
            trailingFunctionArgument2(arguments)
          );
        };
        Git2.prototype.catFile = function(options, then) {
          return this._catFile("utf-8", arguments);
        };
        Git2.prototype.binaryCatFile = function() {
          return this._catFile("buffer", arguments);
        };
        Git2.prototype._catFile = function(format, args) {
          var handler = trailingFunctionArgument2(args);
          var command = ["cat-file"];
          var options = args[0];
          if (typeof options === "string") {
            return this._runTask(
              configurationErrorTask2("Git.catFile: options must be supplied as an array of strings"),
              handler
            );
          }
          if (Array.isArray(options)) {
            command.push.apply(command, options);
          }
          const task = format === "buffer" ? straightThroughBufferTask2(command) : straightThroughStringTask2(command);
          return this._runTask(task, handler);
        };
        Git2.prototype.diff = function(options, then) {
          const task = filterString2(options) ? configurationErrorTask2(
            "git.diff: supplying options as a single string is no longer supported, switch to an array of strings"
          ) : straightThroughStringTask2(["diff", ...getTrailingOptions2(arguments)]);
          return this._runTask(task, trailingFunctionArgument2(arguments));
        };
        Git2.prototype.diffSummary = function() {
          return this._runTask(
            diffSummaryTask2(getTrailingOptions2(arguments, 1)),
            trailingFunctionArgument2(arguments)
          );
        };
        Git2.prototype.applyPatch = function(patches) {
          const task = !filterStringOrStringArray2(patches) ? configurationErrorTask2(
            `git.applyPatch requires one or more string patches as the first argument`
          ) : applyPatchTask2(asArray2(patches), getTrailingOptions2([].slice.call(arguments, 1)));
          return this._runTask(task, trailingFunctionArgument2(arguments));
        };
        Git2.prototype.revparse = function() {
          const commands = ["rev-parse", ...getTrailingOptions2(arguments, true)];
          return this._runTask(
            straightThroughStringTask2(commands, true),
            trailingFunctionArgument2(arguments)
          );
        };
        Git2.prototype.clean = function(mode, options, then) {
          const usingCleanOptionsArray = isCleanOptionsArray2(mode);
          const cleanMode = usingCleanOptionsArray && mode.join("") || filterType2(mode, filterString2) || "";
          const customArgs = getTrailingOptions2([].slice.call(arguments, usingCleanOptionsArray ? 1 : 0));
          return this._runTask(
            cleanWithOptionsTask2(cleanMode, customArgs),
            trailingFunctionArgument2(arguments)
          );
        };
        Git2.prototype.exec = function(then) {
          const task = {
            commands: [],
            format: "utf-8",
            parser() {
              if (typeof then === "function") {
                then();
              }
            }
          };
          return this._runTask(task);
        };
        Git2.prototype.clearQueue = function() {
          return this._runTask(
            adhocExecTask2(
              () => console.warn(
                "simple-git deprecation notice: clearQueue() is deprecated and will be removed, switch to using the abortPlugin instead."
              )
            )
          );
        };
        Git2.prototype.checkIgnore = function(pathnames, then) {
          return this._runTask(
            checkIgnoreTask2(asArray2(filterType2(pathnames, filterStringOrStringArray2, []))),
            trailingFunctionArgument2(arguments)
          );
        };
        Git2.prototype.checkIsRepo = function(checkType, then) {
          return this._runTask(
            checkIsRepoTask2(filterType2(checkType, filterString2)),
            trailingFunctionArgument2(arguments)
          );
        };
        module22.exports = Git2;
      }
    });
    var git_factory_exports = {};
    __export2(git_factory_exports, {
      esModuleFactory: () => esModuleFactory,
      gitExportFactory: () => gitExportFactory,
      gitInstanceFactory: () => gitInstanceFactory
    });
    function esModuleFactory(defaultExport) {
      return Object.defineProperties(defaultExport, {
        __esModule: { value: true },
        default: { value: defaultExport }
      });
    }
    function gitExportFactory(factory) {
      return Object.assign(factory.bind(null), api_exports);
    }
    function gitInstanceFactory(baseDir, options) {
      const plugins = new PluginStore();
      const config = createInstanceConfig(
        baseDir && (typeof baseDir === "string" ? { baseDir } : baseDir) || {},
        options
      );
      if (!folderExists(config.baseDir)) {
        throw new GitConstructError(
          config,
          `Cannot use simple-git on a directory that does not exist`
        );
      }
      if (Array.isArray(config.config)) {
        plugins.add(commandConfigPrefixingPlugin(config.config));
      }
      plugins.add(blockUnsafeOperationsPlugin(config.unsafe));
      plugins.add(completionDetectionPlugin(config.completion));
      config.abort && plugins.add(abortPlugin(config.abort));
      config.progress && plugins.add(progressMonitorPlugin(config.progress));
      config.timeout && plugins.add(timeoutPlugin(config.timeout));
      config.spawnOptions && plugins.add(spawnOptionsPlugin(config.spawnOptions));
      plugins.add(suffixPathsPlugin());
      plugins.add(errorDetectionPlugin(errorDetectionHandler(true)));
      config.errors && plugins.add(errorDetectionPlugin(config.errors));
      customBinaryPlugin(plugins, config.binary, config.unsafe?.allowUnsafeCustomBinary);
      return new Git(config, plugins);
    }
    var Git;
    var init_git_factory = __esm2({
      "src/lib/git-factory.ts"() {
        "use strict";
        init_api();
        init_plugins();
        init_suffix_paths_plugin();
        init_utils();
        Git = require_git();
      }
    });
    var promise_wrapped_exports = {};
    __export2(promise_wrapped_exports, {
      gitP: () => gitP
    });
    function gitP(...args) {
      let git;
      let chain = Promise.resolve();
      try {
        git = gitInstanceFactory(...args);
      } catch (e) {
        chain = Promise.reject(e);
      }
      function builderReturn() {
        return promiseApi;
      }
      function chainReturn() {
        return chain;
      }
      const promiseApi = [...functionNamesBuilderApi, ...functionNamesPromiseApi].reduce(
        (api, name) => {
          const isAsync = functionNamesPromiseApi.includes(name);
          const valid = isAsync ? asyncWrapper(name, git) : syncWrapper(name, git, api);
          const alternative = isAsync ? chainReturn : builderReturn;
          Object.defineProperty(api, name, {
            enumerable: false,
            configurable: false,
            value: git ? valid : alternative
          });
          return api;
        },
        {}
      );
      return promiseApi;
      function asyncWrapper(fn, git2) {
        return function(...args2) {
          if (typeof args2[args2.length] === "function") {
            throw new TypeError(
              "Promise interface requires that handlers are not supplied inline, trailing function not allowed in call to " + fn
            );
          }
          return chain.then(function() {
            return new Promise(function(resolve, reject) {
              const callback = (err, result) => {
                if (err) {
                  return reject(toError(err));
                }
                resolve(result);
              };
              args2.push(callback);
              git2[fn].apply(git2, args2);
            });
          });
        };
      }
      function syncWrapper(fn, git2, api) {
        return (...args2) => {
          git2[fn](...args2);
          return api;
        };
      }
    }
    function toError(error) {
      if (error instanceof Error) {
        return error;
      }
      if (typeof error === "string") {
        return new Error(error);
      }
      return new GitResponseError(error);
    }
    var functionNamesBuilderApi;
    var functionNamesPromiseApi;
    var init_promise_wrapped = __esm2({
      "src/lib/runners/promise-wrapped.ts"() {
        "use strict";
        init_git_response_error();
        init_git_factory();
        functionNamesBuilderApi = ["customBinary", "env", "outputHandler", "silent"];
        functionNamesPromiseApi = [
          "add",
          "addAnnotatedTag",
          "addConfig",
          "addRemote",
          "addTag",
          "applyPatch",
          "binaryCatFile",
          "branch",
          "branchLocal",
          "catFile",
          "checkIgnore",
          "checkIsRepo",
          "checkout",
          "checkoutBranch",
          "checkoutLatestTag",
          "checkoutLocalBranch",
          "clean",
          "clone",
          "commit",
          "cwd",
          "deleteLocalBranch",
          "deleteLocalBranches",
          "diff",
          "diffSummary",
          "exec",
          "fetch",
          "getRemotes",
          "init",
          "listConfig",
          "listRemote",
          "log",
          "merge",
          "mergeFromTo",
          "mirror",
          "mv",
          "pull",
          "push",
          "pushTags",
          "raw",
          "rebase",
          "remote",
          "removeRemote",
          "reset",
          "revert",
          "revparse",
          "rm",
          "rmKeepLocal",
          "show",
          "stash",
          "stashList",
          "status",
          "subModule",
          "submoduleAdd",
          "submoduleInit",
          "submoduleUpdate",
          "tag",
          "tags",
          "updateServerInfo"
        ];
      }
    });
    var { gitP: gitP2 } = (init_promise_wrapped(), __toCommonJS2(promise_wrapped_exports));
    var { esModuleFactory: esModuleFactory2, gitInstanceFactory: gitInstanceFactory2, gitExportFactory: gitExportFactory2 } = (init_git_factory(), __toCommonJS2(git_factory_exports));
    var simpleGit = esModuleFactory2(gitExportFactory2(gitInstanceFactory2));
    module2.exports = Object.assign(simpleGit, { gitP: gitP2, simpleGit });
  }
});

// electron/git-review-ops.cjs
var require_git_review_ops = __commonJS({
  "electron/git-review-ops.cjs"(exports2, module2) {
    "use strict";
    var { execFile } = require("node:child_process");
    var fs2 = require("node:fs/promises");
    var path2 = require("node:path");
    var simpleGit = require_cjs();
    var { resolveRequestedPathForIpc: resolveRequestedPathForIpc2 } = require_hardening();
    var COMMIT_CONTEXT_DIFF_MAX_CHARS = 12e4;
    var COMMIT_CONTEXT_UNTRACKED_MAX = 80;
    var UNTRACKED_LINE_COUNT_CONCURRENCY = 16;
    var UNTRACKED_LINE_COUNT_MAX_BYTES = 1024 * 1024;
    function ghEnv(ghBin) {
      const extra = [ghBin ? path2.dirname(ghBin) : "", "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"].filter(
        (dir) => dir && dir !== "."
      );
      return { ...process.env, PATH: [...extra, process.env.PATH].filter(Boolean).join(path2.delimiter) };
    }
    function runGh(args, cwd, ghBin) {
      return new Promise((resolve) => {
        execFile(
          ghBin || "gh",
          args,
          { cwd, env: ghEnv(ghBin), windowsHide: true, timeout: 3e4, maxBuffer: 8 * 1024 * 1024 },
          (err, stdout) => resolve({ ok: !err, stdout: String(stdout || "") })
        );
      });
    }
    function gitFor(cwd, gitBin) {
      return simpleGit({ baseDir: cwd, binary: gitBin || "git", maxConcurrentProcesses: 4, trimmed: false });
    }
    function resolveRenamePath(raw) {
      const path3 = String(raw || "").trim();
      if (!path3.includes(" => ")) {
        return path3;
      }
      const brace = path3.match(/^(.*)\{(.*) => (.*)\}(.*)$/);
      if (brace) {
        const [, prefix, , to, suffix] = brace;
        return `${prefix}${to}${suffix}`.replace(/\/{2,}/g, "/");
      }
      return path3.split(" => ").pop().trim();
    }
    function countsByPath(summary) {
      const map = /* @__PURE__ */ new Map();
      for (const file of summary.files) {
        map.set(resolveRenamePath(file.file), {
          added: file.binary ? 0 : file.insertions,
          removed: file.binary ? 0 : file.deletions
        });
      }
      return map;
    }
    async function untrackedInsertions(cwd, relPath) {
      try {
        const fullPath = path2.join(cwd, relPath);
        const stat = await fs2.stat(fullPath);
        if (!stat.isFile() || stat.size > UNTRACKED_LINE_COUNT_MAX_BYTES) {
          return 0;
        }
        const buf = await fs2.readFile(fullPath);
        if (buf.includes(0)) {
          return 0;
        }
        let lines = 0;
        for (const byte of buf) {
          if (byte === 10) {
            lines++;
          }
        }
        return buf.length > 0 && buf[buf.length - 1] !== 10 ? lines + 1 : lines;
      } catch {
        return 0;
      }
    }
    function capText(text, maxChars, label = "truncated") {
      const value = String(text || "");
      if (value.length <= maxChars) {
        return value;
      }
      return `${value.slice(0, maxChars)}
# ${label}: ${value.length - maxChars} chars omitted
`;
    }
    async function fillUntrackedCounts(cwd, files) {
      const pending = files.filter((file) => file.status === "?" && file.added === 0 && file.removed === 0);
      for (let i = 0; i < pending.length; i += UNTRACKED_LINE_COUNT_CONCURRENCY) {
        await Promise.all(
          pending.slice(i, i + UNTRACKED_LINE_COUNT_CONCURRENCY).map(async (file) => {
            file.added = await untrackedInsertions(cwd, file.path);
          })
        );
      }
    }
    async function branchBase(git) {
      const candidates = [];
      try {
        const head = (await git.revparse(["--abbrev-ref", "origin/HEAD"])).trim();
        if (head) {
          candidates.push(head);
        }
      } catch {
      }
      candidates.push("origin/main", "origin/master", "main", "master");
      for (const ref of candidates) {
        try {
          const base = (await git.raw(["merge-base", "HEAD", ref])).trim();
          if (base) {
            return base;
          }
        } catch {
        }
      }
      return null;
    }
    async function defaultBranchName(git) {
      try {
        const head = (await git.revparse(["--abbrev-ref", "origin/HEAD"])).trim();
        if (head && head !== "origin/HEAD") {
          return head.replace(/^origin\//, "");
        }
      } catch {
      }
      for (const ref of ["refs/heads/main", "refs/heads/master", "refs/remotes/origin/main", "refs/remotes/origin/master"]) {
        try {
          await git.raw(["rev-parse", "--verify", "--quiet", ref]);
          return ref.replace(/^refs\/(?:heads|remotes\/origin)\//, "");
        } catch {
        }
      }
      return null;
    }
    function statusLetter(file) {
      if (file.index === "?" || file.working_dir === "?") {
        return "?";
      }
      const code = file.index && file.index !== " " ? file.index : file.working_dir;
      return (code || "M").toUpperCase();
    }
    var isStaged = (file) => Boolean(file.index && file.index !== " " && file.index !== "?");
    async function reviewList2(repoPath, scope, baseRef, gitBin) {
      let cwd;
      try {
        cwd = resolveRequestedPathForIpc2(repoPath, { purpose: "Review list" });
      } catch {
        return { files: [], base: null };
      }
      const git = gitFor(cwd, gitBin);
      try {
        if (scope === "branch" || scope === "lastTurn") {
          const base = scope === "branch" ? await branchBase(git) : baseRef;
          if (!base) {
            return { files: [], base: null };
          }
          const range = scope === "branch" ? `${base}...HEAD` : base;
          const summary = await git.diffSummary([range]);
          const files2 = summary.files.map((file) => ({
            path: resolveRenamePath(file.file),
            added: file.binary ? 0 : file.insertions,
            removed: file.binary ? 0 : file.deletions,
            status: "M",
            staged: false
          }));
          if (scope === "lastTurn") {
            const status2 = await git.status();
            for (const path3 of status2.not_added) {
              if (!files2.some((f) => f.path === path3)) {
                files2.push({ path: path3, added: 0, removed: 0, status: "?", staged: false });
              }
            }
          }
          files2.sort((a, b) => a.path.localeCompare(b.path));
          await fillUntrackedCounts(cwd, files2);
          return { files: files2, base };
        }
        const [status, staged, unstaged] = await Promise.all([
          git.status(),
          git.diffSummary(["--cached"]),
          git.diffSummary([])
        ]);
        const stagedCounts = countsByPath(staged);
        const unstagedCounts = countsByPath(unstaged);
        const files = status.files.map((file) => {
          const filePath = resolveRenamePath(file.path);
          const sc = stagedCounts.get(filePath) || { added: 0, removed: 0 };
          const uc = unstagedCounts.get(filePath) || { added: 0, removed: 0 };
          return {
            path: filePath,
            added: sc.added + uc.added,
            removed: sc.removed + uc.removed,
            status: statusLetter(file),
            staged: isStaged(file)
          };
        });
        files.sort((a, b) => a.path.localeCompare(b.path));
        await fillUntrackedCounts(cwd, files);
        return { files, base: null };
      } catch {
        return { files: [], base: null };
      }
    }
    async function reviewDiff2(repoPath, filePath, scope, baseRef, staged, gitBin) {
      let cwd;
      try {
        cwd = resolveRequestedPathForIpc2(repoPath, { purpose: "Review diff" });
      } catch {
        return "";
      }
      const git = gitFor(cwd, gitBin);
      const safe = (args) => git.diff(args).catch(() => "");
      if (scope === "branch") {
        const base = await branchBase(git);
        return base ? safe([`${base}...HEAD`, "--", filePath]) : "";
      }
      if (scope === "lastTurn") {
        return baseRef ? safe([baseRef, "--", filePath]) : "";
      }
      if (staged) {
        return safe(["--cached", "--", filePath]);
      }
      const worktree = await safe(["--", filePath]);
      if (worktree.trim()) {
        return worktree;
      }
      return new Promise((resolve) => {
        execFile(
          gitBin || "git",
          ["diff", "--no-index", "--", "/dev/null", filePath],
          { cwd, windowsHide: true, timeout: 3e4, maxBuffer: 32 * 1024 * 1024 },
          (_err, stdout) => resolve(String(stdout || ""))
        );
      });
    }
    async function fileDiffVsHead2(repoPath, filePath, gitBin) {
      let cwd;
      try {
        cwd = resolveRequestedPathForIpc2(repoPath, { purpose: "File diff" });
      } catch {
        return "";
      }
      const git = gitFor(cwd, gitBin);
      const head = await git.diff(["HEAD", "--", filePath]).catch(() => "");
      if (head.trim()) {
        return head;
      }
      const status = await git.raw(["status", "--porcelain", "--", filePath]).catch(() => "");
      if (!status.trim().startsWith("??")) {
        return "";
      }
      return new Promise((resolve) => {
        execFile(
          gitBin || "git",
          ["diff", "--no-index", "--", "/dev/null", filePath],
          { cwd, windowsHide: true, timeout: 3e4, maxBuffer: 32 * 1024 * 1024 },
          (_err, stdout) => resolve(String(stdout || ""))
        );
      });
    }
    async function reviewStage2(repoPath, filePath, gitBin) {
      const cwd = resolveRequestedPathForIpc2(repoPath, { purpose: "Review stage" });
      await gitFor(cwd, gitBin).raw(filePath ? ["add", "--", filePath] : ["add", "-A"]);
      return { ok: true };
    }
    async function reviewUnstage2(repoPath, filePath, gitBin) {
      const cwd = resolveRequestedPathForIpc2(repoPath, { purpose: "Review unstage" });
      await gitFor(cwd, gitBin).raw(filePath ? ["reset", "-q", "HEAD", "--", filePath] : ["reset", "-q", "HEAD"]);
      return { ok: true };
    }
    async function reviewRevert2(repoPath, filePath, gitBin) {
      const cwd = resolveRequestedPathForIpc2(repoPath, { purpose: "Review revert" });
      const git = gitFor(cwd, gitBin);
      if (filePath) {
        await git.raw(["checkout", "HEAD", "--", filePath]).catch(() => void 0);
        await git.raw(["clean", "-fd", "--", filePath]).catch(() => void 0);
      } else {
        await git.raw(["checkout", "HEAD", "--", "."]).catch(() => void 0);
        await git.raw(["clean", "-fd"]).catch(() => void 0);
      }
      return { ok: true };
    }
    async function reviewRevParse2(repoPath, ref, gitBin) {
      let cwd;
      try {
        cwd = resolveRequestedPathForIpc2(repoPath, { purpose: "Review rev-parse" });
      } catch {
        return null;
      }
      try {
        return (await gitFor(cwd, gitBin).revparse([ref || "HEAD"])).trim() || null;
      } catch {
        return null;
      }
    }
    async function reviewCommit2(repoPath, message, push, gitBin) {
      const cwd = resolveRequestedPathForIpc2(repoPath, { purpose: "Review commit" });
      const git = gitFor(cwd, gitBin);
      const status = await git.status();
      if (status.staged.length === 0) {
        await git.raw(["add", "-A"]);
      }
      await git.commit(message);
      if (push) {
        const fresh = await git.status();
        if (fresh.tracking) {
          await git.push();
        } else if (fresh.current) {
          await git.raw(["push", "-u", "origin", fresh.current]);
        }
      }
      return { ok: true };
    }
    async function reviewCommitContext2(repoPath, gitBin) {
      let cwd;
      try {
        cwd = resolveRequestedPathForIpc2(repoPath, { purpose: "Review commit context" });
      } catch {
        return { diff: "", recent: "" };
      }
      const git = gitFor(cwd, gitBin);
      const safe = (args) => git.diff(args).catch(() => "");
      let status;
      try {
        status = await git.status();
      } catch {
        return { diff: "", recent: "" };
      }
      let diff = capText(
        status.staged.length > 0 ? await safe(["--cached"]) : await safe(["HEAD"]),
        COMMIT_CONTEXT_DIFF_MAX_CHARS,
        "diff truncated for commit-message generation"
      );
      const untracked = status.not_added || [];
      if (untracked.length > 0) {
        const visible = untracked.slice(0, COMMIT_CONTEXT_UNTRACKED_MAX);
        const omitted = untracked.length - visible.length;
        const note = `
# New (untracked) files:
${visible.map((p) => `#   ${p}`).join("\n")}
` + (omitted > 0 ? `#   ... ${omitted} more omitted
` : "");
        diff = diff ? `${diff}${note}` : note;
      }
      const recent = await git.raw(["log", "-n", "10", "--pretty=format:%s"]).catch(() => "");
      return { diff: diff || "", recent: String(recent || "").trim() };
    }
    async function reviewPush2(repoPath, gitBin) {
      const cwd = resolveRequestedPathForIpc2(repoPath, { purpose: "Review push" });
      const git = gitFor(cwd, gitBin);
      const status = await git.status();
      if (status.tracking) {
        await git.push();
      } else if (status.current) {
        await git.raw(["push", "-u", "origin", status.current]);
      }
      return { ok: true };
    }
    async function reviewShipInfo2(repoPath, ghBin) {
      let cwd;
      try {
        cwd = resolveRequestedPathForIpc2(repoPath, { purpose: "Review ship info" });
      } catch {
        return { ghReady: false, pr: null };
      }
      const auth = await runGh(["auth", "status"], cwd, ghBin);
      if (!auth.ok) {
        return { ghReady: false, pr: null };
      }
      const view = await runGh(["pr", "view", "--json", "url,state,number"], cwd, ghBin);
      if (!view.ok) {
        return { ghReady: true, pr: null };
      }
      try {
        const pr = JSON.parse(view.stdout);
        return { ghReady: true, pr: pr && pr.url ? { url: pr.url, state: pr.state, number: pr.number } : null };
      } catch {
        return { ghReady: true, pr: null };
      }
    }
    async function reviewCreatePr2(repoPath, gitBin, ghBin) {
      const cwd = resolveRequestedPathForIpc2(repoPath, { purpose: "Review create PR" });
      await reviewPush2(repoPath, gitBin).catch(() => void 0);
      const created = await runGh(["pr", "create", "--fill"], cwd, ghBin);
      if (!created.ok) {
        throw new Error("gh pr create failed (is gh installed and authenticated?)");
      }
      const url = created.stdout.trim().split("\n").filter(Boolean).pop() || "";
      return { url };
    }
    async function repoStatus2(repoPath, gitBin) {
      let cwd;
      try {
        cwd = resolveRequestedPathForIpc2(repoPath, { purpose: "Repo status" });
      } catch {
        return null;
      }
      try {
        const stat = await fs2.stat(cwd);
        if (!stat.isDirectory()) {
          return null;
        }
      } catch {
        return null;
      }
      let git;
      try {
        git = gitFor(cwd, gitBin);
      } catch {
        return null;
      }
      let status;
      try {
        status = await git.status();
      } catch {
        return null;
      }
      const detached = typeof status.detached === "boolean" ? status.detached : !status.current;
      const files = status.files.map((file) => ({
        path: file.path,
        staged: isStaged(file),
        unstaged: Boolean(file.working_dir && file.working_dir !== " " && file.working_dir !== "?"),
        untracked: file.index === "?" || file.working_dir === "?",
        conflicted: file.index === "U" || file.working_dir === "U"
      }));
      const result = {
        branch: detached ? null : status.current || null,
        defaultBranch: await defaultBranchName(git),
        detached,
        ahead: status.ahead || 0,
        behind: status.behind || 0,
        staged: files.filter((f) => f.staged).length,
        unstaged: files.filter((f) => f.unstaged).length,
        untracked: status.not_added.length,
        conflicted: status.conflicted.length,
        changed: files.length,
        added: 0,
        removed: 0,
        files: files.slice(0, 200)
      };
      try {
        const summary = await git.diffSummary(["HEAD"]);
        result.added = summary.insertions;
        result.removed = summary.deletions;
      } catch {
      }
      try {
        const untracked = status.not_added.slice(0, 500);
        for (let i = 0; i < untracked.length; i += UNTRACKED_LINE_COUNT_CONCURRENCY) {
          const batch = await Promise.all(
            untracked.slice(i, i + UNTRACKED_LINE_COUNT_CONCURRENCY).map((path3) => untrackedInsertions(cwd, path3))
          );
          result.added += batch.reduce((sum, n) => sum + n, 0);
        }
      } catch {
      }
      return result;
    }
    module2.exports = {
      branchBase,
      fileDiffVsHead: fileDiffVsHead2,
      repoStatus: repoStatus2,
      resolveRenamePath,
      reviewCommit: reviewCommit2,
      reviewCommitContext: reviewCommitContext2,
      reviewCreatePr: reviewCreatePr2,
      reviewDiff: reviewDiff2,
      reviewList: reviewList2,
      reviewPush: reviewPush2,
      reviewRevParse: reviewRevParse2,
      reviewRevert: reviewRevert2,
      reviewShipInfo: reviewShipInfo2,
      reviewStage: reviewStage2,
      reviewUnstage: reviewUnstage2
    };
  }
});

// electron/git-repo-scan.cjs
var require_git_repo_scan = __commonJS({
  "electron/git-repo-scan.cjs"(exports2, module2) {
    "use strict";
    var fs2 = require("node:fs");
    var os2 = require("node:os");
    var path2 = require("node:path");
    var fsp = fs2.promises;
    var DEFAULT_MAX_DEPTH = 3;
    var MAX_CONCURRENCY = 32;
    var JUNK_DIRS = /* @__PURE__ */ new Set(["Applications", "Library", "node_modules", "site-packages", "vendor", "venv"]);
    async function mapLimit(items, limit, fn) {
      let cursor = 0;
      async function worker() {
        while (cursor < items.length) {
          const index = cursor;
          cursor += 1;
          await fn(items[index]);
        }
      }
      await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
    }
    async function scanGitRepos2(roots, options = {}) {
      const maxDepth = Number(options.maxDepth) || DEFAULT_MAX_DEPTH;
      const searchRoots = Array.isArray(roots) && roots.length > 0 ? roots : [os2.homedir()];
      const found = /* @__PURE__ */ new Map();
      async function walk(dir, depth) {
        if (depth > maxDepth) {
          return;
        }
        let entries;
        try {
          entries = await fsp.readdir(dir, { withFileTypes: true });
        } catch {
          return;
        }
        if (entries.some((entry) => entry.name === ".git" && entry.isDirectory())) {
          const root = dir.replace(/[/\\]+$/, "");
          found.set(root, path2.basename(root) || root);
          return;
        }
        const subdirs = [];
        for (const entry of entries) {
          if (!entry.isDirectory() || entry.name.startsWith(".") || JUNK_DIRS.has(entry.name)) {
            continue;
          }
          subdirs.push(path2.join(dir, entry.name));
        }
        await mapLimit(subdirs, MAX_CONCURRENCY, (sub) => walk(sub, depth + 1));
      }
      await mapLimit(
        searchRoots.map((root) => String(root || "").trim()).filter(Boolean),
        MAX_CONCURRENCY,
        (root) => walk(root, 0)
      );
      return [...found.entries()].map(([root, label]) => ({ label, root }));
    }
    module2.exports = { scanGitRepos: scanGitRepos2 };
  }
});

// electron/update-remote.cjs
var require_update_remote = __commonJS({
  "electron/update-remote.cjs"(exports2, module2) {
    "use strict";
    var OFFICIAL_REPO_HTTPS_URL2 = "https://github.com/NousResearch/hermes-agent.git";
    var OFFICIAL_REPO_CANONICAL = "github.com/nousresearch/hermes-agent";
    function canonicalGitHubRemote(url) {
      if (!url) return "";
      let value = String(url).trim();
      if (value.startsWith("git@github.com:")) {
        value = `github.com/${value.slice("git@github.com:".length)}`;
      } else if (value.startsWith("ssh://git@github.com/")) {
        value = `github.com/${value.slice("ssh://git@github.com/".length)}`;
      } else {
        try {
          const parsed = new URL(value);
          if (parsed.hostname && parsed.pathname) value = `${parsed.hostname}${parsed.pathname}`;
        } catch {
        }
      }
      value = value.trim().replace(/\/+$/, "");
      if (value.endsWith(".git")) value = value.slice(0, -4);
      return value.toLowerCase();
    }
    function isSshRemote(url) {
      const value = String(url || "").trim().toLowerCase();
      return value.startsWith("git@") || value.startsWith("ssh://");
    }
    function isOfficialSshRemote2(url) {
      return isSshRemote(url) && canonicalGitHubRemote(url) === OFFICIAL_REPO_CANONICAL;
    }
    module2.exports = {
      OFFICIAL_REPO_HTTPS_URL: OFFICIAL_REPO_HTTPS_URL2,
      OFFICIAL_REPO_CANONICAL,
      canonicalGitHubRemote,
      isSshRemote,
      isOfficialSshRemote: isOfficialSshRemote2
    };
  }
});

// electron/update-count.cjs
var require_update_count = __commonJS({
  "electron/update-count.cjs"(exports2, module2) {
    "use strict";
    function shouldCountCommits2({ isShallow, hasMergeBase }) {
      return !(isShallow && !hasMergeBase);
    }
    function resolveBehindCount2({ countStr, currentSha, targetSha, isShallow, hasMergeBase }) {
      if (!shouldCountCommits2({ isShallow, hasMergeBase })) {
        if (currentSha && targetSha && currentSha === targetSha) return 0;
        return 1;
      }
      return Number.parseInt(countStr, 10) || 0;
    }
    module2.exports = { resolveBehindCount: resolveBehindCount2, shouldCountCommits: shouldCountCommits2 };
  }
});

// electron/update-rebuild.cjs
var require_update_rebuild = __commonJS({
  "electron/update-rebuild.cjs"(exports2, module2) {
    "use strict";
    function shouldRetryRebuild(code) {
      return code !== 0;
    }
    async function runRebuildWithRetry2(rebuild) {
      let result = await rebuild(0);
      if (shouldRetryRebuild(result.code)) {
        result = await rebuild(1);
      }
      return result;
    }
    module2.exports = { shouldRetryRebuild, runRebuildWithRetry: runRebuildWithRetry2 };
  }
});

// electron/desktop-uninstall.cjs
var require_desktop_uninstall = __commonJS({
  "electron/desktop-uninstall.cjs"(exports2, module2) {
    "use strict";
    var path2 = require("node:path");
    var UNINSTALL_MODES = ["gui", "lite", "full"];
    function uninstallArgsForMode2(mode) {
      if (!UNINSTALL_MODES.includes(mode)) {
        throw new Error(`Unknown uninstall mode: ${mode}`);
      }
      return ["-m", "hermes_cli.uninstall", "--mode", mode];
    }
    function modeRemovesAgent2(mode) {
      return mode === "lite" || mode === "full";
    }
    function modeRemovesUserData2(mode) {
      return mode === "full";
    }
    function resolveRemovableAppPath2(execPath, platform, env2 = {}) {
      const exe = String(execPath || "");
      if (!exe) return null;
      const p = platform === "win32" ? path2.win32 : path2.posix;
      if (platform === "darwin") {
        const macOsDir = p.dirname(exe);
        const contents = p.dirname(macOsDir);
        const appBundle = p.dirname(contents);
        if (appBundle.endsWith(".app")) return appBundle;
        return null;
      }
      if (platform === "win32") {
        const dir2 = p.dirname(exe);
        if (/[\\/]Hermes$/i.test(dir2) || /[\\/]hermes-desktop$/i.test(dir2)) return dir2;
        return null;
      }
      if (env2.APPIMAGE) return env2.APPIMAGE;
      const dir = p.dirname(exe);
      if (/-unpacked$/.test(dir)) return dir;
      return null;
    }
    function shouldRemoveAppBundle2(isPackaged, appPath) {
      return Boolean(isPackaged) && Boolean(appPath);
    }
    function buildPosixCleanupScript2({ desktopPid, pythonExe, pythonPath, agentRoot, uninstallArgs, appPath, hermesHome }) {
      const q = (s) => `'${String(s).replace(/'/g, `'\\''`)}'`;
      const lines = [
        "#!/bin/bash",
        "set -u",
        "# Wait (up to ~30s) for the desktop process to exit so the venv python",
        "# and the app bundle are no longer in use.",
        `pid=${Number(desktopPid) || 0}`,
        'if [ "$pid" -gt 0 ]; then',
        "  for _ in $(seq 1 60); do",
        '    kill -0 "$pid" 2>/dev/null || break',
        "    sleep 0.5",
        "  done",
        "fi",
        `export HERMES_HOME=${q(hermesHome)}`
      ];
      if (pythonPath) {
        lines.push(`export PYTHONPATH=${q(pythonPath)}\${PYTHONPATH:+:$PYTHONPATH}`);
      }
      lines.push(
        `cd ${q(agentRoot)} 2>/dev/null || true`,
        `${q(pythonExe)} ${uninstallArgs.map(q).join(" ")} || true`
      );
      if (appPath) {
        lines.push(`rm -rf ${q(appPath)} || true`);
      }
      lines.push('rm -f "$0" 2>/dev/null || true');
      lines.push("");
      return lines.join("\n");
    }
    function buildWindowsCleanupScript2({ desktopPid, pythonExe, pythonPath, agentRoot, uninstallArgs, appPath, hermesHome }) {
      const pid = Number(desktopPid) || 0;
      const q = (s) => `"${String(s).replace(/"/g, "")}"`;
      const lines = [
        "@echo off",
        "setlocal enableextensions",
        `set "HERMES_HOME=${String(hermesHome).replace(/"/g, "")}"`,
        `set "PID=${pid}"`
      ];
      if (pythonPath) {
        lines.push(`set "PYTHONPATH=${String(pythonPath).replace(/"/g, "")};%PYTHONPATH%"`);
      }
      lines.push(
        "set /a waited=0",
        ":waitloop",
        'rem /FI "PID eq %PID%" is an EXACT filter \u2014 tasklist outputs the one task',
        'rem row for that PID, or "INFO: No tasks..." otherwise. /NH drops the',
        "rem header; findstr matches the PID as a whole space-delimited token so",
        "rem PID 99 cannot match 990 (the substring trap of a bare `find`).",
        'tasklist /NH /FI "PID eq %PID%" 2>nul | findstr /r /c:" %PID% " >nul',
        "if %ERRORLEVEL% neq 0 goto waited_done",
        "set /a waited+=1",
        "if %waited% geq 60 goto waited_done",
        "timeout /t 1 /nobreak >nul",
        "goto waitloop",
        ":waited_done",
        `cd /d ${q(agentRoot)}`,
        `${q(pythonExe)} ${uninstallArgs.map(q).join(" ")}`
      );
      if (appPath) {
        lines.push(
          "set /a tries=0",
          ":rmloop",
          `if not exist ${q(appPath)} goto rmdone`,
          `rmdir /s /q ${q(appPath)} >nul 2>&1`,
          `if not exist ${q(appPath)} goto rmdone`,
          "set /a tries+=1",
          "if %tries% geq 10 goto rmdone",
          "timeout /t 1 /nobreak >nul",
          "goto rmloop",
          ":rmdone"
        );
      }
      lines.push('del "%~f0"');
      lines.push("");
      return lines.join("\r\n");
    }
    module2.exports = {
      UNINSTALL_MODES,
      buildPosixCleanupScript: buildPosixCleanupScript2,
      buildWindowsCleanupScript: buildWindowsCleanupScript2,
      modeRemovesAgent: modeRemovesAgent2,
      modeRemovesUserData: modeRemovesUserData2,
      resolveRemovableAppPath: resolveRemovableAppPath2,
      shouldRemoveAppBundle: shouldRemoveAppBundle2,
      uninstallArgsForMode: uninstallArgsForMode2
    };
  }
});

// electron/workspace-cwd.cjs
var require_workspace_cwd = __commonJS({
  "electron/workspace-cwd.cjs"(exports2, module2) {
    "use strict";
    var path2 = require("node:path");
    function isPackagedInstallPath2(dir, { installRoots, isPackaged }) {
      if (!isPackaged || !dir) {
        return false;
      }
      let resolved;
      try {
        resolved = path2.resolve(String(dir));
      } catch {
        return false;
      }
      const roots = new Set(
        (installRoots ?? []).filter(Boolean).map((candidate) => path2.resolve(String(candidate)))
      );
      for (const root of roots) {
        if (resolved === root) {
          return true;
        }
        const rel = path2.relative(root, resolved);
        if (rel && !rel.startsWith("..") && !path2.isAbsolute(rel)) {
          return true;
        }
      }
      return false;
    }
    module2.exports = { isPackagedInstallPath: isPackagedInstallPath2 };
  }
});

// electron/window-state.cjs
var require_window_state = __commonJS({
  "electron/window-state.cjs"(exports2, module2) {
    "use strict";
    var DEFAULT_WIDTH = 1220;
    var DEFAULT_HEIGHT = 800;
    var MIN_WIDTH = 400;
    var MIN_HEIGHT = 620;
    var MIN_VISIBLE = 48;
    var finite = (v) => typeof v === "number" && Number.isFinite(v);
    var clamp = (v, lo, hi) => Math.max(lo, Math.min(v, hi));
    function sanitizeWindowState2(raw) {
      if (!raw || typeof raw !== "object" || !finite(raw.width) || !finite(raw.height)) return null;
      const state = {
        width: Math.max(MIN_WIDTH, Math.round(raw.width)),
        height: Math.max(MIN_HEIGHT, Math.round(raw.height)),
        isMaximized: raw.isMaximized === true
      };
      if (finite(raw.x) && finite(raw.y)) {
        state.x = Math.round(raw.x);
        state.y = Math.round(raw.y);
      }
      return state;
    }
    function onScreen(bounds, displays) {
      if (!Array.isArray(displays)) return false;
      return displays.some(({ workArea: a } = {}) => {
        if (!a) return false;
        const x = Math.min(bounds.x + bounds.width, a.x + a.width) - Math.max(bounds.x, a.x);
        const y = Math.min(bounds.y + bounds.height, a.y + a.height) - Math.max(bounds.y, a.y);
        return x >= MIN_VISIBLE && y >= MIN_VISIBLE;
      });
    }
    function computeWindowOptions2(state, displays) {
      const opts = {
        width: finite(state?.width) ? state.width : DEFAULT_WIDTH,
        height: finite(state?.height) ? state.height : DEFAULT_HEIGHT
      };
      const cap = (Array.isArray(displays) ? displays : []).reduce(
        (m, { workArea: a } = {}) => a && finite(a.width) && finite(a.height) ? { width: Math.max(m.width, a.width), height: Math.max(m.height, a.height) } : m,
        { width: 0, height: 0 }
      );
      if (cap.width && cap.height) {
        opts.width = clamp(opts.width, MIN_WIDTH, cap.width);
        opts.height = clamp(opts.height, MIN_HEIGHT, cap.height);
      }
      if (state && finite(state.x) && finite(state.y) && onScreen({ x: state.x, y: state.y, width: opts.width, height: opts.height }, displays)) {
        opts.x = state.x;
        opts.y = state.y;
      }
      return opts;
    }
    function debounce2(fn, delayMs) {
      let timer = null;
      const debounced = () => {
        clearTimeout(timer);
        timer = setTimeout(() => {
          timer = null;
          fn();
        }, delayMs);
      };
      debounced.flush = () => {
        clearTimeout(timer);
        timer = null;
        fn();
      };
      return debounced;
    }
    module2.exports = {
      DEFAULT_WIDTH,
      DEFAULT_HEIGHT,
      MIN_WIDTH,
      MIN_HEIGHT,
      MIN_VISIBLE,
      sanitizeWindowState: sanitizeWindowState2,
      onScreen,
      computeWindowOptions: computeWindowOptions2,
      debounce: debounce2
    };
  }
});

// electron/connection-config.cjs
var require_connection_config = __commonJS({
  "electron/connection-config.cjs"(exports2, module2) {
    "use strict";
    var AT_COOKIE_VARIANTS = ["__Host-hermes_session_at", "__Secure-hermes_session_at", "hermes_session_at"];
    var RT_COOKIE_VARIANTS = ["__Host-hermes_session_rt", "__Secure-hermes_session_rt", "hermes_session_rt"];
    function normalizeRemoteBaseUrl2(rawUrl) {
      const value = String(rawUrl || "").trim();
      if (!value) {
        throw new Error("Remote gateway URL is required.");
      }
      let parsed;
      try {
        parsed = new URL(value);
      } catch (error) {
        throw new Error(`Remote gateway URL is not valid: ${error.message}`);
      }
      if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
        throw new Error(`Remote gateway URL must be http:// or https://, got ${parsed.protocol}`);
      }
      parsed.hash = "";
      parsed.search = "";
      parsed.pathname = parsed.pathname.replace(/\/+$/, "");
      return parsed.toString().replace(/\/+$/, "");
    }
    function buildGatewayWsUrl2(baseUrl, token) {
      const parsed = new URL(baseUrl);
      const wsScheme = parsed.protocol === "https:" ? "wss" : "ws";
      const prefix = parsed.pathname.replace(/\/+$/, "");
      return `${wsScheme}://${parsed.host}${prefix}/api/ws?token=${encodeURIComponent(token)}`;
    }
    function buildGatewayWsUrlWithTicket2(baseUrl, ticket) {
      const parsed = new URL(baseUrl);
      const wsScheme = parsed.protocol === "https:" ? "wss" : "ws";
      const prefix = parsed.pathname.replace(/\/+$/, "");
      return `${wsScheme}://${parsed.host}${prefix}/api/ws?ticket=${encodeURIComponent(ticket)}`;
    }
    async function resolveTestWsUrl2(baseUrl, authMode, token, deps = {}) {
      if (authMode === "oauth") {
        const mintTicket = deps.mintTicket;
        if (typeof mintTicket !== "function") {
          throw new Error("resolveTestWsUrl: a mintTicket function is required in OAuth mode.");
        }
        let ticket;
        try {
          ticket = await mintTicket(baseUrl);
        } catch (error) {
          const err = new Error(
            "Reached the gateway over HTTP, but could not mint a WebSocket ticket for the OAuth session (it may have expired). Open Settings \u2192 Gateway and sign in again."
          );
          err.needsOauthLogin = true;
          err.cause = error;
          throw err;
        }
        return buildGatewayWsUrlWithTicket2(baseUrl, ticket);
      }
      if (!token) {
        return null;
      }
      return buildGatewayWsUrl2(baseUrl, token);
    }
    function connectionScopeKey2(profile) {
      return String(profile ?? "").trim() || null;
    }
    function normAuthMode2(mode) {
      return mode === "oauth" ? "oauth" : "token";
    }
    function profileRemoteOverride2(config, profile) {
      const key = connectionScopeKey2(profile);
      const entry = key ? config?.profiles?.[key] : null;
      if (!entry || typeof entry !== "object" || entry.mode !== "remote") {
        return null;
      }
      const url = String(entry.url || "").trim();
      if (!url) {
        return null;
      }
      return { url, authMode: normAuthMode2(entry.authMode), token: entry.token };
    }
    function pathWithGlobalRemoteProfile2(path2, profile, opts = {}) {
      const scopedProfile = connectionScopeKey2(profile);
      if (!scopedProfile || !opts.globalRemote || opts.profileRemoteOverride) {
        return path2;
      }
      const rawPath = String(path2 || "");
      if (!rawPath) {
        return path2;
      }
      let parsed;
      try {
        parsed = new URL(rawPath, "http://hermes.local");
      } catch {
        return path2;
      }
      if (parsed.searchParams.has("profile")) {
        return path2;
      }
      parsed.searchParams.set("profile", scopedProfile);
      return `${parsed.pathname}${parsed.search}${parsed.hash}`;
    }
    function tokenPreview2(value) {
      const raw = String(value || "");
      if (!raw) {
        return null;
      }
      return raw.length <= 8 ? "set" : `...${raw.slice(-6)}`;
    }
    function authModeFromStatus2(statusBody) {
      return statusBody && statusBody.auth_required ? "oauth" : "token";
    }
    function resolveAuthMode2(inputAuthMode, existingAuthMode) {
      if (inputAuthMode === "oauth") return "oauth";
      if (inputAuthMode === "token") return "token";
      if (existingAuthMode === "oauth") return "oauth";
      return "token";
    }
    function cookiesHaveSession2(cookies) {
      if (!Array.isArray(cookies)) return false;
      return cookies.some((c) => c && AT_COOKIE_VARIANTS.includes(c.name) && c.value);
    }
    function cookiesHaveLiveSession2(cookies) {
      if (!Array.isArray(cookies)) return false;
      return cookies.some(
        (c) => c && c.value && (AT_COOKIE_VARIANTS.includes(c.name) || RT_COOKIE_VARIANTS.includes(c.name))
      );
    }
    module2.exports = {
      AT_COOKIE_VARIANTS,
      RT_COOKIE_VARIANTS,
      authModeFromStatus: authModeFromStatus2,
      buildGatewayWsUrl: buildGatewayWsUrl2,
      buildGatewayWsUrlWithTicket: buildGatewayWsUrlWithTicket2,
      connectionScopeKey: connectionScopeKey2,
      cookiesHaveSession: cookiesHaveSession2,
      cookiesHaveLiveSession: cookiesHaveLiveSession2,
      normAuthMode: normAuthMode2,
      normalizeRemoteBaseUrl: normalizeRemoteBaseUrl2,
      pathWithGlobalRemoteProfile: pathWithGlobalRemoteProfile2,
      profileRemoteOverride: profileRemoteOverride2,
      resolveAuthMode: resolveAuthMode2,
      resolveTestWsUrl: resolveTestWsUrl2,
      tokenPreview: tokenPreview2
    };
  }
});

// electron/main.cjs
var {
  app,
  BrowserWindow,
  Menu,
  Notification,
  clipboard,
  dialog,
  ipcMain,
  nativeImage,
  nativeTheme,
  net: electronNet,
  powerMonitor,
  protocol,
  safeStorage,
  screen,
  session,
  shell,
  systemPreferences
} = require("electron");
var crypto = require("node:crypto");
var fs = require("node:fs");
var http = require("node:http");
var https = require("node:https");
var net = require("node:net");
var path = require("node:path");
var { pathToFileURL } = require("node:url");
var { execFileSync, spawn } = require("node:child_process");
var { detectRemoteDisplay, isWindowsBinaryPathInWsl, isWslEnvironment } = require_bootstrap_platform();
var { runBootstrap } = require_bootstrap_runner();
var {
  buildSessionWindowUrl,
  chatWindowWebPreferences,
  createSessionWindowRegistry,
  SESSION_WINDOW_MIN_HEIGHT,
  SESSION_WINDOW_MIN_WIDTH
} = require_session_windows();
var { canImportHermesCli, verifyHermesCli } = require_backend_probes();
var { createLinkTitleWindow } = require_link_title_window();
var { probeGatewayWebSocket } = require_gateway_ws_probe();
var { adoptServedDashboardToken } = require_dashboard_token();
var { waitForDashboardPortAnnouncement } = require_backend_ready();
var { serializeJsonBody, setJsonRequestHeaders } = require_oauth_net_request();
var { fetchMarketplaceThemes, searchMarketplaceThemes } = require_vscode_marketplace();
var { buildDesktopBackendEnv, normalizeHermesHomeRoot } = require_backend_env();
var { readWindowsUserEnvVar } = require_windows_user_env();
var { readDirForIpc } = require_fs_read_dir();
var { readLiveUpdateMarker } = require_update_marker();
var {
  resolveUnpackedRelease,
  decideRelaunchOutcome,
  sandboxPreflight,
  sandboxFallbackFromEnv,
  collectRelaunchArgs,
  collectRelaunchEnv,
  buildRelaunchScript
} = require_update_relaunch();
var { gitRootForIpc } = require_git_root();
var { addWorktree, listBranches, listWorktrees, removeWorktree, switchBranch } = require_git_worktree_ops();
var {
  fileDiffVsHead,
  repoStatus,
  reviewCommit,
  reviewCommitContext,
  reviewCreatePr,
  reviewDiff,
  reviewList,
  reviewPush,
  reviewRevParse,
  reviewRevert,
  reviewShipInfo,
  reviewStage,
  reviewUnstage
} = require_git_review_ops();
var { scanGitRepos } = require_git_repo_scan();
var { OFFICIAL_REPO_HTTPS_URL, isOfficialSshRemote } = require_update_remote();
var { resolveBehindCount, shouldCountCommits } = require_update_count();
var { runRebuildWithRetry } = require_update_rebuild();
var {
  buildPosixCleanupScript,
  buildWindowsCleanupScript,
  modeRemovesAgent,
  modeRemovesUserData,
  resolveRemovableAppPath,
  shouldRemoveAppBundle,
  uninstallArgsForMode
} = require_desktop_uninstall();
var { isPackagedInstallPath: isPackagedInstallPathUnderRoots } = require_workspace_cwd();
var {
  MIN_WIDTH: WINDOW_MIN_WIDTH,
  MIN_HEIGHT: WINDOW_MIN_HEIGHT,
  sanitizeWindowState,
  computeWindowOptions,
  debounce
} = require_window_state();
var {
  authModeFromStatus,
  buildGatewayWsUrl,
  buildGatewayWsUrlWithTicket,
  connectionScopeKey,
  cookiesHaveSession,
  cookiesHaveLiveSession,
  normAuthMode,
  normalizeRemoteBaseUrl,
  pathWithGlobalRemoteProfile,
  profileRemoteOverride,
  resolveAuthMode,
  resolveTestWsUrl,
  tokenPreview
} = require_connection_config();
var {
  DATA_URL_READ_MAX_BYTES,
  DEFAULT_FETCH_TIMEOUT_MS,
  TEXT_PREVIEW_SOURCE_MAX_BYTES,
  encryptDesktopSecret: encryptDesktopSecretStrict,
  resolveReadableFileForIpc,
  resolveRequestedPathForIpc,
  resolveTimeoutMs
} = require_hardening();
var nodePty = null;
var nodePtyDir = null;
try {
  nodePty = require("node-pty");
  nodePtyDir = path.dirname(require.resolve("node-pty/package.json"));
} catch {
  try {
    const path2 = require("node:path");
    const resourcesPath = process.resourcesPath;
    if (resourcesPath) {
      nodePtyDir = path2.join(resourcesPath, "native-deps", "node-pty");
      nodePty = require(nodePtyDir);
    }
  } catch {
    console.log(`[terminal] failed to load node-pty from path ${nodePtyDir}`);
    nodePty = null;
    nodePtyDir = null;
  }
}
var USER_DATA_OVERRIDE = process.env.HERMES_DESKTOP_USER_DATA_DIR;
if (USER_DATA_OVERRIDE) {
  const resolvedUserData = path.resolve(USER_DATA_OVERRIDE);
  fs.mkdirSync(resolvedUserData, { recursive: true });
  app.setPath("userData", resolvedUserData);
}
var DEV_SERVER = process.env.HERMES_DESKTOP_DEV_SERVER;
var IS_PACKAGED = app.isPackaged;
var IS_MAC = process.platform === "darwin";
var IS_WINDOWS = process.platform === "win32";
var IS_WSL = isWslEnvironment();
var APP_ROOT = app.getAppPath();
function hiddenWindowsChildOptions(options = {}) {
  if (!IS_WINDOWS || Object.prototype.hasOwnProperty.call(options, "windowsHide")) {
    return options;
  }
  return { ...options, windowsHide: true };
}
var REMOTE_DISPLAY_REASON = detectRemoteDisplay();
if (REMOTE_DISPLAY_REASON) {
  app.disableHardwareAcceleration();
  app.commandLine.appendSwitch("disable-gpu-compositing");
  console.log(
    `[hermes] remote display detected (${REMOTE_DISPLAY_REASON}); disabling GPU hardware acceleration to prevent flicker`
  );
}
ipcMain.handle("hermes:get-remote-display-reason", () => REMOTE_DISPLAY_REASON);
app.commandLine.appendSwitch("disable-renderer-backgrounding");
app.commandLine.appendSwitch("disable-backgrounding-occluded-windows");
app.commandLine.appendSwitch("disable-background-timer-throttling");
var SOURCE_REPO_ROOT = path.resolve(APP_ROOT, "../..");
var INSTALL_STAMP_SCHEMA_VERSION = 1;
function loadInstallStamp() {
  const candidates = [
    process.resourcesPath ? path.join(process.resourcesPath, "install-stamp.json") : null,
    path.join(APP_ROOT, "build", "install-stamp.json")
  ].filter(Boolean);
  for (const p of candidates) {
    try {
      const raw = fs.readFileSync(p, "utf8");
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === "object" && typeof parsed.commit === "string" && parsed.commit.length >= 7) {
        if (parsed.schemaVersion !== INSTALL_STAMP_SCHEMA_VERSION) {
          console.warn(
            `[hermes] install-stamp.json schemaVersion ${parsed.schemaVersion} != expected ${INSTALL_STAMP_SCHEMA_VERSION}; ignoring`
          );
          continue;
        }
        return Object.freeze({
          schemaVersion: parsed.schemaVersion,
          commit: parsed.commit,
          branch: parsed.branch || null,
          builtAt: parsed.builtAt || null,
          dirty: Boolean(parsed.dirty),
          source: parsed.source || null,
          path: p
        });
      }
    } catch {
    }
  }
  return null;
}
var INSTALL_STAMP = loadInstallStamp();
if (INSTALL_STAMP) {
  console.log(
    `[hermes] install stamp: ${INSTALL_STAMP.commit.slice(0, 12)}${INSTALL_STAMP.branch ? ` (${INSTALL_STAMP.branch})` : ""}${INSTALL_STAMP.dirty ? " [DIRTY]" : ""} from ${INSTALL_STAMP.source || "unknown"}`
  );
} else if (IS_PACKAGED) {
  console.error(
    "[hermes] WARNING: no install-stamp.json found in packaged build. First-launch bootstrap will not have a pinned ref to install."
  );
}
function resolveHermesHome() {
  if (process.env.HERMES_HOME) return normalizeHermesHomeRoot(process.env.HERMES_HOME);
  if (USER_DATA_OVERRIDE) return path.join(path.resolve(USER_DATA_OVERRIDE), "hermes-home");
  if (IS_WINDOWS) {
    const fromRegistry = readWindowsUserEnvVar("HERMES_HOME");
    if (fromRegistry) return normalizeHermesHomeRoot(fromRegistry);
  }
  if (IS_WINDOWS && process.env.LOCALAPPDATA) {
    const localappdata = path.join(process.env.LOCALAPPDATA, "hermes");
    const legacy = path.join(app.getPath("home"), ".hermes");
    if (!directoryExists(localappdata) && directoryExists(legacy)) return legacy;
    return localappdata;
  }
  return path.join(app.getPath("home"), ".hermes");
}
var HERMES_HOME = resolveHermesHome();
function hermesManagedNodePathEntries() {
  const root = path.join(HERMES_HOME, "node");
  const bin = path.join(root, "bin");
  const entries = IS_WINDOWS ? [root, bin] : [bin, root];
  return entries.filter(directoryExists);
}
function pathWithHermesManagedNode(...entries) {
  return [...hermesManagedNodePathEntries(), ...entries, process.env.PATH].filter(Boolean).join(path.delimiter);
}
var ACTIVE_HERMES_ROOT = path.join(HERMES_HOME, "hermes-agent");
var VENV_ROOT = path.join(ACTIVE_HERMES_ROOT, "venv");
var BOOTSTRAP_COMPLETE_MARKER = path.join(ACTIVE_HERMES_ROOT, ".hermes-bootstrap-complete");
var BOOTSTRAP_MARKER_SCHEMA_VERSION = 1;
var DESKTOP_CONNECTION_CONFIG_PATH = path.join(app.getPath("userData"), "connection.json");
var DESKTOP_UPDATE_CONFIG_PATH = path.join(app.getPath("userData"), "updates.json");
var DESKTOP_WINDOW_STATE_PATH = path.join(app.getPath("userData"), "window-state.json");
var DESKTOP_PROFILE_CONFIG_PATH = path.join(app.getPath("userData"), "active-profile.json");
var PROFILE_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/;
var DEFAULT_UPDATE_BRANCH = "main";
var DESKTOP_LOG_PATH = path.join(HERMES_HOME, "logs", "desktop.log");
var DESKTOP_LOG_FLUSH_MS = 120;
var DESKTOP_LOG_BUFFER_MAX_CHARS = 64 * 1024;
var DESKTOP_LOG_MAX_BYTES = 10 * 1024 * 1024;
var DESKTOP_LOG_BACKUP_COUNT = 3;
var DESKTOP_LOG_DISCARD_BYTES = DESKTOP_LOG_MAX_BYTES * 4;
var desktopLogBackupPath = (n) => `${DESKTOP_LOG_PATH}.${n}`;
var BOOT_FAKE_MODE = process.env.HERMES_DESKTOP_BOOT_FAKE === "1";
var BOOT_FAKE_STEP_MS = (() => {
  const raw = Number.parseInt(String(process.env.HERMES_DESKTOP_BOOT_FAKE_STEP_MS || ""), 10);
  if (!Number.isFinite(raw) || raw <= 0) return 650;
  return Math.max(120, raw);
})();
var APP_NAME = "Hermes";
var TITLEBAR_HEIGHT = 34;
var MACOS_TRAFFIC_LIGHTS_HEIGHT = 14;
var WINDOW_BUTTON_POSITION = {
  x: 24,
  y: TITLEBAR_HEIGHT / 2 - MACOS_TRAFFIC_LIGHTS_HEIGHT / 2
};
var NATIVE_OVERLAY_BUTTON_WIDTH = 144;
var APP_ICON_PATHS = [
  path.join(APP_ROOT, "public", "apple-touch-icon.png"),
  path.join(APP_ROOT, "dist", "apple-touch-icon.png"),
  path.join(unpackedPathFor(APP_ROOT), "dist", "apple-touch-icon.png")
];
var rendererTitleBarTheme = null;
var terminalSessions = /* @__PURE__ */ new Map();
var NATIVE_THEME_CONFIG_PATH = path.join(app.getPath("userData"), "native-theme.json");
var THEME_SOURCES = /* @__PURE__ */ new Set(["dark", "light", "system"]);
function readPersistedThemeSource() {
  try {
    const parsed = JSON.parse(fs.readFileSync(NATIVE_THEME_CONFIG_PATH, "utf8"));
    if (parsed && THEME_SOURCES.has(parsed.themeSource)) {
      return parsed.themeSource;
    }
  } catch {
  }
  return "system";
}
function writePersistedThemeSource(mode) {
  try {
    fs.mkdirSync(path.dirname(NATIVE_THEME_CONFIG_PATH), { recursive: true });
    fs.writeFileSync(NATIVE_THEME_CONFIG_PATH, JSON.stringify({ themeSource: mode }, null, 2), "utf8");
  } catch (error) {
    rememberLog(`[theme] write native theme failed: ${error.message}`);
  }
}
nativeTheme.themeSource = readPersistedThemeSource();
var TRANSLUCENCY_CONFIG_PATH = path.join(app.getPath("userData"), "translucency.json");
function clampIntensity(value) {
  const n = Math.round(Number(value));
  return Number.isFinite(n) ? Math.min(100, Math.max(0, n)) : 0;
}
function readPersistedTranslucency() {
  try {
    return clampIntensity(JSON.parse(fs.readFileSync(TRANSLUCENCY_CONFIG_PATH, "utf8")).intensity);
  } catch {
    return 0;
  }
}
function writePersistedTranslucency(intensity) {
  try {
    fs.mkdirSync(path.dirname(TRANSLUCENCY_CONFIG_PATH), { recursive: true });
    fs.writeFileSync(TRANSLUCENCY_CONFIG_PATH, JSON.stringify({ intensity }, null, 2), "utf8");
  } catch (error) {
    rememberLog(`[translucency] write failed: ${error.message}`);
  }
}
var translucencyIntensity = readPersistedTranslucency();
function windowOpacity() {
  return 1 - translucencyIntensity / 100 * 0.7;
}
function applyWindowTranslucency(win) {
  if (!win || win.isDestroyed() || typeof win.setOpacity !== "function") {
    return;
  }
  try {
    win.setOpacity(windowOpacity());
  } catch (error) {
    rememberLog(`[translucency] apply failed: ${error.message}`);
  }
}
function isHexColor(value) {
  return typeof value === "string" && /^#[0-9a-f]{6}$/i.test(value);
}
function getWindowBackgroundColor() {
  if (rendererTitleBarTheme && isHexColor(rendererTitleBarTheme.background)) {
    return rendererTitleBarTheme.background;
  }
  return nativeTheme.shouldUseDarkColors ? "#111111" : "#f7f7f7";
}
function getTitleBarOverlayOptions() {
  if (IS_MAC) {
    return { height: TITLEBAR_HEIGHT };
  }
  if (rendererTitleBarTheme) {
    return {
      color: rendererTitleBarTheme.background,
      height: TITLEBAR_HEIGHT,
      symbolColor: rendererTitleBarTheme.foreground
    };
  }
  const useDarkColors = nativeTheme.shouldUseDarkColors;
  return {
    color: useDarkColors ? "#111111" : "#f7f7f7",
    height: TITLEBAR_HEIGHT,
    symbolColor: useDarkColors ? "#f7f7f7" : "#242424"
  };
}
var MEDIA_MIME_TYPES = {
  ".avi": "video/x-msvideo",
  ".bmp": "image/bmp",
  ".flac": "audio/flac",
  ".gif": "image/gif",
  ".jpeg": "image/jpeg",
  ".jpg": "image/jpeg",
  ".m4a": "audio/mp4",
  ".mkv": "video/x-matroska",
  ".mov": "video/quicktime",
  ".mp3": "audio/mpeg",
  ".mp4": "video/mp4",
  ".ogg": "audio/ogg",
  ".opus": "audio/ogg; codecs=opus",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".wav": "audio/wav",
  ".webm": "video/webm",
  ".webp": "image/webp"
};
var PREVIEW_HTML_EXTENSIONS = /* @__PURE__ */ new Set([".html", ".htm"]);
var PREVIEW_WATCH_DEBOUNCE_MS = 120;
var LOCAL_PREVIEW_HOSTS = /* @__PURE__ */ new Set(["0.0.0.0", "127.0.0.1", "::1", "[::1]", "localhost"]);
var TEXT_PREVIEW_MAX_BYTES = 512 * 1024;
var PREVIEW_LANGUAGE_BY_EXT = {
  ".c": "c",
  ".conf": "ini",
  ".cpp": "cpp",
  ".css": "css",
  ".csv": "csv",
  ".go": "go",
  ".graphql": "graphql",
  ".h": "c",
  ".hpp": "cpp",
  ".html": "html",
  ".java": "java",
  ".js": "javascript",
  ".json": "json",
  ".jsx": "jsx",
  ".kt": "kotlin",
  ".lua": "lua",
  ".md": "markdown",
  ".mjs": "javascript",
  ".py": "python",
  ".rb": "ruby",
  ".rs": "rust",
  ".sh": "shell",
  ".sql": "sql",
  ".svg": "xml",
  ".toml": "toml",
  ".ts": "typescript",
  ".tsx": "tsx",
  ".txt": "text",
  ".xml": "xml",
  ".yaml": "yaml",
  ".yml": "yaml",
  ".zsh": "shell"
};
function looksBinary(buffer) {
  if (!buffer.length) return false;
  let suspicious = 0;
  for (const byte of buffer) {
    if (byte === 0) return true;
    if (byte < 32 && byte !== 9 && byte !== 10 && byte !== 13) suspicious += 1;
  }
  return suspicious / buffer.length > 0.12;
}
function previewFileMetadata(filePath, mimeType) {
  let byteSize = 0;
  let binary = false;
  try {
    const stat = fs.statSync(filePath);
    byteSize = stat.size;
    if (!mimeType.startsWith("image/")) {
      const fd = fs.openSync(filePath, "r");
      try {
        const sample = Buffer.alloc(Math.min(byteSize, 4096));
        const bytesRead = fs.readSync(fd, sample, 0, sample.length, 0);
        binary = looksBinary(sample.subarray(0, bytesRead));
      } finally {
        fs.closeSync(fd);
      }
    }
  } catch {
  }
  return {
    binary,
    byteSize,
    large: byteSize > TEXT_PREVIEW_MAX_BYTES
  };
}
app.setName(APP_NAME);
if (IS_WINDOWS) {
  app.setAppUserModelId("com.nousresearch.hermes");
}
app.setAboutPanelOptions({
  applicationName: APP_NAME,
  applicationVersion: resolveHermesVersion(),
  copyright: "Copyright \xA9 2026 Nous Research"
});
var MEDIA_PROTOCOL = "hermes-media";
var STREAMABLE_MEDIA_EXTS = /* @__PURE__ */ new Set([
  ".avi",
  ".flac",
  ".m4a",
  ".mkv",
  ".mov",
  ".mp3",
  ".mp4",
  ".ogg",
  ".opus",
  ".wav",
  ".webm"
]);
protocol.registerSchemesAsPrivileged([
  {
    scheme: MEDIA_PROTOCOL,
    privileges: {
      secure: true,
      standard: true,
      stream: true,
      supportFetchAPI: true
    }
  }
]);
function registerMediaProtocol() {
  protocol.handle(MEDIA_PROTOCOL, async (request) => {
    let resolvedPath;
    try {
      const url = new URL(request.url);
      const filePath = decodeURIComponent(url.pathname.replace(/^\/+/, ""));
      ({ resolvedPath } = await resolveReadableFileForIpc(filePath, { purpose: "Media stream" }));
    } catch {
      return new Response("Media not found", { status: 404 });
    }
    if (!STREAMABLE_MEDIA_EXTS.has(path.extname(resolvedPath).toLowerCase())) {
      return new Response("Unsupported media type", { status: 415 });
    }
    return electronNet.fetch(pathToFileURL(resolvedPath).toString(), {
      bypassCustomProtocolHandlers: true,
      headers: request.headers
    });
  });
}
var mainWindow = null;
var hermesProcess = null;
var connectionPromise = null;
var backendPool = /* @__PURE__ */ new Map();
var POOL_MAX_BACKENDS = Math.max(1, Number(process.env.HERMES_DESKTOP_POOL_MAX) || 3);
var POOL_IDLE_MS = Math.max(6e4, Number(process.env.HERMES_DESKTOP_POOL_IDLE_MS) || 10 * 6e4);
var POOL_KEEPALIVE_FRESH_MS = 9e4;
var poolIdleReaper = null;
var RENDERER_RELOAD_WINDOW_MS = 6e4;
var RENDERER_RELOAD_MAX = 3;
var rendererReloadTimes = [];
var bootstrapFailure = null;
var backendStartFailure = null;
var bootstrapAbortController = null;
var connectionConfigCache = null;
var connectionConfigCacheMtime = null;
var hermesLog = [];
var previewWatchers = /* @__PURE__ */ new Map();
var previewShortcutActive = false;
var desktopLogBuffer = "";
var desktopLogFlushTimer = null;
var desktopLogFlushPromise = Promise.resolve();
var nativeThemeListenerInstalled = false;
var bootProgressState = {
  error: null,
  fakeMode: BOOT_FAKE_MODE,
  message: "Waiting to start Hermes backend",
  phase: "idle",
  progress: 0,
  running: false,
  timestamp: Date.now()
};
function planDesktopLogRotation(size) {
  if (size < DESKTOP_LOG_MAX_BYTES) return [];
  const backups = (n) => Array.from({ length: n }, (_, i) => desktopLogBackupPath(i + 1));
  if (size > DESKTOP_LOG_DISCARD_BYTES) {
    return [DESKTOP_LOG_PATH, ...backups(DESKTOP_LOG_BACKUP_COUNT)].map((p) => ["rm", p]);
  }
  const ops = [["rm", desktopLogBackupPath(DESKTOP_LOG_BACKUP_COUNT)]];
  for (let i = DESKTOP_LOG_BACKUP_COUNT - 1; i >= 1; i--) {
    ops.push(["mv", desktopLogBackupPath(i), desktopLogBackupPath(i + 1)]);
  }
  ops.push(["mv", DESKTOP_LOG_PATH, desktopLogBackupPath(1)]);
  return ops;
}
function rotateDesktopLogIfNeededSync() {
  let size;
  try {
    size = fs.statSync(DESKTOP_LOG_PATH).size;
  } catch {
    return;
  }
  for (const [op, src, dst] of planDesktopLogRotation(size)) {
    try {
      if (op === "rm") fs.rmSync(src, { force: true });
      else fs.renameSync(src, dst);
    } catch {
    }
  }
}
async function rotateDesktopLogIfNeededAsync() {
  let size;
  try {
    size = (await fs.promises.stat(DESKTOP_LOG_PATH)).size;
  } catch {
    return;
  }
  for (const [op, src, dst] of planDesktopLogRotation(size)) {
    try {
      if (op === "rm") await fs.promises.rm(src, { force: true });
      else await fs.promises.rename(src, dst);
    } catch {
    }
  }
}
function flushDesktopLogBufferSync() {
  if (!desktopLogBuffer) return;
  const chunk = desktopLogBuffer;
  desktopLogBuffer = "";
  try {
    fs.mkdirSync(path.dirname(DESKTOP_LOG_PATH), { recursive: true });
    rotateDesktopLogIfNeededSync();
    fs.appendFileSync(DESKTOP_LOG_PATH, chunk);
  } catch {
  }
}
function flushDesktopLogBufferAsync() {
  if (!desktopLogBuffer) return desktopLogFlushPromise;
  const chunk = desktopLogBuffer;
  desktopLogBuffer = "";
  desktopLogFlushPromise = desktopLogFlushPromise.then(async () => {
    await fs.promises.mkdir(path.dirname(DESKTOP_LOG_PATH), { recursive: true });
    await rotateDesktopLogIfNeededAsync();
    await fs.promises.appendFile(DESKTOP_LOG_PATH, chunk);
  }).catch(() => {
  });
  return desktopLogFlushPromise;
}
function scheduleDesktopLogFlush() {
  if (desktopLogFlushTimer) return;
  desktopLogFlushTimer = setTimeout(() => {
    desktopLogFlushTimer = null;
    void flushDesktopLogBufferAsync();
  }, DESKTOP_LOG_FLUSH_MS);
}
function rememberLog(chunk) {
  const text = String(chunk || "").trim();
  if (!text) return;
  const lines = text.split(/\r?\n/).map((line) => `[hermes] ${line}`);
  hermesLog.push(...lines);
  if (hermesLog.length > 300) {
    hermesLog.splice(0, hermesLog.length - 300);
  }
  desktopLogBuffer += `${lines.join("\n")}
`;
  if (desktopLogBuffer.length >= DESKTOP_LOG_BUFFER_MAX_CHARS) {
    if (desktopLogFlushTimer) {
      clearTimeout(desktopLogFlushTimer);
      desktopLogFlushTimer = null;
    }
    void flushDesktopLogBufferAsync();
    return;
  }
  scheduleDesktopLogFlush();
}
function openExternalUrl(rawUrl) {
  const raw = String(rawUrl || "").trim();
  if (!raw) return false;
  let parsed;
  try {
    parsed = new URL(raw);
  } catch {
    return false;
  }
  if (parsed.protocol === "file:") {
    let localPath;
    try {
      localPath = resolveRequestedPathForIpc(parsed.toString(), { purpose: "Open external file" });
    } catch {
      return false;
    }
    void shell.openPath(localPath).then((error) => {
      if (!error) {
        return;
      }
      rememberLog(`[file] openPath failed: ${error}; revealing in folder instead`);
      try {
        shell.showItemInFolder(localPath);
      } catch (revealError) {
        rememberLog(`[file] showItemInFolder failed: ${revealError.message}`);
      }
    }).catch((error) => rememberLog(`[file] openPath rejected: ${error.message}`));
    return true;
  }
  if (!["http:", "https:", "mailto:"].includes(parsed.protocol)) {
    return false;
  }
  const url = parsed.toString();
  if (IS_WSL) {
    rememberLog(`[link] opening via WSL\u2192Windows: ${url}`);
    const proc = spawn("cmd.exe", ["/c", "start", '""', url], {
      detached: true,
      stdio: "ignore",
      windowsHide: true
    });
    proc.on("error", (error) => {
      rememberLog(`[link] cmd.exe start failed: ${error.message}; falling back to xdg-open`);
      shell.openExternal(url).catch((fallback) => rememberLog(`[link] xdg-open failed: ${fallback.message}`));
    });
    proc.unref();
    return true;
  }
  shell.openExternal(url).catch((error) => rememberLog(`[link] openExternal failed: ${error.message}`));
  return true;
}
async function openPreviewInBrowser(rawUrl) {
  const raw = String(rawUrl || "").trim();
  if (!raw) return false;
  let parsed;
  try {
    parsed = new URL(raw);
  } catch {
    return false;
  }
  if (parsed.protocol === "file:") {
    let localPath;
    try {
      localPath = resolveRequestedPathForIpc(parsed.toString(), { purpose: "Open preview in browser" });
    } catch {
      return false;
    }
    await shell.openExternal(pathToFileURL(localPath).toString());
    return true;
  }
  return openExternalUrl(raw);
}
function ensureWslWindowsFonts() {
  if (!IS_WSL) return;
  const fontsDir = ["/mnt/c/Windows/Fonts", "/mnt/c/windows/fonts"].find((candidate) => {
    try {
      return fs.statSync(candidate).isDirectory();
    } catch {
      return false;
    }
  });
  if (!fontsDir) return;
  try {
    const confDir = path.join(app.getPath("home"), ".config", "fontconfig", "conf.d");
    const confPath = path.join(confDir, "99-hermes-wsl-windows-fonts.conf");
    let existing = "";
    try {
      existing = fs.readFileSync(confPath, "utf8");
    } catch {
      existing = "";
    }
    if (existing.includes(fontsDir)) return;
    fs.mkdirSync(confDir, { recursive: true });
    fs.writeFileSync(
      confPath,
      `<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig>
  <dir>${fontsDir}</dir>
</fontconfig>
`
    );
    rememberLog(`[fonts] wired WSL Windows fonts for renderer: ${fontsDir}`);
    const cache = spawn("fc-cache", ["-f", fontsDir], { detached: true, stdio: "ignore" });
    cache.on("error", () => void 0);
    cache.unref();
  } catch (error) {
    rememberLog(`[fonts] WSL font setup skipped: ${error.message}`);
  }
}
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
function clampBootProgress(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  return Math.max(0, Math.min(100, Math.round(numeric)));
}
function broadcastBootProgress() {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  const { webContents } = mainWindow;
  if (!webContents || webContents.isDestroyed()) return;
  webContents.send("hermes:boot-progress", bootProgressState);
}
var BOOTSTRAP_LOG_RING_MAX = 500;
var bootstrapState = {
  active: false,
  manifest: null,
  stages: {},
  error: null,
  log: [],
  startedAt: null,
  completedAt: null,
  unsupportedPlatform: null
};
function broadcastBootstrapEvent(ev) {
  if (ev.type === "manifest") {
    bootstrapState.manifest = ev;
    bootstrapState.active = true;
    bootstrapState.startedAt = bootstrapState.startedAt || Date.now();
    bootstrapState.stages = {};
    for (const stage of ev.stages || []) {
      bootstrapState.stages[stage.name] = { state: "pending", json: null, durationMs: null, error: null };
    }
  } else if (ev.type === "stage") {
    bootstrapState.stages[ev.name] = {
      state: ev.state,
      durationMs: ev.durationMs ?? null,
      json: ev.json ?? null,
      error: ev.error ?? null
    };
  } else if (ev.type === "log") {
    bootstrapState.log.push({ ts: Date.now(), stage: ev.stage || null, line: ev.line, stream: ev.stream || "stdout" });
    if (bootstrapState.log.length > BOOTSTRAP_LOG_RING_MAX) {
      bootstrapState.log.splice(0, bootstrapState.log.length - BOOTSTRAP_LOG_RING_MAX);
    }
  } else if (ev.type === "complete") {
    bootstrapState.active = false;
    bootstrapState.completedAt = Date.now();
    bootstrapState.error = null;
    bootstrapState.unsupportedPlatform = null;
  } else if (ev.type === "failed") {
    bootstrapState.active = false;
    bootstrapState.error = ev.error || "unknown error";
  } else if (ev.type === "unsupported-platform") {
    bootstrapState.active = false;
    bootstrapState.unsupportedPlatform = {
      platform: ev.platform,
      activeRoot: ev.activeRoot,
      installCommand: ev.installCommand,
      docsUrl: ev.docsUrl
    };
  }
  if (!mainWindow || mainWindow.isDestroyed()) return;
  const { webContents } = mainWindow;
  if (!webContents || webContents.isDestroyed()) return;
  webContents.send("hermes:bootstrap:event", ev);
}
function getBootstrapState() {
  return bootstrapState;
}
function updateBootProgress(update, options = {}) {
  const nextProgressRaw = typeof update.progress === "number" ? clampBootProgress(update.progress) : bootProgressState.progress;
  const nextProgress = options.allowDecrease ? nextProgressRaw : Math.max(bootProgressState.progress, nextProgressRaw);
  bootProgressState = {
    ...bootProgressState,
    ...update,
    error: update.error === void 0 ? bootProgressState.error : update.error,
    fakeMode: BOOT_FAKE_MODE || Boolean(update.fakeMode),
    progress: nextProgress,
    timestamp: Date.now()
  };
  if (update.message) {
    rememberLog(`[boot] ${update.message}`);
  }
  broadcastBootProgress();
}
async function advanceBootProgress(phase, message, progress) {
  updateBootProgress({
    phase,
    message,
    progress,
    running: true,
    error: null
  });
  if (BOOT_FAKE_MODE) {
    await sleep(BOOT_FAKE_STEP_MS);
  }
}
function fileExists(filePath) {
  try {
    return fs.statSync(filePath).isFile();
  } catch {
    return false;
  }
}
function directoryExists(filePath) {
  try {
    return fs.statSync(filePath).isDirectory();
  } catch {
    return false;
  }
}
var UPDATE_WAIT_TIMEOUT_MS = 20 * 60 * 1e3;
var UPDATE_WAIT_POLL_MS = 1e3;
var UPDATE_HANDOFF_DWELL_MS = 2500;
async function waitForUpdateToFinish() {
  let marker = readLiveUpdateMarker(HERMES_HOME);
  if (!marker) return false;
  rememberLog(`[updates] update in progress (pid=${marker.pid}); deferring backend start until it finishes`);
  const deadline = Date.now() + UPDATE_WAIT_TIMEOUT_MS;
  while (marker && Date.now() < deadline) {
    await advanceBootProgress(
      "backend.update-wait",
      "An update is finishing \u2014 Hermes will start automatically when it completes\u2026",
      12
    );
    await new Promise((r) => setTimeout(r, UPDATE_WAIT_POLL_MS));
    marker = readLiveUpdateMarker(HERMES_HOME);
  }
  if (marker) {
    rememberLog("[updates] update still in progress after wait timeout; starting backend anyway");
  } else {
    rememberLog("[updates] update finished; proceeding with backend start");
  }
  return true;
}
function unpackedPathFor(filePath) {
  return filePath.replace(/app\.asar(?=$|[\\/])/, "app.asar.unpacked");
}
function findOnPath(command) {
  if (!command) return null;
  if (path.isAbsolute(command) || command.includes(path.sep) || IS_WINDOWS && command.includes("/")) {
    if (!fileExists(command)) return null;
    if (isWindowsBinaryPathInWsl(command, { isWsl: IS_WSL })) return null;
    return command;
  }
  const pathEntries = String(process.env.PATH || "").split(path.delimiter).filter(Boolean);
  const extensions = IS_WINDOWS ? ["", ...(process.env.PATHEXT || ".COM;.EXE;.BAT;.CMD").split(";").filter(Boolean)] : [""];
  for (const entry of pathEntries) {
    for (const extension of extensions) {
      const candidate = path.join(entry, `${command}${extension}`);
      if (fileExists(candidate)) return candidate;
    }
  }
  return null;
}
function isCommandScript(command) {
  return IS_WINDOWS && /\.(cmd|bat)$/i.test(command || "");
}
function unwrapWindowsVenvHermesCommand(command, dashboardArgs) {
  if (!IS_WINDOWS || !command || isCommandScript(command)) return null;
  const resolved = path.resolve(String(command));
  if (!/^hermes(?:\.exe)?$/i.test(path.basename(resolved))) return null;
  const scriptsDir = path.dirname(resolved);
  if (path.basename(scriptsDir).toLowerCase() !== "scripts") return null;
  const venvRoot = path.dirname(scriptsDir);
  const python = getNoConsoleVenvPython(venvRoot);
  if (!fileExists(python)) return null;
  const root = path.dirname(venvRoot);
  return {
    label: `existing Hermes no-console Python at ${python}`,
    command: python,
    args: ["-m", "hermes_cli.main", ...dashboardArgs],
    bootstrap: false,
    env: buildDesktopBackendEnv({
      hermesHome: HERMES_HOME,
      pythonPathEntries: [
        ...directoryExists(root) ? [root] : [],
        ...getVenvSitePackagesEntries(venvRoot)
      ],
      venvRoot
    }),
    kind: "python",
    readyFile: true,
    shell: false
  };
}
function normalizeExecutablePathForCompare(commandPath) {
  if (!commandPath) return null;
  let resolved = path.resolve(String(commandPath));
  try {
    resolved = fs.realpathSync.native ? fs.realpathSync.native(resolved) : fs.realpathSync(resolved);
  } catch {
  }
  return IS_WINDOWS ? resolved.toLowerCase() : resolved;
}
function looksLikeDesktopAppBinary(commandPath) {
  if (!IS_WINDOWS || !commandPath) return false;
  const normalizedCandidate = normalizeExecutablePathForCompare(commandPath);
  const normalizedCurrentExec = normalizeExecutablePathForCompare(process.execPath);
  if (normalizedCandidate && normalizedCurrentExec && normalizedCandidate === normalizedCurrentExec) {
    return true;
  }
  let resolved = path.resolve(String(commandPath));
  try {
    resolved = fs.realpathSync.native ? fs.realpathSync.native(resolved) : fs.realpathSync(resolved);
  } catch {
  }
  const resourcesDir = path.join(path.dirname(resolved), "resources");
  return fileExists(path.join(resourcesDir, "app.asar")) || directoryExists(path.join(resourcesDir, "app.asar.unpacked"));
}
function isHermesSourceRoot(root) {
  return directoryExists(root) && fileExists(path.join(root, "hermes_cli", "main.py"));
}
function findPythonForRoot(root) {
  const override = process.env.HERMES_DESKTOP_PYTHON;
  if (override && fileExists(override)) return override;
  const relativePaths = IS_WINDOWS ? [path.join(".venv", "Scripts", "python.exe"), path.join("venv", "Scripts", "python.exe")] : [path.join(".venv", "bin", "python"), path.join("venv", "bin", "python")];
  for (const relativePath of relativePaths) {
    const candidate = path.join(root, relativePath);
    if (fileExists(candidate)) return candidate;
  }
  return findSystemPython();
}
function findSystemPython() {
  if (!IS_WINDOWS) {
    for (const command of ["python3", "python"]) {
      const candidate = findOnPath(command);
      if (candidate) return candidate;
    }
    return null;
  }
  const SUPPORTED_VERSIONS = ["3.11", "3.12", "3.13"];
  const SUPPORTED_VERSIONS_NO_DOT = ["311", "312", "313"];
  for (const hive of ["HKLM", "HKCU"]) {
    for (const version of SUPPORTED_VERSIONS) {
      try {
        const out = execFileSync(
          "reg",
          ["query", `${hive}\\SOFTWARE\\Python\\PythonCore\\${version}\\InstallPath`, "/ve", "/reg:64"],
          hiddenWindowsChildOptions({ encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] })
        );
        const match = out.match(/REG_SZ\s+(.+?)\s*$/m);
        if (match) {
          const installPath = match[1].trim();
          const pythonExe = path.join(installPath, "python.exe");
          if (fileExists(pythonExe)) return pythonExe;
        }
      } catch {
      }
    }
  }
  const programFiles = process.env["ProgramFiles"] || "C:\\Program Files";
  const localAppData = process.env.LOCALAPPDATA || "";
  for (const versionDir of SUPPORTED_VERSIONS_NO_DOT) {
    const systemWide = path.join(programFiles, `Python${versionDir}`, "python.exe");
    if (fileExists(systemWide)) return systemWide;
    if (localAppData) {
      const perUser = path.join(localAppData, "Programs", "Python", `Python${versionDir}`, "python.exe");
      if (fileExists(perUser)) return perUser;
    }
  }
  const pyExe = findOnPath("py.exe");
  if (pyExe) {
    for (const version of SUPPORTED_VERSIONS) {
      try {
        const out = execFileSync(
          pyExe,
          [`-${version}`, "-c", "import sys; print(sys.executable)"],
          hiddenWindowsChildOptions({
            encoding: "utf8",
            stdio: ["ignore", "pipe", "ignore"]
          })
        );
        const candidate = out.trim();
        if (candidate && fileExists(candidate)) return candidate;
      } catch {
      }
    }
  }
  return null;
}
function findGitBash() {
  if (!IS_WINDOWS) {
    return findOnPath("bash");
  }
  const localAppData = process.env.LOCALAPPDATA || "";
  const candidates = [];
  if (localAppData) {
    candidates.push(path.join(localAppData, "hermes", "git", "bin", "bash.exe"));
    candidates.push(path.join(localAppData, "hermes", "git", "usr", "bin", "bash.exe"));
  }
  candidates.push(path.join(process.env["ProgramFiles"] || "C:\\Program Files", "Git", "bin", "bash.exe"));
  candidates.push(path.join(process.env["ProgramFiles(x86)"] || "C:\\Program Files (x86)", "Git", "bin", "bash.exe"));
  if (localAppData) {
    candidates.push(path.join(localAppData, "Programs", "Git", "bin", "bash.exe"));
  }
  for (const candidate of candidates) {
    if (fileExists(candidate)) return candidate;
  }
  return findOnPath("bash");
}
function getVenvPython(venvRoot) {
  return path.join(venvRoot, IS_WINDOWS ? path.join("Scripts", "python.exe") : path.join("bin", "python"));
}
function readVenvHome(venvRoot) {
  try {
    const cfg = fs.readFileSync(path.join(venvRoot, "pyvenv.cfg"), "utf8");
    const match = cfg.match(/^home\s*=\s*(.+?)\s*$/im);
    return match ? match[1].trim() : null;
  } catch {
    return null;
  }
}
function getNoConsoleVenvPython(venvRoot) {
  if (!IS_WINDOWS) return getVenvPython(venvRoot);
  const venvPythonw = path.join(venvRoot, "Scripts", "pythonw.exe");
  if (fileExists(venvPythonw)) return venvPythonw;
  const baseHome = readVenvHome(venvRoot);
  if (baseHome) {
    const basePythonw = path.join(baseHome, "pythonw.exe");
    if (fileExists(basePythonw)) return basePythonw;
  }
  return venvPythonw;
}
function toNoConsolePython(pythonPath) {
  if (!IS_WINDOWS || !pythonPath) return pythonPath;
  const resolved = String(pythonPath);
  if (/pythonw\.exe$/i.test(resolved)) return resolved;
  if (/python\.exe$/i.test(resolved)) {
    const pythonw = path.join(path.dirname(resolved), "pythonw.exe");
    if (fileExists(pythonw)) return pythonw;
  }
  return pythonPath;
}
function applyWindowsNoConsoleSpawnHints(backend) {
  if (!IS_WINDOWS || !backend?.command) return backend;
  const usesHermesModule = backend.kind === "python" || Array.isArray(backend.args) && backend.args[0] === "-m" && backend.args[1] === "hermes_cli.main";
  if (!usesHermesModule) return backend;
  backend.command = toNoConsolePython(backend.command);
  if (/pythonw\.exe$/i.test(path.basename(String(backend.command || "")))) {
    backend.readyFile = true;
  }
  return backend;
}
function getVenvSitePackagesEntries(venvRoot) {
  const entries = [];
  if (!venvRoot) return entries;
  if (IS_WINDOWS) {
    const sitePackages = path.join(venvRoot, "Lib", "site-packages");
    if (directoryExists(sitePackages)) entries.push(sitePackages);
    return entries;
  }
  const version = (() => {
    try {
      const cfg = fs.readFileSync(path.join(venvRoot, "pyvenv.cfg"), "utf8");
      const match = cfg.match(/^version_info\s*=\s*(\d+\.\d+)/im);
      return match ? match[1].trim() : null;
    } catch {
      return null;
    }
  })();
  if (version) {
    const sitePackages = path.join(venvRoot, "lib", `python${version}`, "site-packages");
    if (directoryExists(sitePackages)) entries.push(sitePackages);
  }
  return entries;
}
function makeDashboardReadyFile() {
  const dir = path.join(app.getPath("userData"), "backend-ready");
  fs.mkdirSync(dir, { recursive: true });
  return path.join(dir, `dashboard-${process.pid}-${Date.now()}-${crypto.randomBytes(6).toString("hex")}.json`);
}
var _gitBinaryCache = null;
function resolveGitBinary() {
  if (_gitBinaryCache) return _gitBinaryCache;
  if (!IS_WINDOWS) {
    _gitBinaryCache = findOnPath("git") || "git";
    return _gitBinaryCache;
  }
  const localAppData = process.env.LOCALAPPDATA || "";
  const candidates = [];
  if (localAppData) {
    candidates.push(path.join(localAppData, "hermes", "git", "cmd", "git.exe"));
    candidates.push(path.join(localAppData, "hermes", "git", "bin", "git.exe"));
  }
  candidates.push(path.join(process.env["ProgramFiles"] || "C:\\Program Files", "Git", "cmd", "git.exe"));
  candidates.push(path.join(process.env["ProgramFiles(x86)"] || "C:\\Program Files (x86)", "Git", "cmd", "git.exe"));
  if (localAppData) {
    candidates.push(path.join(localAppData, "Programs", "Git", "cmd", "git.exe"));
  }
  _gitBinaryCache = candidates.find(fileExists) || findOnPath("git") || "git";
  return _gitBinaryCache;
}
var _ghBinaryCache = null;
function resolveGhBinary() {
  if (_ghBinaryCache) return _ghBinaryCache;
  const candidates = [];
  if (IS_WINDOWS) {
    candidates.push(path.join(process.env["ProgramFiles"] || "C:\\Program Files", "GitHub CLI", "gh.exe"));
    if (process.env.LOCALAPPDATA) {
      candidates.push(path.join(process.env.LOCALAPPDATA, "Microsoft", "WinGet", "Links", "gh.exe"));
    }
  } else {
    const home = app.getPath("home");
    candidates.push("/opt/homebrew/bin/gh", "/usr/local/bin/gh", "/usr/bin/gh", path.join(home, ".local", "bin", "gh"));
  }
  _ghBinaryCache = candidates.find(fileExists) || findOnPath("gh") || "gh";
  return _ghBinaryCache;
}
function recentHermesLog() {
  return hermesLog.slice(-20).join("\n");
}
function readDesktopUpdateConfig() {
  try {
    const parsed = JSON.parse(fs.readFileSync(DESKTOP_UPDATE_CONFIG_PATH, "utf8"));
    const branch = typeof parsed?.branch === "string" ? parsed.branch.trim() : "";
    return { branch: branch || DEFAULT_UPDATE_BRANCH };
  } catch {
    return { branch: DEFAULT_UPDATE_BRANCH };
  }
}
function writeFileAtomic(targetPath, data, encoding) {
  const tmp = targetPath + ".tmp";
  fs.writeFileSync(tmp, data, encoding);
  fs.renameSync(tmp, targetPath);
}
function writeDesktopUpdateConfig(config) {
  fs.mkdirSync(path.dirname(DESKTOP_UPDATE_CONFIG_PATH), { recursive: true });
  writeFileAtomic(DESKTOP_UPDATE_CONFIG_PATH, JSON.stringify(config, null, 2));
}
function readWindowState() {
  try {
    return sanitizeWindowState(JSON.parse(fs.readFileSync(DESKTOP_WINDOW_STATE_PATH, "utf8")));
  } catch {
    return null;
  }
}
function persistWindowState() {
  if (!mainWindow || mainWindow.isDestroyed() || mainWindow.isMinimized()) return;
  try {
    const { x, y, width, height } = mainWindow.getNormalBounds();
    fs.mkdirSync(path.dirname(DESKTOP_WINDOW_STATE_PATH), { recursive: true });
    writeFileAtomic(
      DESKTOP_WINDOW_STATE_PATH,
      JSON.stringify({ x, y, width, height, isMaximized: mainWindow.isMaximized() }, null, 2)
    );
  } catch (err) {
    rememberLog(`[window-state] persist failed: ${err?.message || err}`);
  }
}
var schedulePersistWindowState = debounce(persistWindowState, 250);
function resolveUpdateRoot() {
  const candidates = [
    process.env.HERMES_DESKTOP_HERMES_ROOT && path.resolve(process.env.HERMES_DESKTOP_HERMES_ROOT),
    !IS_PACKAGED && isHermesSourceRoot(SOURCE_REPO_ROOT) ? SOURCE_REPO_ROOT : null,
    isHermesSourceRoot(ACTIVE_HERMES_ROOT) ? ACTIVE_HERMES_ROOT : null
  ].filter(Boolean);
  return candidates.find((c) => directoryExists(path.join(c, ".git"))) || candidates[0] || ACTIVE_HERMES_ROOT;
}
function runGit(args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(
      resolveGitBinary(),
      IS_WINDOWS ? ["-c", "windows.appendAtomically=false", ...args] : args,
      hiddenWindowsChildOptions({
        cwd: options.cwd,
        env: { ...process.env, ...options.env || {}, GIT_TERMINAL_PROMPT: "0" },
        stdio: ["ignore", "pipe", "pipe"]
      })
    );
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      const text = chunk.toString();
      stdout += text;
      options.onLine?.("stdout", text);
    });
    child.stderr.on("data", (chunk) => {
      const text = chunk.toString();
      stderr += text;
      options.onLine?.("stderr", text);
    });
    child.once("error", reject);
    child.once("exit", (code) => resolve({ code, stdout, stderr }));
  });
}
var firstLine = (text) => (text || "").split("\n").find(Boolean) || "";
async function getOriginUrl(updateRoot) {
  const origin = await runGit(["remote", "get-url", "origin"], { cwd: updateRoot });
  return origin.code === 0 ? origin.stdout.trim() : "";
}
function emitUpdateProgress(payload) {
  const merged = { stage: "idle", message: "", percent: null, error: null, ...payload, at: Date.now() };
  rememberLog(`[updates] ${merged.stage}: ${merged.message || merged.error || ""}`);
  for (const window2 of BrowserWindow.getAllWindows()) {
    window2.webContents.send("hermes:updates:progress", merged);
  }
}
async function resolveHealedBranch(updateRoot, branch) {
  if (!branch || branch === "main") {
    return branch || "main";
  }
  const originUrl = await getOriginUrl(updateRoot);
  const remote = isOfficialSshRemote(originUrl) ? OFFICIAL_REPO_HTTPS_URL : "origin";
  const probe = await runGit(["ls-remote", "--exit-code", "--heads", remote, branch], { cwd: updateRoot });
  if (probe.code !== 2) {
    return branch;
  }
  rememberLog(`[updates] origin/${branch} is gone (merged?); falling back to main`);
  const config = readDesktopUpdateConfig();
  if (config.branch !== "main") {
    writeDesktopUpdateConfig({ ...config, branch: "main" });
  }
  return "main";
}
async function checkUpdates() {
  const updateRoot = resolveUpdateRoot();
  let { branch } = readDesktopUpdateConfig();
  const gitDir = path.join(updateRoot, ".git");
  if (!directoryExists(gitDir)) {
    return {
      supported: false,
      reason: "not-a-git-checkout",
      message: `${updateRoot} isn't a git checkout \u2014 desktop self-update only runs against a source install.`,
      hermesRoot: updateRoot,
      branch
    };
  }
  branch = await resolveHealedBranch(updateRoot, branch);
  const originUrl = await getOriginUrl(updateRoot);
  if (isOfficialSshRemote(originUrl)) {
    const git2 = (args) => runGit(args, { cwd: updateRoot }).then((r) => r.stdout.trim());
    const [currentSha2, target, dirtyStr2, currentBranch2] = await Promise.all([
      git2(["rev-parse", "HEAD"]),
      runGit(["ls-remote", OFFICIAL_REPO_HTTPS_URL, `refs/heads/${branch}`], { cwd: updateRoot }),
      git2(["status", "--porcelain"]),
      git2(["rev-parse", "--abbrev-ref", "HEAD"])
    ]);
    const targetSha2 = firstLine(target.stdout).split(/\s+/)[0] || "";
    if (target.code !== 0 || !targetSha2) {
      return {
        supported: true,
        branch,
        error: "fetch-failed",
        message: firstLine(target.stderr) || "git ls-remote failed.",
        hermesRoot: updateRoot,
        fetchedAt: Date.now()
      };
    }
    return {
      supported: true,
      branch,
      currentBranch: currentBranch2,
      behind: currentSha2 && currentSha2 === targetSha2 ? 0 : 1,
      currentSha: currentSha2,
      targetSha: targetSha2,
      commits: [],
      dirty: dirtyStr2.length > 0,
      hermesRoot: updateRoot,
      fetchedAt: Date.now()
    };
  }
  const fetched = await runGit(["fetch", "--quiet", "origin", branch], { cwd: updateRoot });
  if (fetched.code !== 0) {
    return {
      supported: true,
      branch,
      error: "fetch-failed",
      message: firstLine(fetched.stderr) || "git fetch failed.",
      hermesRoot: updateRoot,
      fetchedAt: Date.now()
    };
  }
  const git = (args) => runGit(args, { cwd: updateRoot }).then((r) => r.stdout.trim());
  const [currentSha, targetSha, dirtyStr, currentBranch, shallowStr, mergeBaseStr] = await Promise.all([
    git(["rev-parse", "HEAD"]),
    git(["rev-parse", `origin/${branch}`]),
    git(["status", "--porcelain"]),
    git(["rev-parse", "--abbrev-ref", "HEAD"]),
    git(["rev-parse", "--is-shallow-repository"]),
    // merge-base exits non-zero with empty stdout when HEAD shares no common
    // ancestor with the freshly fetched tip — exactly the shallow-clone case.
    git(["merge-base", "HEAD", `origin/${branch}`])
  ]);
  const isShallow = shallowStr === "true";
  const hasMergeBase = Boolean(mergeBaseStr);
  const countStr = shouldCountCommits({ isShallow, hasMergeBase }) ? await git(["rev-list", `HEAD..origin/${branch}`, "--count"]) : "";
  const behind = resolveBehindCount({
    countStr,
    currentSha,
    targetSha,
    isShallow,
    hasMergeBase
  });
  const commits = behind > 0 ? await readCommitLog(updateRoot, branch) : [];
  return {
    supported: true,
    branch,
    currentBranch,
    behind,
    currentSha,
    targetSha,
    commits,
    dirty: dirtyStr.length > 0,
    hermesRoot: updateRoot,
    fetchedAt: Date.now()
  };
}
async function readCommitLog(cwd, branch) {
  const SEP = "";
  const REC = "";
  const { stdout } = await runGit(
    ["log", `HEAD..origin/${branch}`, `--pretty=format:%H${SEP}%s${SEP}%an${SEP}%at${REC}`, "-n", "40"],
    { cwd }
  );
  return stdout.split(REC).map((line) => line.trim()).filter(Boolean).map((line) => {
    const [sha, summary, author, at] = line.split(SEP);
    return { sha, summary, author, at: Number.parseInt(at, 10) * 1e3 };
  });
}
var updateInFlight = false;
function resolveUpdaterBinary() {
  const name = IS_WINDOWS ? "hermes-setup.exe" : "hermes-setup";
  const candidate = path.join(HERMES_HOME, name);
  return fileExists(candidate) ? candidate : null;
}
function repairMacUpdaterHelper(updater) {
  if (!IS_MAC || !updater) return;
  try {
    execFileSync("/usr/bin/xattr", ["-cr", updater], { stdio: "ignore" });
  } catch (err) {
    rememberLog(`[updates] macOS updater helper quarantine repair skipped: ${err.message}`);
  }
  try {
    execFileSync("/usr/bin/codesign", ["--verify", updater], { stdio: "ignore" });
    return;
  } catch {
  }
  try {
    execFileSync("/usr/bin/codesign", ["--force", "--sign", "-", updater], { stdio: "ignore" });
    rememberLog("[updates] repaired macOS updater helper signature");
  } catch (err) {
    rememberLog(`[updates] macOS updater helper signature repair skipped: ${err.message}`);
  }
}
function venvHermesShimPath(updateRoot) {
  return IS_WINDOWS ? path.join(updateRoot, "venv", "Scripts", "hermes.exe") : path.join(updateRoot, "venv", "bin", "hermes");
}
function isShimLocked(shimPath) {
  if (!IS_WINDOWS) return false;
  let fd;
  try {
    fd = fs.openSync(shimPath, "r+");
    return false;
  } catch (err) {
    return err && err.code !== "ENOENT";
  } finally {
    if (fd !== void 0) {
      try {
        fs.closeSync(fd);
      } catch {
      }
    }
  }
}
function forceKillProcessTree(pid) {
  if (!IS_WINDOWS) return;
  if (!Number.isInteger(pid) || pid <= 0) return;
  try {
    execFileSync("taskkill", ["/PID", String(pid), "/T", "/F"], hiddenWindowsChildOptions({ stdio: "ignore" }));
  } catch {
  }
}
async function releaseBackendLockForUpdate(updateRoot) {
  return releaseBackendLock(updateRoot, "updates");
}
async function releaseBackendLock(updateRoot, tag) {
  if (!IS_WINDOWS) return { unlocked: true };
  const pids = [];
  if (hermesProcess && Number.isInteger(hermesProcess.pid)) pids.push(hermesProcess.pid);
  for (const entry of backendPool.values()) {
    if (entry.process && Number.isInteger(entry.process.pid)) pids.push(entry.process.pid);
  }
  if (hermesProcess && !hermesProcess.killed) {
    try {
      hermesProcess.kill("SIGTERM");
    } catch {
    }
  }
  stopAllPoolBackends();
  for (const pid of pids) forceKillProcessTree(pid);
  const shim = venvHermesShimPath(updateRoot);
  const deadlineMs = Date.now() + 15e3;
  while (Date.now() < deadlineMs) {
    if (!isShimLocked(shim)) {
      rememberLog(`[${tag}] venv shim unlocked; safe to proceed`);
      return { unlocked: true };
    }
    await new Promise((r) => setTimeout(r, 300));
  }
  rememberLog(`[${tag}] venv shim still locked after 15s; proceeding anyway (force)`);
  return { unlocked: false };
}
async function applyUpdates(opts = {}) {
  if (updateInFlight) {
    throw new Error("An update is already in progress.");
  }
  updateInFlight = true;
  try {
    const updater = resolveUpdaterBinary();
    if (!updater && !IS_WINDOWS) {
      return await applyUpdatesPosixInApp(opts);
    }
    if (!updater) {
      const updateRoot2 = resolveUpdateRoot();
      let command = "hermes update";
      try {
        const head = await runGit(["rev-parse", "--abbrev-ref", "HEAD"], { cwd: updateRoot2 });
        const current = (head.stdout || "").trim();
        if (head.code === 0 && current && current !== "HEAD") {
          const branch2 = await resolveHealedBranch(updateRoot2, current);
          if (branch2 !== "main") command = `hermes update --branch ${branch2}`;
        }
      } catch {
      }
      rememberLog(`[updates] no staged updater; surfacing manual \`${command}\` for CLI install at ${updateRoot2}`);
      emitUpdateProgress({ stage: "manual", message: command, percent: null });
      return { ok: true, manual: true, command, hermesRoot: updateRoot2 };
    }
    emitUpdateProgress({
      stage: "restart",
      message: "Updating Hermes \u2014 this window will close and the updater will open. Don\u2019t reopen Hermes yourself; it restarts automatically when the update finishes.",
      percent: 100
    });
    repairMacUpdaterHelper(updater);
    const updateRoot = resolveUpdateRoot();
    const { branch: configuredBranch } = readDesktopUpdateConfig();
    const branch = await resolveHealedBranch(updateRoot, configuredBranch || DEFAULT_UPDATE_BRANCH);
    const updaterArgs = ["--update", "--branch", branch];
    const targetApp = IS_MAC ? runningAppBundle() : null;
    if (targetApp) {
      updaterArgs.push("--target-app", targetApp);
    }
    const venvBin = path.join(updateRoot, "venv", IS_WINDOWS ? "Scripts" : "bin");
    await releaseBackendLockForUpdate(updateRoot);
    const child = spawn(updater, updaterArgs, {
      cwd: HERMES_HOME,
      env: {
        ...process.env,
        HERMES_HOME,
        PATH: pathWithHermesManagedNode(venvBin)
      },
      detached: true,
      stdio: "ignore",
      windowsHide: false
    });
    child.unref();
    rememberLog(`[updates] launched updater: ${updater} ${updaterArgs.join(" ")}; exiting desktop to release venv shim`);
    setTimeout(() => {
      app.quit();
    }, UPDATE_HANDOFF_DWELL_MS);
    return { ok: true, handedOff: true, updater };
  } finally {
    updateInFlight = false;
  }
}
async function handOffWindowsBootstrapRecovery(reason) {
  if (!IS_WINDOWS || !IS_PACKAGED) return false;
  const updater = resolveUpdaterBinary();
  if (!updater) return false;
  const updateRoot = resolveUpdateRoot();
  const { branch: configuredBranch } = readDesktopUpdateConfig();
  const branch = directoryExists(path.join(updateRoot, ".git")) ? await resolveHealedBranch(updateRoot, configuredBranch || DEFAULT_UPDATE_BRANCH) : configuredBranch || DEFAULT_UPDATE_BRANCH;
  const venvBin = path.join(updateRoot, "venv", IS_WINDOWS ? "Scripts" : "bin");
  const venvHermes = path.join(venvBin, IS_WINDOWS ? "hermes.exe" : "hermes");
  const updaterArgs = fileExists(venvHermes) ? ["--update", "--branch", branch] : ["--repair", "--branch", branch];
  await releaseBackendLockForUpdate(updateRoot);
  const child = spawn(updater, updaterArgs, {
    cwd: HERMES_HOME,
    env: {
      ...process.env,
      HERMES_HOME,
      PATH: pathWithHermesManagedNode(venvBin)
    },
    detached: true,
    stdio: "ignore",
    windowsHide: false
  });
  child.unref();
  rememberLog(`[bootstrap] handed off ${reason} recovery to updater: ${updater} ${updaterArgs.join(" ")}; exiting desktop to release app.asar`);
  setTimeout(() => {
    app.quit();
  }, UPDATE_HANDOFF_DWELL_MS);
  return true;
}
function resolveHermesCliBinary(updateRoot) {
  const venvHermes = path.join(updateRoot, "venv", "bin", "hermes");
  if (fileExists(venvHermes)) return venvHermes;
  return findOnPath("hermes") || null;
}
function runStreamedUpdate(command, args, { cwd, env: env2, stage } = {}) {
  return new Promise((resolve) => {
    let child;
    try {
      child = spawn(
        command,
        args,
        hiddenWindowsChildOptions({
          cwd,
          env: { ...process.env, ...env2 || {} },
          stdio: ["ignore", "pipe", "pipe"]
        })
      );
    } catch (err) {
      resolve({ code: 1, error: err.message });
      return;
    }
    const emitLines = (chunk) => {
      for (const line of chunk.toString().split("\n")) {
        const trimmed = line.trim();
        if (trimmed) emitUpdateProgress({ stage, message: trimmed, percent: null });
      }
    };
    child.stdout.on("data", emitLines);
    child.stderr.on("data", emitLines);
    child.once("error", (err) => resolve({ code: 1, error: err.message }));
    child.once("exit", (code) => resolve({ code }));
  });
}
function runningAppBundle() {
  if (!IS_MAC) return null;
  let dir = path.dirname(app.getPath("exe"));
  for (let i = 0; i < 2; i++) dir = path.dirname(dir);
  return dir.endsWith(".app") ? dir : null;
}
function shellQuote(value) {
  return `'${String(value).replace(/'/g, `'\\''`)}'`;
}
async function applyUpdatesPosixInApp() {
  const updateRoot = resolveUpdateRoot();
  const hermes = resolveHermesCliBinary(updateRoot);
  if (!hermes) {
    emitUpdateProgress({ stage: "manual", message: "hermes update", percent: null });
    return { ok: true, manual: true, command: "hermes update", hermesRoot: updateRoot };
  }
  const env2 = {
    HERMES_HOME,
    PATH: pathWithHermesManagedNode(path.join(updateRoot, "venv", "bin"))
  };
  const desktopChildPids = [];
  if (hermesProcess && Number.isInteger(hermesProcess.pid)) {
    desktopChildPids.push(hermesProcess.pid);
  }
  for (const entry of backendPool.values()) {
    if (entry.process && Number.isInteger(entry.process.pid)) {
      desktopChildPids.push(entry.process.pid);
    }
  }
  if (desktopChildPids.length) {
    env2.HERMES_DESKTOP_CHILD_PID = desktopChildPids.join(",");
  }
  let branchArgs = [];
  try {
    const head = await runGit(["rev-parse", "--abbrev-ref", "HEAD"], { cwd: updateRoot });
    const current = (head.stdout || "").trim();
    if (head.code === 0 && current && current !== "HEAD") {
      branchArgs = ["--branch", await resolveHealedBranch(updateRoot, current)];
    }
  } catch {
  }
  emitUpdateProgress({ stage: "update", message: "Updating Hermes (git + dependencies)\u2026", percent: 10 });
  const updated = await runStreamedUpdate(hermes, ["update", "--yes", ...branchArgs], {
    cwd: updateRoot,
    env: env2,
    stage: "update"
  });
  if (updated.code !== 0) {
    emitUpdateProgress({ stage: "error", message: "hermes update failed.", error: updated.error || "update-failed" });
    return { ok: false, error: "hermes update failed" };
  }
  emitUpdateProgress({ stage: "rebuild", message: "Rebuilding the desktop app\u2026", percent: 60 });
  const rebuilt = await runRebuildWithRetry((attempt) => {
    if (attempt > 0) {
      emitUpdateProgress({ stage: "rebuild", message: "Retrying the desktop rebuild\u2026", percent: 60 });
    }
    return runStreamedUpdate(hermes, ["desktop", "--build-only"], { cwd: updateRoot, env: env2, stage: "rebuild" });
  });
  if (rebuilt.code !== 0) {
    emitUpdateProgress({
      stage: "error",
      message: "Backend updated, but the desktop rebuild failed. Restart Hermes to retry.",
      error: rebuilt.error || "rebuild-failed"
    });
    return { ok: false, backendUpdated: true, error: "desktop rebuild failed" };
  }
  if (!IS_MAC) {
    const unpackedDir = resolveUnpackedRelease(process.execPath, updateRoot, process.platform);
    const underUnpacked = unpackedDir !== null;
    const preflight = underUnpacked ? sandboxPreflight(unpackedDir, (p) => fs.statSync(p)) : { ok: false, reason: "not-under-unpacked", path: null };
    const sandboxFallback = sandboxFallbackFromEnv(process.env, process.argv.slice(1));
    const sandboxOk = preflight.ok || sandboxFallback;
    if (underUnpacked && !preflight.ok) {
      rememberLog(
        `[updates] sandbox preflight: not launchable (${preflight.reason}) at ${preflight.path}; fallback=${sandboxFallback ? "env/--no-sandbox" : "none"}`
      );
    }
    const outcome = decideRelaunchOutcome({ underUnpacked, sandboxOk });
    if (outcome === "relaunch") {
      emitUpdateProgress({ stage: "restart", message: "Restarting Hermes\u2026", percent: 100 });
      const relaunchArgs = collectRelaunchArgs(process.argv.slice(1));
      const relaunchEnv = collectRelaunchEnv(process.env);
      const relaunchScript = buildRelaunchScript({
        pid: process.pid,
        execPath: process.execPath,
        args: relaunchArgs,
        env: relaunchEnv,
        cwd: process.cwd()
      });
      const scriptPath2 = path.join(app.getPath("temp"), `hermes-desktop-update-${Date.now()}.sh`);
      try {
        fs.writeFileSync(scriptPath2, relaunchScript, { mode: 493 });
        const child2 = spawn("/bin/bash", [scriptPath2], { detached: true, stdio: "ignore" });
        child2.unref();
        rememberLog(
          `[updates] launched linux relaunch: ${scriptPath2} -> ${process.execPath} (args=${relaunchArgs.length}, env=${Object.keys(relaunchEnv).length})`
        );
        setTimeout(() => app.quit(), UPDATE_HANDOFF_DWELL_MS);
        return { ok: true, handedOff: true };
      } catch (err) {
        rememberLog(`[updates] linux relaunch failed: ${err.message}; falling back to manual restart`);
        return {
          ok: true,
          backendUpdated: true,
          guiUpdated: false,
          manualRestart: true,
          message: "Backend updated. Quit and reopen Hermes to load the new version."
        };
      }
    }
    if (outcome === "guiSkew") {
      emitUpdateProgress({
        stage: "guiSkew",
        message: "Backend updated, but the desktop app package was not changed. Update or reinstall the Hermes desktop app to match.",
        percent: 100
      });
      rememberLog(
        `[updates] gui/backend skew: execPath ${process.execPath} not under release/*-unpacked; backend updated, GUI package unchanged (AppImage/.deb/.rpm/dev/unresolved)`
      );
      return { ok: true, backendUpdated: true, guiUpdated: false, guiSkew: true };
    }
    rememberLog(
      `[updates] sandbox not launchable (${preflight.reason}); skipping auto-relaunch, returning manual-restart so the user keeps a working window`
    );
    return {
      ok: true,
      backendUpdated: true,
      guiUpdated: false,
      manualRestart: true,
      sandboxBlocked: true,
      message: "Backend updated. The rebuilt app can\u2019t relaunch automatically (sandbox helper needs root). Quit and reopen Hermes to finish."
    };
  }
  const rebuiltApp = [
    path.join(updateRoot, "apps", "desktop", "release", "mac-arm64", "Hermes.app"),
    path.join(updateRoot, "apps", "desktop", "release", "mac", "Hermes.app")
  ].find(directoryExists);
  const targetApp = runningAppBundle();
  if (!rebuiltApp || !targetApp) {
    emitUpdateProgress({
      stage: "done",
      message: "Backend updated. Restart Hermes to load the new version.",
      percent: 100
    });
    return { ok: true, backendUpdated: true, rebuiltApp: rebuiltApp || null };
  }
  emitUpdateProgress({ stage: "restart", message: "Installing the updated app and restarting\u2026", percent: 95 });
  const swapScript = `#!/bin/bash
set -u
APP_PID=${process.pid}
SRC=${shellQuote(rebuiltApp)}
DST=${shellQuote(targetApp)}
for _ in $(seq 1 240); do
  kill -0 "$APP_PID" 2>/dev/null || break
  sleep 0.5
done
if [ "$SRC" != "$DST" ]; then
  if /usr/bin/ditto "$SRC" "$DST.hermes-update-new"; then
    rm -rf "$DST.hermes-update-old" 2>/dev/null || true
    mv "$DST" "$DST.hermes-update-old" 2>/dev/null || rm -rf "$DST"
    mv "$DST.hermes-update-new" "$DST"
    rm -rf "$DST.hermes-update-old" 2>/dev/null || true
  fi
fi
/usr/bin/xattr -dr com.apple.quarantine "$DST" 2>/dev/null || true
/usr/bin/open "$DST"
`;
  const scriptPath = path.join(app.getPath("temp"), `hermes-desktop-update-${Date.now()}.sh`);
  try {
    fs.writeFileSync(scriptPath, swapScript, { mode: 493 });
  } catch (err) {
    emitUpdateProgress({
      stage: "done",
      message: "Backend + app updated. Restart Hermes to load the new version.",
      percent: 100
    });
    rememberLog(`[updates] could not write swap script: ${err.message}; rebuilt app at ${rebuiltApp}`);
    return { ok: true, backendUpdated: true, rebuiltApp };
  }
  const child = spawn("/bin/bash", [scriptPath], { detached: true, stdio: "ignore" });
  child.unref();
  rememberLog(`[updates] launched mac swap+relaunch: ${scriptPath} (${rebuiltApp} -> ${targetApp})`);
  setTimeout(() => app.quit(), 600);
  return { ok: true, handedOff: true, rebuiltApp, targetApp };
}
function readJson(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}
function readBootstrapMarker() {
  return readJson(BOOTSTRAP_COMPLETE_MARKER);
}
function isBootstrapComplete() {
  const marker = readBootstrapMarker();
  if (!marker || typeof marker !== "object") return false;
  if (marker.schemaVersion !== BOOTSTRAP_MARKER_SCHEMA_VERSION) return false;
  if (typeof marker.pinnedCommit !== "string" || marker.pinnedCommit.length < 7) return false;
  return isHermesSourceRoot(ACTIVE_HERMES_ROOT) && fileExists(getVenvPython(VENV_ROOT));
}
function writeBootstrapMarker(payload) {
  fs.mkdirSync(path.dirname(BOOTSTRAP_COMPLETE_MARKER), { recursive: true });
  const merged = {
    schemaVersion: BOOTSTRAP_MARKER_SCHEMA_VERSION,
    pinnedCommit: payload.pinnedCommit || null,
    pinnedBranch: payload.pinnedBranch || null,
    completedAt: (/* @__PURE__ */ new Date()).toISOString(),
    desktopVersion: app.getVersion()
  };
  writeFileAtomic(BOOTSTRAP_COMPLETE_MARKER, JSON.stringify(merged, null, 2) + "\n", "utf8");
  return merged;
}
function resolveWebDist() {
  const override = process.env.HERMES_DESKTOP_WEB_DIST;
  if (override && directoryExists(path.resolve(override))) return path.resolve(override);
  const unpackedDist = path.join(unpackedPathFor(APP_ROOT), "dist");
  if (directoryExists(unpackedDist)) return unpackedDist;
  const fallback = path.join(APP_ROOT, "dist");
  if (IS_PACKAGED && /app\.asar(?=$|[\\/])/.test(fallback) && !directoryExists(fallback)) {
    rememberLog(
      `[web-dist] dashboard frontend dir resolved to an asar-internal path that is not a real directory: ${fallback}. Static routes will 404. Ensure dist/** is unpacked (asarUnpack) or set HERMES_DESKTOP_WEB_DIST.`
    );
  }
  return fallback;
}
function resolveRendererIndex() {
  const candidates = [path.join(APP_ROOT, "dist", "index.html"), path.join(resolveWebDist(), "index.html")];
  const found = candidates.find(fileExists);
  if (found) return found;
  rememberLog(
    `[renderer] index.html not found \u2014 the desktop app was packaged without a renderer bundle. Tried: ${candidates.join(", ")}. Rebuild with: hermes desktop --force-build`
  );
  return candidates[0];
}
function isPackagedInstallPath(dir) {
  return isPackagedInstallPathUnderRoots(dir, {
    isPackaged: IS_PACKAGED,
    installRoots: [
      APP_ROOT,
      path.dirname(process.execPath),
      resolveRemovableAppPath(process.execPath, process.platform, process.env)
    ]
  });
}
function resolveHermesCwd() {
  const candidates = [
    readDefaultProjectDir(),
    process.env.HERMES_DESKTOP_CWD,
    IS_PACKAGED ? null : process.env.INIT_CWD,
    IS_PACKAGED ? null : process.cwd(),
    !IS_PACKAGED ? SOURCE_REPO_ROOT : null,
    app.getPath("home")
  ];
  for (const candidate of candidates) {
    if (!candidate) continue;
    const resolved = path.resolve(String(candidate));
    if (isPackagedInstallPath(resolved)) {
      continue;
    }
    if (directoryExists(resolved)) return resolved;
  }
  return app.getPath("home");
}
function sanitizeWorkspaceCwd(cwd) {
  const trimmed = typeof cwd === "string" ? cwd.trim() : "";
  if (!trimmed || isPackagedInstallPath(trimmed)) {
    return { cwd: resolveHermesCwd(), sanitized: Boolean(trimmed) };
  }
  try {
    const resolved = path.resolve(trimmed);
    if (directoryExists(resolved)) {
      return { cwd: resolved, sanitized: false };
    }
  } catch {
  }
  return { cwd: resolveHermesCwd(), sanitized: Boolean(trimmed) };
}
var DEFAULT_PROJECT_DIR_CONFIG_FILENAME = "project-dir.json";
function defaultProjectDirConfigPath() {
  return path.join(app.getPath("userData"), DEFAULT_PROJECT_DIR_CONFIG_FILENAME);
}
function readDefaultProjectDir() {
  try {
    const raw = fs.readFileSync(defaultProjectDirConfigPath(), "utf8");
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed.dir === "string" && parsed.dir.trim()) {
      const resolved = path.resolve(parsed.dir);
      if (directoryExists(resolved)) {
        return resolved;
      }
    }
  } catch {
  }
  return null;
}
function writeDefaultProjectDir(dir) {
  const target = defaultProjectDirConfigPath();
  const payload = dir ? JSON.stringify({ dir: path.resolve(dir) }, null, 2) : JSON.stringify({}, null, 2);
  try {
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, payload, "utf8");
  } catch (error) {
    rememberLog(`[settings] write default project dir failed: ${error.message}`);
  }
}
function createPythonBackend(root, label, dashboardArgs, options = {}) {
  const python = findPythonForRoot(root);
  if (!python) return null;
  const venvRoot = path.join(root, "venv");
  const venvPython = getVenvPython(venvRoot);
  const command = IS_WINDOWS && fileExists(venvPython) ? getNoConsoleVenvPython(venvRoot) : toNoConsolePython(python);
  return applyWindowsNoConsoleSpawnHints({
    kind: "python",
    label,
    command,
    args: ["-m", "hermes_cli.main", ...dashboardArgs],
    env: buildDesktopBackendEnv({
      hermesHome: HERMES_HOME,
      pythonPathEntries: [root],
      venvRoot
    }),
    root,
    bootstrap: Boolean(options.bootstrap),
    shell: false
  });
}
function createActiveBackend(dashboardArgs) {
  const venvPython = getVenvPython(VENV_ROOT);
  const command = fileExists(venvPython) ? getNoConsoleVenvPython(VENV_ROOT) : toNoConsolePython(findSystemPython());
  return applyWindowsNoConsoleSpawnHints({
    kind: "python",
    label: `Hermes at ${ACTIVE_HERMES_ROOT}`,
    command,
    args: ["-m", "hermes_cli.main", ...dashboardArgs],
    env: buildDesktopBackendEnv({
      hermesHome: HERMES_HOME,
      pythonPathEntries: [ACTIVE_HERMES_ROOT],
      venvRoot: VENV_ROOT
    }),
    root: ACTIVE_HERMES_ROOT,
    bootstrap: true,
    shell: false
  });
}
function resolveHermesBackend(dashboardArgs) {
  const overrideRoot = process.env.HERMES_DESKTOP_HERMES_ROOT && path.resolve(process.env.HERMES_DESKTOP_HERMES_ROOT);
  if (overrideRoot && isHermesSourceRoot(overrideRoot)) {
    const backend = createPythonBackend(overrideRoot, `Hermes source at ${overrideRoot}`, dashboardArgs);
    if (backend) return backend;
  }
  if (!IS_PACKAGED && isHermesSourceRoot(SOURCE_REPO_ROOT)) {
    const backend = createPythonBackend(SOURCE_REPO_ROOT, `Hermes source at ${SOURCE_REPO_ROOT}`, dashboardArgs);
    if (backend) return backend;
  }
  if (isBootstrapComplete()) {
    return createActiveBackend(dashboardArgs);
  }
  if (process.env.HERMES_DESKTOP_IGNORE_EXISTING !== "1") {
    let hermesCommand = null;
    const hermesOverride = process.env.HERMES_DESKTOP_HERMES;
    if (hermesOverride) {
      const resolvedOverride = findOnPath(hermesOverride);
      if (resolvedOverride) {
        hermesCommand = resolvedOverride;
      } else if (!isWindowsBinaryPathInWsl(hermesOverride, { isWsl: IS_WSL })) {
        hermesCommand = hermesOverride;
      } else {
        rememberLog(`Ignoring Windows Hermes override under WSL: ${hermesOverride}`);
      }
    } else {
      hermesCommand = findOnPath("hermes");
    }
    if (hermesCommand) {
      if (looksLikeDesktopAppBinary(hermesCommand)) {
        rememberLog(`Ignoring desktop app executable on PATH while resolving Hermes CLI: ${hermesCommand}`);
        hermesCommand = null;
      }
    }
    if (hermesCommand) {
      const unwrapped = unwrapWindowsVenvHermesCommand(hermesCommand, dashboardArgs);
      if (unwrapped) {
        return unwrapped;
      }
      const shellForProbe = isCommandScript(hermesCommand);
      if (verifyHermesCli(hermesCommand, { shell: shellForProbe })) {
        return unwrapWindowsVenvHermesCommand(hermesCommand, dashboardArgs) || {
          label: `existing Hermes CLI at ${hermesCommand}`,
          command: hermesCommand,
          args: dashboardArgs,
          bootstrap: false,
          env: {},
          kind: "command",
          shell: shellForProbe
        };
      }
      rememberLog(
        `Ignoring existing Hermes CLI at ${hermesCommand}: --version probe failed; falling through to bootstrap.`
      );
    }
  }
  const python = findSystemPython();
  if (python) {
    if (canImportHermesCli(python)) {
      return applyWindowsNoConsoleSpawnHints({
        kind: "python",
        label: `installed hermes_cli module via ${python}`,
        command: toNoConsolePython(python),
        args: ["-m", "hermes_cli.main", ...dashboardArgs],
        bootstrap: false,
        env: {},
        shell: false
      });
    }
    rememberLog(`Ignoring system Python ${python}: hermes_cli is not importable; falling through to bootstrap.`);
  }
  return {
    kind: "bootstrap-needed",
    label: "Hermes Agent not installed yet; bootstrap required",
    command: null,
    args: dashboardArgs,
    bootstrap: true,
    env: {},
    shell: false,
    // Hints for the bootstrap runner / UI layer:
    activeRoot: ACTIVE_HERMES_ROOT,
    installStamp: INSTALL_STAMP,
    // may be null in dev
    isPackaged: IS_PACKAGED,
    platform: process.platform
  };
}
async function ensureRuntime(backend) {
  if (!backend.bootstrap) {
    await advanceBootProgress("runtime.external", `Using ${backend.label}`, 32);
    return applyWindowsNoConsoleSpawnHints(backend);
  }
  if (backend.kind === "bootstrap-needed") {
    rememberLog("[bootstrap] no Hermes install found; starting first-launch bootstrap");
    if (await handOffWindowsBootstrapRecovery("bootstrap-needed")) {
      const handoffError = new Error("Hermes recovery was handed off to Hermes Setup. The desktop will restart when recovery completes.");
      handoffError.isBootstrapFailure = true;
      handoffError.bootstrapHandedOff = true;
      bootstrapFailure = handoffError;
      throw handoffError;
    }
    try {
      broadcastBootstrapEvent({
        type: "manifest",
        stages: [],
        protocolVersion: null
      });
    } catch {
    }
    bootstrapAbortController = new AbortController();
    const bootstrapResult = await runBootstrap({
      installStamp: backend.installStamp,
      activeRoot: backend.activeRoot,
      sourceRepoRoot: SOURCE_REPO_ROOT,
      hermesHome: HERMES_HOME,
      logRoot: path.join(HERMES_HOME, "logs"),
      abortSignal: bootstrapAbortController.signal,
      onEvent: (ev) => {
        try {
          rememberLog(`[bootstrap] ${JSON.stringify(ev)}`);
        } catch {
        }
        try {
          broadcastBootstrapEvent(ev);
        } catch {
        }
      },
      writeMarker: writeBootstrapMarker
    });
    bootstrapAbortController = null;
    if (bootstrapResult.cancelled) {
      const cancelledError = new Error("Hermes install was cancelled.");
      cancelledError.isBootstrapFailure = true;
      cancelledError.bootstrapCancelled = true;
      bootstrapFailure = cancelledError;
      throw cancelledError;
    }
    if (!bootstrapResult.ok) {
      const bootstrapError = new Error(
        `Hermes bootstrap failed${bootstrapResult.failedStage ? ` at stage '${bootstrapResult.failedStage}'` : ""}: ${bootstrapResult.error || "unknown error"}. Check ${path.join(HERMES_HOME, "logs", "desktop.log")} for the full transcript.`
      );
      bootstrapError.isBootstrapFailure = true;
      bootstrapError.failedStage = bootstrapResult.failedStage || null;
      bootstrapFailure = bootstrapError;
      throw bootstrapError;
    }
    rememberLog("[bootstrap] bootstrap complete; marker written. Re-resolving backend.");
    return ensureRuntime(resolveHermesBackend(backend.args));
  }
  if (!isHermesSourceRoot(ACTIVE_HERMES_ROOT)) {
    throw new Error(
      `Hermes install at ${ACTIVE_HERMES_ROOT} is missing or incomplete. Reinstall via the desktop installer or scripts/install.ps1.`
    );
  }
  if (IS_WINDOWS && !findGitBash()) {
    throw new Error(
      "Git for Windows is required for Hermes on Windows (provides Git Bash, which the agent's terminal tool uses). Install it from https://git-scm.com/download/win or run `winget install -e --id Git.Git`, then relaunch Hermes."
    );
  }
  const venvPython = getVenvPython(VENV_ROOT);
  if (!fileExists(venvPython)) {
    throw new Error(
      `Hermes venv missing at ${VENV_ROOT}. Re-run the desktop installer or \`scripts/install.ps1\` to rebuild it.`
    );
  }
  backend.command = getNoConsoleVenvPython(VENV_ROOT);
  backend.label = `Hermes at ${ACTIVE_HERMES_ROOT} (venv: ${VENV_ROOT})`;
  updateBootProgress({
    phase: "runtime.ready",
    message: "Hermes runtime is ready",
    progress: 82,
    running: true,
    error: null
  });
  return applyWindowsNoConsoleSpawnHints(backend);
}
function fetchJson(url, token, options = {}) {
  return new Promise((resolve, reject) => {
    const body = options.body === void 0 ? void 0 : Buffer.from(JSON.stringify(options.body));
    const parsed = new URL(url);
    const client = parsed.protocol === "https:" ? https : http;
    const timeoutMs = resolveTimeoutMs(options.timeoutMs, DEFAULT_FETCH_TIMEOUT_MS);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      reject(new Error(`Unsupported Hermes backend URL protocol: ${parsed.protocol}`));
      return;
    }
    const req = client.request(
      parsed,
      {
        method: options.method || "GET",
        headers: {
          "Content-Type": "application/json",
          "X-Hermes-Session-Token": token,
          ...body ? { "Content-Length": String(body.length) } : {}
        }
      },
      (res) => {
        const chunks = [];
        res.on("error", reject);
        res.on("data", (chunk) => chunks.push(chunk));
        res.on("end", () => {
          const text = Buffer.concat(chunks).toString("utf8");
          if ((res.statusCode || 500) >= 400) {
            reject(new Error(`${res.statusCode}: ${text || res.statusMessage}`));
            return;
          }
          if (!text) {
            resolve(null);
            return;
          }
          const looksHtml = /^\s*<(?:!doctype|html)/i.test(text);
          const contentType = String(res.headers["content-type"] || "");
          if (looksHtml || contentType.includes("text/html")) {
            reject(
              new Error(
                `Expected JSON from ${url} but got HTML (status ${res.statusCode}). The endpoint is likely missing on the Hermes backend.`
              )
            );
            return;
          }
          try {
            resolve(JSON.parse(text));
          } catch {
            reject(new Error(`Invalid JSON from ${url} (status ${res.statusCode}): ${text.slice(0, 200)}`));
          }
        });
      }
    );
    req.on("error", reject);
    req.setTimeout(timeoutMs, () => {
      req.destroy(new Error(`Timed out connecting to Hermes backend after ${timeoutMs}ms`));
    });
    if (body) req.write(body);
    req.end();
  });
}
function fetchPublicJson(url, options = {}) {
  return new Promise((resolve, reject) => {
    const body = options.body === void 0 ? void 0 : Buffer.from(JSON.stringify(options.body));
    let parsed;
    try {
      parsed = new URL(url);
    } catch (error) {
      reject(new Error(`Invalid URL: ${error.message}`));
      return;
    }
    const client = parsed.protocol === "https:" ? https : http;
    const timeoutMs = resolveTimeoutMs(options.timeoutMs, DEFAULT_FETCH_TIMEOUT_MS);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      reject(new Error(`Unsupported Hermes backend URL protocol: ${parsed.protocol}`));
      return;
    }
    const req = client.request(
      parsed,
      {
        method: options.method || "GET",
        headers: {
          "Content-Type": "application/json",
          ...body ? { "Content-Length": String(body.length) } : {}
        }
      },
      (res) => {
        const chunks = [];
        res.on("data", (chunk) => chunks.push(chunk));
        res.on("end", () => {
          const text = Buffer.concat(chunks).toString("utf8");
          if ((res.statusCode || 500) >= 400) {
            reject(new Error(`${res.statusCode}: ${text || res.statusMessage}`));
            return;
          }
          if (!text) {
            resolve(null);
            return;
          }
          const looksHtml = /^\s*<(?:!doctype|html)/i.test(text);
          const contentType = String(res.headers["content-type"] || "");
          if (looksHtml || contentType.includes("text/html")) {
            reject(
              new Error(
                `Expected JSON from ${url} but got HTML (status ${res.statusCode}). The endpoint is likely missing on the Hermes backend.`
              )
            );
            return;
          }
          try {
            resolve(JSON.parse(text));
          } catch {
            reject(new Error(`Invalid JSON from ${url} (status ${res.statusCode}): ${text.slice(0, 200)}`));
          }
        });
      }
    );
    req.on("error", reject);
    req.setTimeout(timeoutMs, () => {
      req.destroy(new Error(`Timed out connecting to Hermes backend after ${timeoutMs}ms`));
    });
    if (body) req.write(body);
    req.end();
  });
}
function mimeTypeForPath(filePath) {
  const ext = path.extname(filePath || "").toLowerCase();
  return MEDIA_MIME_TYPES[ext] || "application/octet-stream";
}
function extensionForMimeType(mimeType) {
  const type = String(mimeType || "").split(";")[0].trim().toLowerCase();
  if (type === "image/png") return ".png";
  if (type === "image/jpeg") return ".jpg";
  if (type === "image/gif") return ".gif";
  if (type === "image/webp") return ".webp";
  if (type === "image/bmp") return ".bmp";
  if (type === "image/svg+xml") return ".svg";
  return "";
}
function filenameFromUrl(rawUrl, fallback = "image") {
  try {
    const parsed = new URL(rawUrl);
    const base = path.basename(decodeURIComponent(parsed.pathname || ""));
    return base && base.includes(".") ? base : fallback;
  } catch {
    return fallback;
  }
}
var titleCache = /* @__PURE__ */ new Map();
var titleInflight = /* @__PURE__ */ new Map();
var TITLE_CACHE_LIMIT = 500;
var TITLE_BYTE_BUDGET = 96 * 1024;
var TITLE_TIMEOUT_MS = 5e3;
var TITLE_MAX_REDIRECTS = 3;
var TITLE_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36";
var TITLE_ERROR_RE = /\b(access denied|attention required|captcha|error|forbidden|just a moment|request blocked|too many requests)\b/i;
var HTML_ENTITIES = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " ", "#39": "'" };
var RENDER_TITLE_MAX_CONCURRENT = 2;
var RENDER_TITLE_TIMEOUT_MS = 8e3;
var RENDER_TITLE_GRACE_MS = 700;
var RENDER_TITLE_BLOCKED_RESOURCES = /* @__PURE__ */ new Set([
  "cspReport",
  "font",
  "imageset",
  "media",
  "object",
  "ping",
  "stylesheet"
]);
var linkTitleSession = null;
var oauthSession = null;
var renderTitleInFlight = 0;
var renderTitleQueue = [];
function canonicalTitleCacheKey(rawUrl) {
  const value = String(rawUrl || "").trim();
  if (!value) return "";
  try {
    const url = new URL(value);
    const host = url.hostname.replace(/^www\./i, "").toLowerCase();
    const pathname = url.pathname === "/" ? "/" : url.pathname.replace(/\/+$/, "") || "/";
    return `${host}${pathname}${url.search || ""}`;
  } catch {
    return value;
  }
}
function cacheTitle(key, title) {
  if (titleCache.size >= TITLE_CACHE_LIMIT) titleCache.delete(titleCache.keys().next().value);
  titleCache.set(key, title);
}
function decodeHtmlEntities(value) {
  return value.replace(/&(amp|lt|gt|quot|apos|nbsp|#39);/gi, (_, k) => HTML_ENTITIES[k.toLowerCase()] ?? "").replace(/&#x([0-9a-f]+);/gi, (_, hex) => String.fromCodePoint(parseInt(hex, 16) || 32)).replace(/&#(\d+);/g, (_, dec) => String.fromCodePoint(parseInt(dec, 10) || 32));
}
function parseHtmlTitle(html) {
  const raw = html.match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1];
  return raw ? decodeHtmlEntities(raw).replace(/\s+/g, " ").trim() : "";
}
function fetchHtmlTitleWithCurl(rawUrl) {
  return new Promise((resolve) => {
    const url = String(rawUrl || "").trim();
    if (!url) return resolve("");
    const args = [
      "--silent",
      "--show-error",
      "--location",
      "--max-redirs",
      String(TITLE_MAX_REDIRECTS),
      "--max-time",
      String(Math.max(2, Math.ceil(TITLE_TIMEOUT_MS / 1e3))),
      "--connect-timeout",
      "4",
      "--user-agent",
      TITLE_USER_AGENT,
      "--header",
      "Accept: text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
      "--header",
      "Accept-Language: en-US,en;q=0.7",
      "--header",
      "Accept-Encoding: identity",
      "--raw",
      url
    ];
    const child = spawn("curl", args, hiddenWindowsChildOptions({ stdio: ["ignore", "pipe", "ignore"] }));
    const chunks = [];
    let bytes = 0;
    child.stdout.on("data", (chunk) => {
      if (bytes >= TITLE_BYTE_BUDGET) return;
      const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      const remaining = TITLE_BYTE_BUDGET - bytes;
      const next = buffer.length > remaining ? buffer.subarray(0, remaining) : buffer;
      chunks.push(next);
      bytes += next.length;
    });
    child.on("error", () => resolve(""));
    child.on("close", () => {
      if (!chunks.length) return resolve("");
      resolve(parseHtmlTitle(Buffer.concat(chunks).toString("utf8")));
    });
  });
}
function getLinkTitleSession() {
  if (linkTitleSession || !app.isReady()) return linkTitleSession;
  linkTitleSession = session.fromPartition("hermes:link-titles", { cache: false });
  linkTitleSession.webRequest.onBeforeRequest((details, callback) => {
    callback({ cancel: RENDER_TITLE_BLOCKED_RESOURCES.has(details.resourceType) });
  });
  return linkTitleSession;
}
function dequeueRenderTitle() {
  while (renderTitleInFlight < RENDER_TITLE_MAX_CONCURRENT && renderTitleQueue.length) {
    const item = renderTitleQueue.shift();
    renderTitleInFlight += 1;
    runRenderTitleJob(item.url).then((title) => {
      renderTitleInFlight -= 1;
      item.resolve(title);
      dequeueRenderTitle();
    });
  }
}
function runRenderTitleJob(rawUrl) {
  return new Promise((resolve) => {
    if (!app.isReady()) return resolve("");
    const partitionSession = getLinkTitleSession();
    if (!partitionSession) return resolve("");
    let settled = false;
    let window2 = null;
    let hardTimer = null;
    let graceTimer = null;
    const finish = (title) => {
      if (settled) return;
      settled = true;
      if (hardTimer) clearTimeout(hardTimer);
      if (graceTimer) clearTimeout(graceTimer);
      const value = (title || "").replace(/\s+/g, " ").trim();
      try {
        if (window2 && !window2.isDestroyed()) window2.destroy();
      } catch {
      }
      resolve(value);
    };
    try {
      window2 = createLinkTitleWindow(BrowserWindow, partitionSession);
    } catch {
      return finish("");
    }
    const readTitle = () => window2?.webContents?.getTitle?.() || "";
    const scheduleGrace = () => {
      if (graceTimer) clearTimeout(graceTimer);
      graceTimer = setTimeout(() => finish(readTitle()), RENDER_TITLE_GRACE_MS);
    };
    hardTimer = setTimeout(() => finish(readTitle()), RENDER_TITLE_TIMEOUT_MS);
    window2.webContents.setUserAgent(TITLE_USER_AGENT);
    window2.webContents.on("page-title-updated", scheduleGrace);
    window2.webContents.on("did-finish-load", scheduleGrace);
    window2.webContents.on("did-fail-load", (_event, _code, _desc, _validatedURL, isMainFrame) => {
      if (isMainFrame) finish("");
    });
    window2.loadURL(rawUrl, {
      httpReferrer: "https://www.google.com/",
      userAgent: TITLE_USER_AGENT
    }).catch(() => finish(""));
  });
}
function fetchHtmlTitleWithRenderer(rawUrl) {
  return new Promise((resolve) => {
    renderTitleQueue.push({ resolve, url: rawUrl });
    dequeueRenderTitle();
  });
}
var usableTitle = (value) => value && !TITLE_ERROR_RE.test(value) ? value : "";
function fetchLinkTitle(rawUrl) {
  const url = String(rawUrl || "").trim();
  const key = canonicalTitleCacheKey(url);
  if (!key) return Promise.resolve("");
  if (titleCache.has(key)) return Promise.resolve(titleCache.get(key));
  if (titleInflight.has(key)) return titleInflight.get(key);
  const pending = fetchHtmlTitleWithCurl(url).catch(() => "").then((value) => usableTitle((value || "").slice(0, 240))).then(
    async (value) => value || usableTitle((await fetchHtmlTitleWithRenderer(url).catch(() => "") || "").slice(0, 240))
  ).then((clean) => {
    cacheTitle(key, clean);
    titleInflight.delete(key);
    return clean;
  });
  titleInflight.set(key, pending);
  return pending;
}
async function resourceBufferFromUrl(rawUrl) {
  if (!rawUrl) throw new Error("Missing URL");
  if (rawUrl.startsWith("data:")) {
    const match = rawUrl.match(/^data:([^;,]+)?(;base64)?,(.*)$/s);
    if (!match) throw new Error("Invalid data URL");
    const mimeType = match[1] || "application/octet-stream";
    const encoded = match[3] || "";
    const buffer = match[2] ? Buffer.from(encoded, "base64") : Buffer.from(decodeURIComponent(encoded), "utf8");
    return { buffer, mimeType };
  }
  if (/^file:/i.test(rawUrl)) {
    const { resolvedPath } = await resolveReadableFileForIpc(rawUrl, { purpose: "Image file" });
    const buffer = await fs.promises.readFile(resolvedPath);
    return { buffer, mimeType: mimeTypeForPath(resolvedPath) };
  }
  const parsed = new URL(rawUrl);
  const client = parsed.protocol === "https:" ? https : http;
  return new Promise((resolve, reject) => {
    const req = client.get(parsed, (res) => {
      if ((res.statusCode || 500) >= 400) {
        reject(new Error(`Failed to fetch ${rawUrl}: ${res.statusCode}`));
        res.resume();
        return;
      }
      const chunks = [];
      res.on("error", reject);
      res.on("data", (chunk) => chunks.push(chunk));
      res.on("end", () => {
        resolve({
          buffer: Buffer.concat(chunks),
          mimeType: res.headers["content-type"] || "application/octet-stream"
        });
      });
    });
    req.on("error", reject);
  });
}
async function copyImageFromUrl(rawUrl) {
  const { buffer } = await resourceBufferFromUrl(rawUrl);
  const image = nativeImage.createFromBuffer(buffer);
  if (image.isEmpty()) throw new Error("Could not read image");
  clipboard.writeImage(image);
}
async function saveImageFromUrl(rawUrl) {
  const { buffer, mimeType } = await resourceBufferFromUrl(rawUrl);
  const fallbackName = filenameFromUrl(rawUrl, `image${extensionForMimeType(mimeType) || ".png"}`);
  const result = await dialog.showSaveDialog(mainWindow, {
    title: "Save Image",
    defaultPath: fallbackName
  });
  if (result.canceled || !result.filePath) return false;
  await fs.promises.writeFile(result.filePath, buffer);
  return true;
}
async function writeComposerImage(buffer, ext = ".png") {
  const rawExt = String(ext || ".png").trim().toLowerCase();
  const normalizedExt = rawExt.startsWith(".") ? rawExt : `.${rawExt}`;
  const safeExt = /^\.[a-z0-9]{1,5}$/.test(normalizedExt) ? normalizedExt : ".png";
  const dir = path.join(app.getPath("userData"), "composer-images");
  await fs.promises.mkdir(dir, { recursive: true });
  const stamp = (/* @__PURE__ */ new Date()).toISOString().replace(/[:.]/g, "-").replace("T", "_").replace("Z", "");
  const random = crypto.randomBytes(3).toString("hex");
  const filePath = path.join(dir, `composer_${stamp}_${random}${safeExt}`);
  await fs.promises.writeFile(filePath, buffer);
  return filePath;
}
function previewLabelForUrl(url) {
  return `${url.host}${url.pathname === "/" ? "" : url.pathname}`;
}
function expandUserPath(filePath) {
  const value = String(filePath || "").trim();
  if (value === "~") {
    return app.getPath("home");
  }
  if (value.startsWith(`~${path.sep}`) || value.startsWith("~/")) {
    return path.join(app.getPath("home"), value.slice(2));
  }
  return value;
}
async function previewFileTarget(rawTarget, baseDir) {
  const raw = String(rawTarget || "").trim();
  const base = baseDir ? path.resolve(expandUserPath(baseDir)) : resolveHermesCwd();
  let resolved = resolveRequestedPathForIpc(/^file:/i.test(raw) ? raw : expandUserPath(raw), {
    baseDir: base,
    purpose: "Preview target"
  });
  if (directoryExists(resolved)) {
    resolved = path.join(resolved, "index.html");
  }
  const ext = path.extname(resolved).toLowerCase();
  if (!fileExists(resolved)) {
    return null;
  }
  ;
  ({ resolvedPath: resolved } = await resolveReadableFileForIpc(resolved, { purpose: "Preview target" }));
  const mimeType = mimeTypeForPath(resolved);
  const metadata = previewFileMetadata(resolved, mimeType);
  const isHtml = PREVIEW_HTML_EXTENSIONS.has(ext);
  const isImage = mimeType.startsWith("image/");
  const previewKind = isHtml ? "html" : isImage ? "image" : metadata.binary ? "binary" : "text";
  return {
    binary: metadata.binary,
    byteSize: metadata.byteSize,
    kind: "file",
    large: metadata.large,
    label: path.basename(resolved),
    language: PREVIEW_LANGUAGE_BY_EXT[ext] || "text",
    mimeType,
    path: resolved,
    previewKind,
    source: raw,
    url: pathToFileURL(resolved).toString()
  };
}
function previewUrlTarget(rawTarget) {
  const raw = String(rawTarget || "").trim();
  const url = new URL(raw);
  if (!["http:", "https:"].includes(url.protocol)) {
    return null;
  }
  if (!LOCAL_PREVIEW_HOSTS.has(url.hostname.toLowerCase())) {
    return null;
  }
  if (url.hostname === "0.0.0.0") {
    url.hostname = "127.0.0.1";
  }
  return {
    kind: "url",
    label: previewLabelForUrl(url),
    source: raw,
    url: url.toString()
  };
}
async function normalizePreviewTarget(rawTarget, baseDir) {
  const raw = String(rawTarget || "").trim();
  if (!raw) {
    return null;
  }
  try {
    if (/^https?:\/\//i.test(raw)) {
      return previewUrlTarget(raw);
    }
    return await previewFileTarget(raw, baseDir);
  } catch {
    return null;
  }
}
async function filePathFromPreviewUrl(rawUrl) {
  const { resolvedPath } = await resolveReadableFileForIpc(String(rawUrl || ""), { purpose: "Preview file" });
  return resolvedPath;
}
function sendPreviewFileChanged(payload) {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  const { webContents } = mainWindow;
  if (!webContents || webContents.isDestroyed()) return;
  webContents.send("hermes:preview-file-changed", payload);
}
async function watchPreviewFile(rawUrl) {
  const filePath = await filePathFromPreviewUrl(rawUrl);
  const watchDir = path.dirname(filePath);
  const targetName = path.basename(filePath);
  const id = crypto.randomBytes(12).toString("base64url");
  let timer = null;
  const watcher = fs.watch(watchDir, (_eventType, filename) => {
    const changedName = filename ? path.basename(String(filename)) : "";
    if (changedName && changedName !== targetName) {
      return;
    }
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      if (!fileExists(filePath)) return;
      sendPreviewFileChanged({ id, path: filePath, url: pathToFileURL(filePath).toString() });
    }, PREVIEW_WATCH_DEBOUNCE_MS);
  });
  previewWatchers.set(id, {
    close: () => {
      if (timer) clearTimeout(timer);
      watcher.close();
    }
  });
  return { id, path: filePath };
}
function stopPreviewFileWatch(id) {
  const watcher = previewWatchers.get(id);
  if (!watcher) {
    return false;
  }
  watcher.close();
  previewWatchers.delete(id);
  return true;
}
function closePreviewWatchers() {
  for (const id of previewWatchers.keys()) {
    stopPreviewFileWatch(id);
  }
}
async function waitForHermes(baseUrl, token) {
  const deadline = Date.now() + 45e3;
  let lastError = null;
  while (Date.now() < deadline) {
    try {
      await fetchJson(`${baseUrl}/api/status`, token);
      return;
    } catch (error) {
      lastError = error;
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
  throw new Error(`Hermes backend did not become ready: ${lastError?.message || "timeout"}`);
}
function getWindowButtonPosition() {
  if (!IS_MAC) return null;
  return mainWindow?.getWindowButtonPosition?.() || WINDOW_BUTTON_POSITION;
}
function getNativeOverlayWidth() {
  return IS_MAC ? 0 : NATIVE_OVERLAY_BUTTON_WIDTH;
}
function getWindowState() {
  return {
    isFullscreen: Boolean(mainWindow?.isFullScreen?.()),
    nativeOverlayWidth: getNativeOverlayWidth(),
    windowButtonPosition: getWindowButtonPosition()
  };
}
function sendBackendExit(payload) {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  const { webContents } = mainWindow;
  if (!webContents || webContents.isDestroyed()) return;
  webContents.send("hermes:backend-exit", payload);
}
function sendClosePreviewRequested() {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  const { webContents } = mainWindow;
  if (!webContents || webContents.isDestroyed()) return;
  webContents.send("hermes:close-preview-requested");
}
function sendPowerResume() {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  const { webContents } = mainWindow;
  if (!webContents || webContents.isDestroyed()) return;
  webContents.send("hermes:power-resume");
}
var powerResumeRegistered = false;
function registerPowerResumeListeners() {
  if (powerResumeRegistered) return;
  powerResumeRegistered = true;
  try {
    powerMonitor.on("resume", sendPowerResume);
    powerMonitor.on("unlock-screen", sendPowerResume);
  } catch {
  }
}
function getAppIconPath() {
  return APP_ICON_PATHS.find(fileExists);
}
function sendOpenUpdatesRequested() {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  const { webContents } = mainWindow;
  if (!webContents || webContents.isDestroyed()) return;
  webContents.send("hermes:open-updates");
  if (!mainWindow.isVisible()) mainWindow.show();
  mainWindow.focus();
}
function sendWindowStateChanged(nextIsFullscreen) {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  const { webContents } = mainWindow;
  if (!webContents || webContents.isDestroyed()) return;
  const state = getWindowState();
  if (typeof nextIsFullscreen === "boolean") {
    state.isFullscreen = nextIsFullscreen;
  }
  webContents.send("hermes:window-state-changed", state);
}
function buildApplicationMenu() {
  const template = [];
  const checkForUpdatesItem = {
    label: "Check for Updates\u2026",
    click: () => sendOpenUpdatesRequested()
  };
  if (IS_MAC) {
    template.push({
      label: APP_NAME,
      submenu: [
        { label: `About ${APP_NAME}`, click: () => showAboutPanelFresh() },
        checkForUpdatesItem,
        { type: "separator" },
        { role: "services" },
        { type: "separator" },
        { role: "hide" },
        { role: "hideOthers" },
        { role: "unhide" },
        { type: "separator" },
        { role: "quit" }
      ]
    });
  }
  template.push({
    label: "File",
    submenu: [
      IS_MAC ? {
        accelerator: "CommandOrControl+W",
        click: () => {
          if (previewShortcutActive) {
            sendClosePreviewRequested();
          } else {
            mainWindow?.close();
          }
        },
        label: "Close"
      } : { role: "quit" }
    ]
  });
  template.push({
    label: "Edit",
    submenu: [
      { role: "undo" },
      { role: "redo" },
      { type: "separator" },
      { role: "cut" },
      { role: "copy" },
      { role: "paste" },
      { role: "delete" },
      { role: "selectAll" }
    ]
  });
  template.push({
    label: "View",
    submenu: [
      { role: "reload" },
      { role: "forceReload" },
      { role: "toggleDevTools" },
      { type: "separator" },
      {
        label: "Actual Size",
        accelerator: "CommandOrControl+0",
        click: () => {
          setAndPersistZoomLevel(mainWindow, 0);
        }
      },
      {
        label: "Zoom In",
        accelerator: "CommandOrControl+Plus",
        click: () => {
          if (mainWindow && !mainWindow.isDestroyed()) {
            setAndPersistZoomLevel(mainWindow, mainWindow.webContents.getZoomLevel() + 0.1);
          }
        }
      },
      {
        label: "Zoom Out",
        accelerator: "CommandOrControl+-",
        click: () => {
          if (mainWindow && !mainWindow.isDestroyed()) {
            setAndPersistZoomLevel(mainWindow, mainWindow.webContents.getZoomLevel() - 0.1);
          }
        }
      },
      { type: "separator" },
      { role: "togglefullscreen" }
    ]
  });
  template.push({
    label: "Window",
    submenu: IS_MAC ? [{ role: "minimize" }, { role: "zoom" }, { role: "front" }] : [{ role: "minimize" }, { role: "close" }]
  });
  template.push({
    label: "Help",
    role: "help",
    submenu: [checkForUpdatesItem]
  });
  return Menu.buildFromTemplate(template);
}
function toggleDevTools(window2) {
  const { webContents } = window2;
  if (webContents.isDevToolsOpened()) {
    webContents.closeDevTools();
  } else {
    webContents.openDevTools({ mode: "detach" });
  }
}
function installDevToolsShortcut(window2) {
  window2.webContents.on("before-input-event", (event, input) => {
    const key = input.key.toLowerCase();
    const isInspectShortcut = input.key === "F12" || IS_MAC && input.meta && input.alt && key === "i" || !IS_MAC && input.control && input.shift && key === "i";
    if (!isInspectShortcut) return;
    event.preventDefault();
    toggleDevTools(window2);
  });
}
function installPreviewShortcut(window2) {
  window2.webContents.on("before-input-event", (event, input) => {
    const key = String(input.key || "").toLowerCase();
    const isPreviewCloseShortcut = key === "w" && (IS_MAC ? input.meta : input.control) && !input.alt && !input.shift;
    if (!isPreviewCloseShortcut || !previewShortcutActive) return;
    event.preventDefault();
    sendClosePreviewRequested();
  });
}
var ZOOM_STORAGE_KEY = "hermes:desktop:zoomLevel";
function clampZoomLevel(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.min(Math.max(value, -9), 9);
}
function setAndPersistZoomLevel(window2, zoomLevel) {
  if (!window2 || window2.isDestroyed()) return;
  const next = clampZoomLevel(zoomLevel);
  window2.webContents.setZoomLevel(next);
  window2.webContents.executeJavaScript(
    `try { localStorage.setItem(${JSON.stringify(ZOOM_STORAGE_KEY)}, ${JSON.stringify(String(next))}) } catch {}`
  ).catch((error) => rememberLog(`[zoom] persist failed: ${error?.message || error}`));
}
function restorePersistedZoomLevel(window2) {
  if (!window2 || window2.isDestroyed()) return;
  window2.webContents.executeJavaScript(
    `(() => { try { return localStorage.getItem(${JSON.stringify(ZOOM_STORAGE_KEY)}) } catch { return null } })()`
  ).then((stored) => {
    if (stored == null || !window2 || window2.isDestroyed()) return;
    const level = clampZoomLevel(Number(stored));
    window2.webContents.setZoomLevel(level);
  }).catch((error) => rememberLog(`[zoom] restore failed: ${error?.message || error}`));
}
function installZoomShortcuts(window2) {
  const ZOOM_STEP = 0.1;
  window2.webContents.on("before-input-event", (event, input) => {
    const mod = IS_MAC ? input.meta : input.control;
    if (!mod || input.alt || input.shift) return;
    const key = input.key;
    if (key === "0") {
      event.preventDefault();
      setAndPersistZoomLevel(window2, 0);
    } else if (key === "=" || key === "+") {
      event.preventDefault();
      setAndPersistZoomLevel(window2, window2.webContents.getZoomLevel() + ZOOM_STEP);
    } else if (key === "-") {
      event.preventDefault();
      setAndPersistZoomLevel(window2, window2.webContents.getZoomLevel() - ZOOM_STEP);
    }
  });
}
function installContextMenu(window2) {
  window2.webContents.on("context-menu", (_event, params) => {
    const template = [];
    const hasSelection = Boolean(params.selectionText?.trim());
    const hasImage = params.mediaType === "image" && Boolean(params.srcURL);
    const hasLink = Boolean(params.linkURL);
    const isEditable = Boolean(params.isEditable);
    if (hasImage) {
      template.push(
        {
          label: "Open Image",
          click: () => {
            if (params.srcURL && !params.srcURL.startsWith("data:")) {
              openExternalUrl(params.srcURL);
            }
          },
          enabled: !params.srcURL.startsWith("data:")
        },
        {
          label: "Copy Image",
          click: () => {
            void copyImageFromUrl(params.srcURL).catch((error) => rememberLog(`Copy image failed: ${error.message}`));
          }
        },
        {
          label: "Copy Image Address",
          click: () => clipboard.writeText(params.srcURL)
        },
        {
          label: "Save Image As...",
          click: () => {
            void saveImageFromUrl(params.srcURL).catch((error) => rememberLog(`Save image failed: ${error.message}`));
          }
        }
      );
    }
    if (hasLink) {
      if (template.length) template.push({ type: "separator" });
      template.push(
        {
          label: "Open Link",
          click: () => openExternalUrl(params.linkURL)
        },
        {
          label: "Copy Link",
          click: () => clipboard.writeText(params.linkURL)
        }
      );
    }
    const suggestions = Array.isArray(params.dictionarySuggestions) ? params.dictionarySuggestions : [];
    if (isEditable && params.misspelledWord && suggestions.length > 0) {
      if (template.length) template.push({ type: "separator" });
      for (const suggestion of suggestions.slice(0, 5)) {
        template.push({
          label: suggestion,
          click: () => window2.webContents.replaceMisspelling(suggestion)
        });
      }
      template.push({ type: "separator" });
      template.push({
        label: "Add to dictionary",
        click: () => window2.webContents.session.addWordToSpellCheckerDictionary(params.misspelledWord)
      });
    }
    if (hasSelection || isEditable) {
      if (template.length) template.push({ type: "separator" });
      if (isEditable) {
        template.push(
          { role: "cut", enabled: params.editFlags.canCut },
          { role: "copy", enabled: params.editFlags.canCopy },
          { role: "paste", enabled: params.editFlags.canPaste },
          { type: "separator" },
          { role: "selectAll", enabled: params.editFlags.canSelectAll }
        );
      } else {
        template.push({ role: "copy", enabled: params.editFlags.canCopy });
      }
    }
    if (!template.length) {
      template.push({ role: "selectAll" });
    }
    Menu.buildFromTemplate(template).popup({ window: window2 });
  });
}
function isAudioCapturePermission(permission, details) {
  if (permission === "audioCapture") {
    return true;
  }
  if (permission !== "media") {
    return false;
  }
  const mediaTypes = details?.mediaTypes;
  if (!Array.isArray(mediaTypes) || mediaTypes.length === 0) {
    return true;
  }
  return mediaTypes.includes("audio") && !mediaTypes.includes("video");
}
function installMediaPermissions() {
  session.defaultSession.setPermissionRequestHandler((_webContents, permission, callback, details) => {
    callback(isAudioCapturePermission(permission, details));
  });
  session.defaultSession.setPermissionCheckHandler((_webContents, permission, _origin, details) => {
    if (permission === "media" || permission === "audioCapture") {
      const mediaType = details?.mediaType;
      if (mediaType === "video") {
        return false;
      }
      return true;
    }
    return false;
  });
}
var OAUTH_SESSION_PARTITION = "persist:hermes-remote-oauth";
function getOauthSession() {
  if (oauthSession || !app.isReady()) return oauthSession;
  oauthSession = session.fromPartition(OAUTH_SESSION_PARTITION);
  return oauthSession;
}
async function hasOauthSessionCookie(baseUrl) {
  const sess = getOauthSession();
  if (!sess) return false;
  const parsed = new URL(baseUrl);
  try {
    const cookies = await sess.cookies.get({ url: baseUrl });
    return cookiesHaveSession(cookies);
  } catch {
    try {
      const cookies = await sess.cookies.get({ domain: parsed.hostname });
      return cookiesHaveSession(cookies);
    } catch {
      return false;
    }
  }
}
async function hasLiveOauthSession(baseUrl) {
  const sess = getOauthSession();
  if (!sess) return false;
  const parsed = new URL(baseUrl);
  try {
    const cookies = await sess.cookies.get({ url: baseUrl });
    return cookiesHaveLiveSession(cookies);
  } catch {
    try {
      const cookies = await sess.cookies.get({ domain: parsed.hostname });
      return cookiesHaveLiveSession(cookies);
    } catch {
      return false;
    }
  }
}
async function clearOauthSession(baseUrl) {
  const sess = getOauthSession();
  if (!sess) return;
  try {
    const cookies = await sess.cookies.get(baseUrl ? { url: baseUrl } : {});
    await Promise.all(
      cookies.map((c) => {
        const scheme = c.secure ? "https" : "http";
        const cookieUrl = `${scheme}://${c.domain.replace(/^\./, "")}${c.path || "/"}`;
        return sess.cookies.remove(cookieUrl, c.name).catch(() => void 0);
      })
    );
  } catch {
  }
}
function openOauthLoginWindow(baseUrl) {
  return new Promise((resolve, reject) => {
    if (!app.isReady()) {
      reject(new Error("Desktop is not ready to start an OAuth login."));
      return;
    }
    const sess = getOauthSession();
    if (!sess) {
      reject(new Error("OAuth session partition is unavailable."));
      return;
    }
    let settled = false;
    let win = null;
    let pollTimer = null;
    const finish = (err) => {
      if (settled) return;
      settled = true;
      if (pollTimer) clearInterval(pollTimer);
      try {
        if (win && !win.isDestroyed()) win.destroy();
      } catch {
      }
      if (err) reject(err);
      else resolve({ baseUrl, ok: true });
    };
    const checkCookie = async () => {
      if (settled) return;
      if (await hasOauthSessionCookie(baseUrl)) finish(null);
    };
    try {
      win = new BrowserWindow({
        width: 520,
        height: 720,
        title: "Sign in to Hermes gateway",
        autoHideMenuBar: true,
        webPreferences: {
          contextIsolation: true,
          nodeIntegration: false,
          sandbox: true,
          session: sess,
          webSecurity: true
        }
      });
    } catch (error) {
      finish(error instanceof Error ? error : new Error(String(error)));
      return;
    }
    win.webContents.on("did-navigate", () => void checkCookie());
    win.webContents.on("did-redirect-navigation", () => void checkCookie());
    win.webContents.on("did-frame-navigate", () => void checkCookie());
    pollTimer = setInterval(() => void checkCookie(), 750);
    win.on("closed", () => {
      if (!settled) finish(new Error("Login window closed before authentication completed."));
    });
    const loginUrl = `${normalizeRemoteBaseUrl(baseUrl)}/login`;
    win.loadURL(loginUrl).catch((error) => {
      finish(error instanceof Error ? error : new Error(String(error)));
    });
  });
}
function fetchJsonViaOauthSession(url, options = {}) {
  return new Promise((resolve, reject) => {
    const sess = getOauthSession();
    if (!sess) {
      reject(new Error("OAuth session partition is unavailable."));
      return;
    }
    let parsed;
    try {
      parsed = new URL(url);
    } catch (error) {
      reject(new Error(`Invalid URL: ${error.message}`));
      return;
    }
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      reject(new Error(`Unsupported Hermes backend URL protocol: ${parsed.protocol}`));
      return;
    }
    const body = serializeJsonBody(options.body);
    const timeoutMs = resolveTimeoutMs(options.timeoutMs, DEFAULT_FETCH_TIMEOUT_MS);
    const request = electronNet.request({
      method: options.method || "GET",
      url,
      session: sess,
      useSessionCookies: true,
      redirect: "follow"
    });
    setJsonRequestHeaders(request);
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      try {
        request.abort();
      } catch {
      }
      reject(new Error(`Timed out connecting to Hermes backend after ${timeoutMs}ms`));
    }, timeoutMs);
    request.on("response", (res) => {
      const chunks = [];
      res.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
      res.on("end", () => {
        if (timedOut) return;
        clearTimeout(timer);
        const text = Buffer.concat(chunks).toString("utf8");
        const statusCode = res.statusCode || 500;
        if (statusCode >= 400) {
          const err = new Error(`${statusCode}: ${text || ""}`);
          err.statusCode = statusCode;
          reject(err);
          return;
        }
        if (!text) {
          resolve(null);
          return;
        }
        const looksHtml = /^\s*<(?:!doctype|html)/i.test(text);
        const contentType = String(res.headers["content-type"] || res.headers["Content-Type"] || "");
        if (looksHtml || contentType.includes("text/html")) {
          reject(new Error(`Expected JSON from ${url} but got HTML (status ${statusCode}).`));
          return;
        }
        try {
          resolve(JSON.parse(text));
        } catch {
          reject(new Error(`Invalid JSON from ${url} (status ${statusCode}): ${text.slice(0, 200)}`));
        }
      });
    });
    request.on("error", (error) => {
      if (timedOut) return;
      clearTimeout(timer);
      reject(error);
    });
    if (body) request.write(body);
    request.end();
  });
}
async function mintGatewayWsTicket(baseUrl) {
  const body = await fetchJsonViaOauthSession(`${baseUrl}/api/auth/ws-ticket`, {
    method: "POST",
    timeoutMs: 8e3
  });
  const ticket = body?.ticket;
  if (!ticket || typeof ticket !== "string") {
    throw new Error("Gateway did not return a WS ticket.");
  }
  return ticket;
}
async function freshGatewayWsUrl(profile) {
  const connection = await ensureBackend(profile);
  if (connection.authMode === "oauth") {
    const ticket = await mintGatewayWsTicket(connection.baseUrl);
    return buildGatewayWsUrlWithTicket(connection.baseUrl, ticket);
  }
  return connection.wsUrl;
}
function encryptDesktopSecret(value) {
  return encryptDesktopSecretStrict(value, safeStorage);
}
function decryptDesktopSecret(secret) {
  if (!secret || typeof secret !== "object") {
    return "";
  }
  const value = String(secret.value || "");
  if (!value) {
    return "";
  }
  if (secret.encoding === "safeStorage") {
    try {
      return safeStorage.decryptString(Buffer.from(value, "base64"));
    } catch {
      return "";
    }
  }
  return value;
}
function sanitizeConnectionProfiles(raw) {
  if (!raw || typeof raw !== "object") {
    return {};
  }
  const out = {};
  for (const [name, entry] of Object.entries(raw)) {
    if (!entry || typeof entry !== "object") {
      continue;
    }
    if (name !== "default" && !PROFILE_NAME_RE.test(name)) {
      continue;
    }
    const cleaned = { mode: entry.mode === "remote" ? "remote" : "local" };
    const url = String(entry.url || "").trim();
    if (url) {
      cleaned.url = url;
    }
    cleaned.authMode = normAuthMode(entry.authMode);
    if (entry.token && typeof entry.token === "object") {
      cleaned.token = entry.token;
    }
    out[name] = cleaned;
  }
  return out;
}
function readDesktopConnectionConfig() {
  let mtime = null;
  try {
    mtime = fs.statSync(DESKTOP_CONNECTION_CONFIG_PATH).mtimeMs;
  } catch {
    mtime = null;
  }
  if (connectionConfigCache && connectionConfigCacheMtime === mtime) {
    return connectionConfigCache;
  }
  let config = { mode: "local", remote: {}, profiles: {} };
  try {
    const raw = fs.readFileSync(DESKTOP_CONNECTION_CONFIG_PATH, "utf8");
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object") {
      const remote = parsed.remote && typeof parsed.remote === "object" ? parsed.remote : {};
      remote.authMode = remote.authMode === "oauth" ? "oauth" : "token";
      config = {
        mode: parsed.mode === "remote" ? "remote" : "local",
        remote,
        // Per-profile remote overrides: each profile may point at its own
        // backend (local spawn or its own remote URL). Preserved verbatim so
        // profileRemoteOverride() can resolve them; normalized lazily on save.
        profiles: sanitizeConnectionProfiles(parsed.profiles)
      };
    }
  } catch {
  }
  connectionConfigCache = config;
  connectionConfigCacheMtime = mtime;
  return config;
}
function writeDesktopConnectionConfig(config) {
  fs.mkdirSync(path.dirname(DESKTOP_CONNECTION_CONFIG_PATH), { recursive: true });
  writeFileAtomic(DESKTOP_CONNECTION_CONFIG_PATH, JSON.stringify(config, null, 2));
  connectionConfigCache = config;
  connectionConfigCacheMtime = fs.statSync(DESKTOP_CONNECTION_CONFIG_PATH).mtimeMs;
}
function readActiveDesktopProfile() {
  try {
    const raw = fs.readFileSync(DESKTOP_PROFILE_CONFIG_PATH, "utf8");
    const parsed = JSON.parse(raw);
    const name = parsed && typeof parsed.profile === "string" ? parsed.profile.trim() : "";
    if (name && (name === "default" || PROFILE_NAME_RE.test(name))) {
      return name;
    }
  } catch {
  }
  return null;
}
function writeActiveDesktopProfile(name) {
  const value = typeof name === "string" ? name.trim() : "";
  if (value && value !== "default" && !PROFILE_NAME_RE.test(value)) {
    throw new Error(`Invalid profile name: ${value}`);
  }
  fs.mkdirSync(path.dirname(DESKTOP_PROFILE_CONFIG_PATH), { recursive: true });
  writeFileAtomic(DESKTOP_PROFILE_CONFIG_PATH, JSON.stringify({ profile: value || null }, null, 2));
  return value || null;
}
async function sanitizeDesktopConnectionConfig(config = readDesktopConnectionConfig(), profile = null) {
  const key = connectionScopeKey(profile);
  const scoped = key ? config.profiles?.[key] || null : null;
  const block = key ? scoped || {} : config.remote || {};
  const envOverride = key ? false : Boolean(process.env.HERMES_DESKTOP_REMOTE_URL);
  const remoteToken = decryptDesktopSecret(block.token);
  const authMode = normAuthMode(block.authMode);
  const remoteUrl = envOverride ? String(process.env.HERMES_DESKTOP_REMOTE_URL || "") : String(block.url || "");
  const mode = envOverride || (key ? scoped?.mode : config.mode) === "remote" ? "remote" : "local";
  let remoteOauthConnected = false;
  if (authMode === "oauth" && remoteUrl) {
    try {
      remoteOauthConnected = await hasLiveOauthSession(remoteUrl);
    } catch {
      remoteOauthConnected = false;
    }
  }
  return {
    mode,
    // Echo the scope back so the UI knows which profile (if any) this reflects.
    profile: key,
    remoteAuthMode: authMode,
    remoteOauthConnected,
    remoteUrl,
    remoteTokenPreview: tokenPreview(remoteToken),
    remoteTokenSet: Boolean(remoteToken),
    // The env override only forces the global/primary connection; a per-profile
    // scope is never overridden by HERMES_DESKTOP_REMOTE_URL.
    envOverride
  };
}
function buildRemoteBlock(remoteUrl, authMode, token) {
  if (authMode !== "oauth" && !decryptDesktopSecret(token)) {
    throw new Error("Remote gateway session token is required.");
  }
  return { url: normalizeRemoteBaseUrl(remoteUrl), authMode, token };
}
function coerceDesktopConnectionConfig(input = {}, existing = readDesktopConnectionConfig(), options = {}) {
  const persistToken = options.persistToken !== false;
  const key = connectionScopeKey(input.profile);
  const mode = input.mode === "remote" ? "remote" : "local";
  const existingBlock = key ? existing.profiles?.[key] || {} : existing.remote || {};
  const remoteUrl = String(input.remoteUrl ?? existingBlock.url ?? "").trim();
  const authMode = resolveAuthMode(input.remoteAuthMode, existingBlock.authMode);
  const incomingToken = typeof input.remoteToken === "string" ? input.remoteToken.trim() : "";
  const nextToken = incomingToken ? persistToken ? encryptDesktopSecret(incomingToken) : { encoding: "plain", value: incomingToken } : existingBlock.token;
  if (key) {
    const profiles = { ...existing.profiles || {} };
    if (mode === "remote") {
      profiles[key] = { mode: "remote", ...buildRemoteBlock(remoteUrl, authMode, nextToken) };
    } else {
      delete profiles[key];
    }
    return { mode: existing.mode === "remote" ? "remote" : "local", remote: existing.remote || {}, profiles };
  }
  const nextRemote = mode === "remote" ? buildRemoteBlock(remoteUrl, authMode, nextToken) : { url: remoteUrl ? normalizeRemoteBaseUrl(remoteUrl) : remoteUrl, authMode, token: nextToken };
  return { mode, remote: nextRemote, profiles: existing.profiles || {} };
}
async function buildRemoteConnection(rawUrl, authMode, token, source) {
  const baseUrl = normalizeRemoteBaseUrl(rawUrl);
  if (authMode === "oauth") {
    if (!await hasLiveOauthSession(baseUrl)) {
      const err = new Error(
        'Remote Hermes gateway uses OAuth, but you are not signed in. Open Settings \u2192 Gateway and click "Sign in", or switch back to Local.'
      );
      err.needsOauthLogin = true;
      throw err;
    }
    let ticket;
    try {
      ticket = await mintGatewayWsTicket(baseUrl);
    } catch (error) {
      const err = new Error(
        'Your remote gateway session has expired. Open Settings \u2192 Gateway and click "Sign in" again.'
      );
      err.needsOauthLogin = true;
      err.cause = error;
      throw err;
    }
    return {
      baseUrl,
      mode: "remote",
      source,
      authMode: "oauth",
      // No static token in OAuth mode; REST is cookie-authed via the partition.
      token: null,
      wsUrl: buildGatewayWsUrlWithTicket(baseUrl, ticket)
    };
  }
  if (!token) {
    throw new Error(
      "Remote Hermes gateway is selected, but no session token is saved. Open Settings \u2192 Gateway and save a token, or switch back to Local."
    );
  }
  return {
    baseUrl,
    mode: "remote",
    source,
    authMode: "token",
    token,
    wsUrl: buildGatewayWsUrl(baseUrl, token)
  };
}
async function resolveRemoteBackend(profile) {
  const config = readDesktopConnectionConfig();
  const override = profileRemoteOverride(config, profile);
  if (override) {
    const token2 = override.authMode === "oauth" ? null : decryptDesktopSecret(override.token);
    return buildRemoteConnection(override.url, override.authMode, token2, "profile");
  }
  const rawEnvUrl = process.env.HERMES_DESKTOP_REMOTE_URL;
  const rawEnvToken = process.env.HERMES_DESKTOP_REMOTE_TOKEN;
  if (rawEnvUrl) {
    if (!rawEnvToken) {
      throw new Error(
        "HERMES_DESKTOP_REMOTE_URL is set but HERMES_DESKTOP_REMOTE_TOKEN is not. Both must be provided to connect to a remote Hermes backend."
      );
    }
    return buildRemoteConnection(rawEnvUrl, "token", rawEnvToken, "env");
  }
  if (config.mode !== "remote") {
    return null;
  }
  const authMode = normAuthMode(config.remote?.authMode);
  const token = authMode === "oauth" ? null : decryptDesktopSecret(config.remote?.token);
  return buildRemoteConnection(config.remote?.url, authMode, token, "settings");
}
function profileHasRemoteOverride(profile) {
  return Boolean(profileRemoteOverride(readDesktopConnectionConfig(), profile));
}
function configuredRemoteProfileNames() {
  const config = readDesktopConnectionConfig();
  return Object.keys(config.profiles || {}).filter((name) => profileRemoteOverride(config, name));
}
function globalRemoteActive() {
  if (process.env.HERMES_DESKTOP_REMOTE_URL) {
    return true;
  }
  return readDesktopConnectionConfig().mode === "remote";
}
async function fetchJsonForProfile(profile, path2) {
  return requestJsonForProfile(profile, path2, "GET");
}
async function requestJsonForProfile(profile, path2, method, body) {
  const conn = await ensureBackend(profile);
  const url = `${conn.baseUrl}${path2}`;
  const opts = { method, body, timeoutMs: DEFAULT_FETCH_TIMEOUT_MS };
  return conn.authMode === "oauth" ? fetchJsonViaOauthSession(url, opts) : fetchJson(url, conn.token, opts);
}
async function probeRemoteAuthMode(rawUrl) {
  const baseUrl = normalizeRemoteBaseUrl(rawUrl);
  let status;
  try {
    status = await fetchPublicJson(`${baseUrl}/api/status`, { timeoutMs: 8e3 });
  } catch (error) {
    return {
      baseUrl,
      reachable: false,
      authMode: "unknown",
      providers: [],
      version: null,
      error: error instanceof Error ? error.message : String(error)
    };
  }
  const authRequired = authModeFromStatus(status) === "oauth";
  let providers = [];
  if (authRequired) {
    try {
      const body = await fetchPublicJson(`${baseUrl}/api/auth/providers`, { timeoutMs: 8e3 });
      if (Array.isArray(body?.providers)) {
        providers = body.providers.filter((p) => p && typeof p === "object").map((p) => ({
          name: String(p.name || ""),
          displayName: String(p.display_name || p.name || ""),
          supportsPassword: Boolean(p.supports_password)
        })).filter((p) => p.name);
      }
    } catch {
    }
  }
  return {
    baseUrl,
    reachable: true,
    authMode: authRequired ? "oauth" : "token",
    providers,
    version: status?.version || null,
    error: null
  };
}
async function testDesktopConnectionConfig(input = {}) {
  const config = coerceDesktopConnectionConfig(input, readDesktopConnectionConfig(), { persistToken: false });
  const key = connectionScopeKey(input.profile);
  const block = key ? config.profiles?.[key] || null : config.remote;
  const wantRemote = block?.mode === "remote" || !key && config.mode === "remote" || input.mode === "remote" && block;
  let baseUrl;
  let token = null;
  let authMode = "token";
  if (wantRemote && block?.url) {
    baseUrl = normalizeRemoteBaseUrl(block.url);
    authMode = normAuthMode(block.authMode);
    if (authMode !== "oauth") {
      token = decryptDesktopSecret(block.token);
    }
  } else {
    const remote = await resolveRemoteBackend(key) || await startHermes();
    baseUrl = remote.baseUrl;
    token = remote.token;
    authMode = normAuthMode(remote.authMode);
  }
  const status = await fetchJson(`${baseUrl}/api/status`, token, { timeoutMs: 8e3 });
  const wsUrl = await resolveTestWsUrl(baseUrl, authMode, token, { mintTicket: mintGatewayWsTicket });
  if (wsUrl && typeof globalThis.WebSocket === "function") {
    const probe = await probeGatewayWebSocket(wsUrl, { WebSocketImpl: globalThis.WebSocket });
    if (!probe.ok) {
      throw new Error(
        `Reached the gateway over HTTP, but the live WebSocket (/api/ws) connection failed: ${probe.reason} The HTTP check can pass while the WebSocket is blocked by a proxy, firewall, or gateway auth/origin guard.`
      );
    }
  }
  return {
    ok: true,
    baseUrl,
    version: status?.version || null
  };
}
function resetBootProgressForReconnect() {
  updateBootProgress(
    {
      error: null,
      message: "Restarting desktop connection",
      phase: "backend.resolve",
      progress: 4,
      running: true
    },
    { allowDecrease: true }
  );
}
function resetHermesConnection() {
  connectionPromise = null;
  backendStartFailure = null;
  if (hermesProcess && !hermesProcess.killed) {
    hermesProcess.kill("SIGTERM");
  }
  hermesProcess = null;
  resetBootProgressForReconnect();
}
async function teardownPrimaryBackendAndWait() {
  const dying = hermesProcess && !hermesProcess.killed ? hermesProcess : null;
  resetHermesConnection();
  await waitForBackendExit(dying);
}
async function waitForBackendExit(child, timeoutMs = 5e3) {
  if (!child) {
    return;
  }
  if (child.exitCode !== null || child.signalCode !== null) {
    return;
  }
  await new Promise((resolve) => {
    const timer = setTimeout(() => {
      try {
        if (IS_WINDOWS && Number.isInteger(child.pid)) {
          forceKillProcessTree(child.pid);
        } else {
          child.kill("SIGKILL");
        }
      } catch {
      }
      resolve();
    }, timeoutMs);
    child.once("exit", () => {
      clearTimeout(timer);
      resolve();
    });
  });
}
function primaryProfileKey() {
  return readActiveDesktopProfile() || "default";
}
async function ensureBackend(profile) {
  const key = profile && String(profile).trim() ? String(profile).trim() : primaryProfileKey();
  if (key === primaryProfileKey()) {
    return startHermes();
  }
  const existing = backendPool.get(key);
  if (existing) {
    existing.lastActiveAt = Date.now();
    return existing.connectionPromise;
  }
  evictLruPoolBackends(POOL_MAX_BACKENDS - 1);
  const entry = { process: null, port: null, token: null, connectionPromise: null, lastActiveAt: Date.now() };
  entry.connectionPromise = spawnPoolBackend(key, entry).catch((error) => {
    backendPool.delete(key);
    throw error;
  });
  backendPool.set(key, entry);
  startPoolIdleReaper();
  return entry.connectionPromise;
}
function touchPoolBackend(profile) {
  const key = profile && String(profile).trim() ? String(profile).trim() : null;
  if (!key) return;
  const entry = backendPool.get(key);
  if (entry) entry.lastActiveAt = Date.now();
}
function evictLruPoolBackends(keep) {
  if (backendPool.size <= keep) return;
  const now = Date.now();
  const evictable = [...backendPool.entries()].filter(([, entry]) => now - (entry.lastActiveAt || 0) > POOL_KEEPALIVE_FRESH_MS).sort((a, b) => (a[1].lastActiveAt || 0) - (b[1].lastActiveAt || 0));
  let removable = backendPool.size - Math.max(0, keep);
  for (const [profile] of evictable) {
    if (removable <= 0) break;
    rememberLog(`Evicting idle profile backend "${profile}" (LRU cap ${POOL_MAX_BACKENDS})`);
    stopPoolBackend(profile);
    removable -= 1;
  }
}
function startPoolIdleReaper() {
  if (poolIdleReaper) return;
  poolIdleReaper = setInterval(() => {
    const now = Date.now();
    for (const [profile, entry] of [...backendPool.entries()]) {
      if (now - (entry.lastActiveAt || 0) > POOL_IDLE_MS) {
        rememberLog(`Reaping idle profile backend "${profile}" (idle > ${Math.round(POOL_IDLE_MS / 1e3)}s)`);
        stopPoolBackend(profile);
      }
    }
    if (backendPool.size === 0 && poolIdleReaper) {
      clearInterval(poolIdleReaper);
      poolIdleReaper = null;
    }
  }, 6e4);
  if (typeof poolIdleReaper.unref === "function") poolIdleReaper.unref();
}
async function spawnPoolBackend(profile, entry) {
  const remote = await resolveRemoteBackend(profile);
  if (remote) {
    await waitForHermes(remote.baseUrl, remote.token);
    return {
      ...remote,
      profile,
      logs: hermesLog.slice(-80),
      ...getWindowState()
    };
  }
  const token = crypto.randomBytes(32).toString("base64url");
  const dashboardArgs = ["--profile", profile, "dashboard", "--no-open", "--host", "127.0.0.1", "--port", "0"];
  const backend = await ensureRuntime(resolveHermesBackend(dashboardArgs));
  const hermesCwd = resolveHermesCwd();
  const webDist = resolveWebDist();
  const readyFile = backend.readyFile ? makeDashboardReadyFile() : null;
  rememberLog(`Starting Hermes backend for profile "${profile}" via ${backend.label}`);
  const child = spawn(
    backend.command,
    backend.args,
    hiddenWindowsChildOptions({
      cwd: hermesCwd,
      env: {
        ...process.env,
        HERMES_HOME,
        ...backend.env,
        // Pin the gateway's tool/terminal cwd to the same directory we chose for
        // the child process. Inherited TERMINAL_CWD (or a stale config bridge)
        // can still point at the install dir even when spawn cwd is home.
        TERMINAL_CWD: hermesCwd,
        HERMES_DASHBOARD_SESSION_TOKEN: token,
        // Marks this dashboard backend as desktop-spawned so it runs the cron
        // scheduler tick loop (the gateway isn't running under the app).
        HERMES_DESKTOP: "1",
        HERMES_WEB_DIST: webDist,
        ...readyFile ? { HERMES_DESKTOP_READY_FILE: readyFile } : {}
      },
      shell: backend.shell,
      stdio: ["ignore", "pipe", "pipe"]
    })
  );
  entry.process = child;
  entry.token = token;
  child.stdout.on("data", rememberLog);
  child.stderr.on("data", rememberLog);
  let ready = false;
  let rejectStart = null;
  const startFailed = new Promise((_resolve, reject) => {
    rejectStart = reject;
  });
  child.once("error", (error) => {
    rememberLog(`Hermes backend for profile "${profile}" failed to start: ${error.message}`);
    backendPool.delete(profile);
    rejectStart?.(error);
  });
  child.once("exit", (code, signal) => {
    rememberLog(`Hermes backend for profile "${profile}" exited (${signal || code})`);
    backendPool.delete(profile);
    if (!ready) {
      rejectStart?.(
        new Error(`Hermes backend for profile "${profile}" exited before it became ready (${signal || code}).`)
      );
    }
  });
  const port = await Promise.race([waitForDashboardPortAnnouncement(child, { readyFile }), startFailed]);
  if (readyFile) {
    fs.unlink(readyFile, () => {
    });
  }
  entry.port = port;
  const baseUrl = `http://127.0.0.1:${port}`;
  await Promise.race([waitForHermes(baseUrl, token), startFailed]);
  ready = true;
  const authToken = await adoptServedDashboardToken(baseUrl, token, {
    childAlive: () => child.exitCode === null && !child.killed,
    label: `Hermes backend for profile "${profile}"`,
    rememberLog
  });
  entry.token = authToken;
  return {
    baseUrl,
    mode: "local",
    source: "local",
    authMode: "token",
    token: authToken,
    profile,
    wsUrl: `ws://127.0.0.1:${port}/api/ws?token=${encodeURIComponent(authToken)}`,
    logs: hermesLog.slice(-80),
    ...getWindowState()
  };
}
function stopPoolBackend(profile) {
  const entry = backendPool.get(profile);
  if (!entry) return;
  backendPool.delete(profile);
  if (entry.process && !entry.process.killed) {
    try {
      entry.process.kill("SIGTERM");
    } catch {
    }
  }
}
async function teardownPoolBackendAndWait(profile) {
  const entry = backendPool.get(profile);
  if (!entry) return;
  backendPool.delete(profile);
  if (entry.process && !entry.process.killed) {
    try {
      entry.process.kill("SIGTERM");
    } catch {
    }
  }
  await waitForBackendExit(entry.process);
}
function stopAllPoolBackends() {
  for (const profile of [...backendPool.keys()]) {
    stopPoolBackend(profile);
  }
}
function profileNameFromDeleteRequest(request) {
  if (!request || String(request.method || "GET").toUpperCase() !== "DELETE") {
    return null;
  }
  const match = String(request.path || "").match(/^\/api\/profiles\/([^/?#]+)(?:[?#].*)?$/);
  if (!match) {
    return null;
  }
  let raw = "";
  try {
    raw = decodeURIComponent(match[1]);
  } catch {
    return null;
  }
  const name = raw.trim();
  if (!name) {
    return null;
  }
  if (name.toLowerCase() === "default") {
    return "default";
  }
  return name.toLowerCase();
}
async function prepareProfileDeleteRequest(request) {
  const profile = profileNameFromDeleteRequest(request);
  if (!profile || profile === "default" || !PROFILE_NAME_RE.test(profile)) {
    return;
  }
  if (profile === primaryProfileKey()) {
    writeActiveDesktopProfile("default");
    await teardownPrimaryBackendAndWait();
    return;
  }
  await teardownPoolBackendAndWait(profile);
}
async function startHermes() {
  if (bootstrapFailure) {
    throw bootstrapFailure;
  }
  if (backendStartFailure) {
    throw backendStartFailure;
  }
  if (connectionPromise) return connectionPromise;
  connectionPromise = (async () => {
    await advanceBootProgress("backend.resolve", "Resolving Hermes backend", 8);
    const remote = await resolveRemoteBackend(primaryProfileKey());
    if (remote) {
      await advanceBootProgress("backend.remote", `Connecting to remote Hermes backend at ${remote.baseUrl}`, 24);
      await waitForHermes(remote.baseUrl, remote.token);
      updateBootProgress({
        phase: "backend.ready",
        message: "Remote Hermes backend is ready",
        progress: 94,
        running: true,
        error: null
      });
      return {
        baseUrl: remote.baseUrl,
        mode: "remote",
        source: remote.source,
        authMode: remote.authMode || "token",
        token: remote.token,
        wsUrl: remote.wsUrl,
        logs: hermesLog.slice(-80),
        ...getWindowState()
      };
    }
    await waitForUpdateToFinish();
    const token = crypto.randomBytes(32).toString("base64url");
    const dashboardArgs = ["dashboard", "--no-open", "--host", "127.0.0.1", "--port", "0"];
    const activeProfile = readActiveDesktopProfile();
    if (activeProfile) {
      dashboardArgs.unshift("--profile", activeProfile);
    }
    await advanceBootProgress("backend.runtime", "Resolving Hermes runtime", 28);
    const backend = await ensureRuntime(resolveHermesBackend(dashboardArgs));
    const hermesCwd = resolveHermesCwd();
    const webDist = resolveWebDist();
    const readyFile = backend.readyFile ? makeDashboardReadyFile() : null;
    await advanceBootProgress("backend.spawn", `Starting Hermes backend via ${backend.label}`, 84);
    rememberLog(`Starting Hermes backend via ${backend.label}`);
    hermesProcess = spawn(
      backend.command,
      backend.args,
      hiddenWindowsChildOptions({
        cwd: hermesCwd,
        env: {
          ...process.env,
          // Explicitly pin HERMES_HOME for the child so Python's get_hermes_home()
          // resolves to the SAME location our resolveHermesHome() picked. Without
          // this pin, Python falls back to ~/.hermes on every platform — fine on
          // mac/linux (where our default matches), but on Windows our default is
          // %LOCALAPPDATA%\hermes, which differs from C:\Users\<u>\.hermes.
          // Mismatch would split config / sessions / .env / logs across two
          // directories. install.ps1 sets HERMES_HOME via setx; the desktop
          // can't reliably do that, so we set it inline for every spawn.
          HERMES_HOME,
          ...backend.env,
          TERMINAL_CWD: hermesCwd,
          HERMES_DASHBOARD_SESSION_TOKEN: token,
          // Marks this dashboard backend as desktop-spawned so it runs the cron
          // scheduler tick loop (the gateway isn't running under the app).
          HERMES_DESKTOP: "1",
          HERMES_WEB_DIST: webDist,
          ...readyFile ? { HERMES_DESKTOP_READY_FILE: readyFile } : {}
        },
        shell: backend.shell,
        stdio: ["ignore", "pipe", "pipe"]
      })
    );
    hermesProcess.stdout.on("data", rememberLog);
    hermesProcess.stderr.on("data", rememberLog);
    let backendReady = false;
    let rejectBackendStart = null;
    const backendStartFailed = new Promise((_resolve, reject) => {
      rejectBackendStart = reject;
    });
    hermesProcess.once("error", (error) => {
      rememberLog(`Hermes backend failed to start: ${error.message}`);
      updateBootProgress(
        {
          error: error.message,
          message: `Hermes backend failed to start: ${error.message}`,
          phase: "backend.error",
          running: false
        },
        { allowDecrease: true }
      );
      hermesProcess = null;
      connectionPromise = null;
      sendBackendExit({ code: null, signal: null, error: error.message });
      rejectBackendStart?.(error);
    });
    hermesProcess.once("exit", (code, signal) => {
      rememberLog(`Hermes backend exited (${signal || code})`);
      hermesProcess = null;
      connectionPromise = null;
      sendBackendExit({ code, signal });
      if (!backendReady) {
        const message = `Hermes backend exited before it became ready (${signal || code}).`;
        updateBootProgress(
          {
            error: message,
            message,
            phase: "backend.error",
            running: false
          },
          { allowDecrease: true }
        );
        rejectBackendStart?.(
          new Error(
            `Hermes backend exited before it became ready (${signal || code}). Log: ${DESKTOP_LOG_PATH}
${recentHermesLog()}`
          )
        );
      }
    });
    await advanceBootProgress("backend.port", "Waiting for Hermes backend to launch", 86);
    const port = await Promise.race([waitForDashboardPortAnnouncement(hermesProcess, { readyFile }), backendStartFailed]);
    if (readyFile) {
      fs.unlink(readyFile, () => {
      });
    }
    const baseUrl = `http://127.0.0.1:${port}`;
    await advanceBootProgress("backend.wait", "Waiting for Hermes backend to become ready", 90);
    await Promise.race([waitForHermes(baseUrl, token), backendStartFailed]);
    backendReady = true;
    backendStartFailure = null;
    const authToken = await adoptServedDashboardToken(baseUrl, token, {
      // The exit/error handlers null hermesProcess when the child dies.
      childAlive: () => hermesProcess !== null && hermesProcess.exitCode === null && !hermesProcess.killed,
      rememberLog
    });
    updateBootProgress({
      phase: "backend.ready",
      message: "Hermes backend is ready. Finalizing desktop startup",
      progress: 94,
      running: true,
      error: null
    });
    return {
      baseUrl,
      mode: "local",
      source: "local",
      authMode: "token",
      token: authToken,
      wsUrl: `ws://127.0.0.1:${port}/api/ws?token=${encodeURIComponent(authToken)}`,
      logs: hermesLog.slice(-80),
      ...getWindowState()
    };
  })().catch((error) => {
    const message = error instanceof Error ? error.message : String(error);
    backendStartFailure = error instanceof Error ? error : new Error(message);
    updateBootProgress(
      {
        error: message,
        message: `Desktop boot failed: ${message}`,
        phase: "backend.error",
        running: false
      },
      { allowDecrease: true }
    );
    connectionPromise = null;
    throw error;
  });
  return connectionPromise;
}
function wireCommonWindowHandlers(win) {
  installPreviewShortcut(win);
  installDevToolsShortcut(win);
  installZoomShortcuts(win);
  installContextMenu(win);
  win.webContents.setWindowOpenHandler((details) => {
    openExternalUrl(details.url);
    return { action: "deny" };
  });
  win.webContents.on("will-navigate", (event, url) => {
    if (DEV_SERVER && url.startsWith(DEV_SERVER) || !DEV_SERVER && url.startsWith("file:")) {
      return;
    }
    event.preventDefault();
    openExternalUrl(url);
  });
}
var sessionWindows = createSessionWindowRegistry();
function focusWindow(win) {
  if (!win || win.isDestroyed()) return;
  if (win.isMinimized()) win.restore();
  if (!win.isVisible()) win.show();
  win.focus();
}
function spawnSecondaryWindow({ sessionId, watch, newSession } = {}) {
  const icon = getAppIconPath();
  const win = new BrowserWindow({
    width: SESSION_WINDOW_MIN_WIDTH,
    height: SESSION_WINDOW_MIN_HEIGHT,
    minWidth: SESSION_WINDOW_MIN_WIDTH,
    minHeight: SESSION_WINDOW_MIN_HEIGHT,
    title: "Hermes",
    titleBarStyle: "hidden",
    titleBarOverlay: getTitleBarOverlayOptions(),
    trafficLightPosition: IS_MAC ? WINDOW_BUTTON_POSITION : void 0,
    vibrancy: IS_MAC ? "sidebar" : void 0,
    opacity: windowOpacity(),
    icon,
    // Don't show until the renderer's first themed paint is ready. macOS
    // `vibrancy` ignores `backgroundColor` and paints a translucent OS
    // material (which follows the OS appearance, not the app theme), so a
    // dark-themed app on a light-mode Mac flashes white until the renderer
    // covers it. ready-to-show fires after the boot-time paint in
    // themes/context.tsx, so the window appears already themed.
    show: false,
    backgroundColor: getWindowBackgroundColor(),
    webPreferences: chatWindowWebPreferences(path.join(__dirname, "preload.cjs"))
  });
  if (IS_MAC) {
    win.setWindowButtonPosition?.(WINDOW_BUTTON_POSITION);
  }
  win.once("ready-to-show", () => {
    if (!win.isDestroyed()) win.show();
  });
  win.on("will-enter-full-screen", () => sendWindowStateChanged(true));
  win.on("enter-full-screen", () => sendWindowStateChanged(true));
  win.on("will-leave-full-screen", () => sendWindowStateChanged(false));
  win.on("leave-full-screen", () => sendWindowStateChanged(false));
  wireCommonWindowHandlers(win);
  win.loadURL(
    buildSessionWindowUrl(sessionId, {
      devServer: DEV_SERVER,
      rendererIndexPath: DEV_SERVER ? void 0 : resolveRendererIndex(),
      watch,
      newSession
    })
  );
  return win;
}
function createSessionWindow(sessionId, { watch = false } = {}) {
  return sessionWindows.openOrFocus(sessionId, () => spawnSecondaryWindow({ sessionId, watch }));
}
function createNewSessionWindow() {
  return spawnSecondaryWindow({ newSession: true });
}
var petOverlayWindow = null;
function petOverlayUrl() {
  if (DEV_SERVER) {
    return `${DEV_SERVER.endsWith("/") ? DEV_SERVER.slice(0, -1) : DEV_SERVER}/?win=overlay#/`;
  }
  return `${pathToFileURL(resolveRendererIndex()).toString()}?win=overlay#/`;
}
function spawnPetOverlayWindow(bounds) {
  const win = new BrowserWindow({
    width: Math.max(80, Math.round(bounds?.width || 220)),
    height: Math.max(80, Math.round(bounds?.height || 220)),
    x: Number.isFinite(bounds?.x) ? Math.round(bounds.x) : void 0,
    y: Number.isFinite(bounds?.y) ? Math.round(bounds.y) : void 0,
    frame: false,
    transparent: true,
    resizable: false,
    movable: true,
    minimizable: false,
    maximizable: false,
    fullscreenable: false,
    // Windows/Linux need this so the helper window does not get its own
    // taskbar/alt-tab entry. On macOS, cmd-tab is app-level and this can make
    // the whole app look like it vanished when the only newly-created visible
    // window is a frameless overlay. Use NSPanel + Mission Control hiding below
    // instead, leaving the main Hermes app as the Dock/cmd-tab anchor.
    skipTaskbar: !IS_MAC,
    hasShadow: false,
    alwaysOnTop: true,
    // macOS panels are non-activating helper windows and can float over full
    // screen spaces without becoming the app's main switcher window.
    type: IS_MAC ? "panel" : void 0,
    hiddenInMissionControl: IS_MAC,
    // Non-activating: the overlay must never become the app's key/main window,
    // or it (a frameless, taskbar-skipping panel) becomes the app's switcher
    // anchor and the Hermes icon drops out of cmd/alt-tab — especially when the
    // main window is minimized. We flip this on only while the composer needs
    // the keyboard (see hermes:pet-overlay:set-focusable).
    focusable: false,
    show: false,
    // Fully transparent — the renderer paints only the sprite + bubble.
    backgroundColor: "#00000000",
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
      devTools: true,
      // Keep the sprite animating + bubble updating while the main window is
      // minimized/blurred — the whole point of the overlay.
      backgroundThrottling: false
    }
  });
  win.setAlwaysOnTop(true, IS_MAC ? "floating" : "screen-saver");
  win.setHiddenInMissionControl?.(true);
  try {
    win.setVisibleOnAllWorkspaces(
      true,
      IS_MAC ? { visibleOnFullScreen: true, skipTransformProcessType: true } : void 0
    );
  } catch {
  }
  wireCommonWindowHandlers(win);
  win.once("ready-to-show", () => {
    if (!win.isDestroyed()) win.showInactive();
  });
  win.on("closed", () => {
    if (petOverlayWindow === win) {
      petOverlayWindow = null;
    }
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send("hermes:pet-overlay:control", { type: "pop-in" });
    }
  });
  win.loadURL(petOverlayUrl());
  return win;
}
function openPetOverlay(bounds) {
  if (petOverlayWindow && !petOverlayWindow.isDestroyed()) {
    if (bounds) {
      petOverlayWindow.setBounds({
        x: Math.round(bounds.x),
        y: Math.round(bounds.y),
        width: Math.max(80, Math.round(bounds.width)),
        height: Math.max(80, Math.round(bounds.height))
      });
    }
    petOverlayWindow.showInactive();
    return petOverlayWindow;
  }
  petOverlayWindow = spawnPetOverlayWindow(bounds);
  return petOverlayWindow;
}
function closePetOverlay() {
  if (petOverlayWindow && !petOverlayWindow.isDestroyed()) {
    petOverlayWindow.close();
  }
  petOverlayWindow = null;
}
function createWindow() {
  const icon = getAppIconPath();
  const savedWindowState = readWindowState();
  mainWindow = new BrowserWindow({
    ...computeWindowOptions(savedWindowState, screen.getAllDisplays()),
    minWidth: WINDOW_MIN_WIDTH,
    minHeight: WINDOW_MIN_HEIGHT,
    title: "Hermes",
    // Frameless title bar on every platform so the renderer can paint the
    // "hide sidebar" button (and other left-side titlebar tools) flush with
    // the top edge — matching the macOS layout where the traffic lights sit
    // inside the same band. On Windows/Linux, titleBarOverlay tells Electron
    // to paint native min/max/close in the top-right of the renderer; on
    // macOS it just reserves a content inset alongside the traffic lights.
    titleBarStyle: "hidden",
    titleBarOverlay: getTitleBarOverlayOptions(),
    trafficLightPosition: IS_MAC ? WINDOW_BUTTON_POSITION : void 0,
    vibrancy: IS_MAC ? "sidebar" : void 0,
    opacity: windowOpacity(),
    icon,
    // Hidden until the first themed paint so macOS `vibrancy` (which ignores
    // `backgroundColor` and follows the OS appearance) can't flash a light
    // material before the renderer paints the app theme. See createSessionWindow.
    show: false,
    backgroundColor: getWindowBackgroundColor(),
    // Shared with the secondary session windows (chatWindowWebPreferences) so
    // both keep `backgroundThrottling: false` — the chat transcript streams via
    // a requestAnimationFrame-gated flush that Chromium pauses for blurred
    // windows, stalling the live answer until refocus. See session-windows.cjs.
    webPreferences: chatWindowWebPreferences(path.join(__dirname, "preload.cjs"))
  });
  if (IS_MAC) {
    mainWindow.setWindowButtonPosition?.(WINDOW_BUTTON_POSITION);
    if (icon) {
      app.dock?.setIcon(icon);
    }
  }
  if (!IS_MAC) {
    if (!nativeThemeListenerInstalled) {
      nativeThemeListenerInstalled = true;
      nativeTheme.on("updated", () => {
        mainWindow?.setTitleBarOverlay?.(getTitleBarOverlayOptions());
      });
    }
  }
  if (savedWindowState?.isMaximized) mainWindow.maximize();
  mainWindow.once("ready-to-show", () => {
    if (mainWindow && !mainWindow.isDestroyed()) mainWindow.show();
  });
  mainWindow.on("will-enter-full-screen", () => sendWindowStateChanged(true));
  mainWindow.on("enter-full-screen", () => sendWindowStateChanged(true));
  mainWindow.on("will-leave-full-screen", () => sendWindowStateChanged(false));
  mainWindow.on("leave-full-screen", () => sendWindowStateChanged(false));
  mainWindow.on("resized", schedulePersistWindowState);
  mainWindow.on("moved", schedulePersistWindowState);
  mainWindow.on("maximize", schedulePersistWindowState);
  mainWindow.on("unmaximize", schedulePersistWindowState);
  mainWindow.on("close", () => schedulePersistWindowState.flush());
  mainWindow.on("closed", () => closePetOverlay());
  wireCommonWindowHandlers(mainWindow);
  mainWindow.webContents.on("render-process-gone", (_event, details) => {
    rememberLog(`[renderer] render-process-gone reason=${details?.reason} exitCode=${details?.exitCode}`);
    if (details?.reason === "crashed" || details?.reason === "oom") {
      const now = Date.now();
      rendererReloadTimes = rendererReloadTimes.filter((t) => now - t < RENDERER_RELOAD_WINDOW_MS);
      if (rendererReloadTimes.length >= RENDERER_RELOAD_MAX) {
        rememberLog(
          `[renderer] suppressing reload: ${rendererReloadTimes.length} crashes within ${RENDERER_RELOAD_WINDOW_MS}ms (likely a crash loop)`
        );
        return;
      }
      rendererReloadTimes.push(now);
      setImmediate(() => {
        if (!mainWindow || mainWindow.isDestroyed()) return;
        try {
          mainWindow.webContents.reload();
        } catch (err) {
          rememberLog(`[renderer] reload after crash failed: ${err?.message || err}`);
        }
      });
    }
  });
  mainWindow.webContents.on("unresponsive", () => rememberLog("[renderer] webContents became unresponsive"));
  mainWindow.webContents.on("console-message", (_event, detailsOrLevel, message, line, sourceId) => {
    const details = detailsOrLevel && typeof detailsOrLevel === "object" ? detailsOrLevel : null;
    const level = details ? details.level : detailsOrLevel;
    if (level !== 3) return;
    const text = details ? details.message : message;
    const src = details ? details.sourceUrl : sourceId;
    const lineNo = details ? details.lineNumber : line;
    rememberLog(`[renderer console] ${text} (${src}:${lineNo})`);
  });
  if (DEV_SERVER) {
    mainWindow.loadURL(DEV_SERVER);
  } else {
    mainWindow.loadURL(pathToFileURL(resolveRendererIndex()).toString());
  }
  mainWindow.webContents.once("did-finish-load", () => {
    restorePersistedZoomLevel(mainWindow);
    broadcastBootProgress();
    sendWindowStateChanged();
    startHermes().catch((error) => rememberLog(error.stack || error.message));
  });
}
ipcMain.handle("hermes:connection", async (_event, profile) => ensureBackend(profile));
ipcMain.handle("hermes:connection:revalidate", async () => {
  if (!connectionPromise) {
    return { ok: true, rebuilt: false };
  }
  let conn = null;
  try {
    conn = await connectionPromise;
  } catch {
    return { ok: true, rebuilt: false };
  }
  if (!conn || conn.mode !== "remote" || !conn.baseUrl) {
    return { ok: true, rebuilt: false };
  }
  const base = conn.baseUrl.replace(/\/+$/, "");
  try {
    await fetchPublicJson(`${base}/api/status`, { timeoutMs: 2500 });
    return { ok: true, rebuilt: false };
  } catch {
    rememberLog("Cached remote Hermes backend failed liveness probe; dropping stale connection.");
    resetHermesConnection();
    return { ok: true, rebuilt: true };
  }
});
ipcMain.handle("hermes:backend:touch", async (_event, profile) => {
  touchPoolBackend(profile);
  return { ok: true };
});
ipcMain.handle("hermes:gateway:ws-url", async (_event, profile) => freshGatewayWsUrl(profile));
ipcMain.handle("hermes:window:openSession", async (_event, sessionId, opts) => {
  if (typeof sessionId !== "string" || !sessionId.trim()) {
    return { ok: false, error: "invalid-session-id" };
  }
  createSessionWindow(sessionId.trim(), { watch: opts?.watch === true });
  return { ok: true };
});
ipcMain.handle("hermes:window:openNewSession", async () => {
  createNewSessionWindow();
  return { ok: true };
});
ipcMain.handle("hermes:pet-overlay:open", async (_event, request) => {
  const bounds = request && request.bounds ? request.bounds : request;
  const isScreen = Boolean(request && request.screen);
  let screenBounds = bounds;
  try {
    if (bounds && !isScreen && mainWindow && !mainWindow.isDestroyed()) {
      const content = mainWindow.getContentBounds();
      screenBounds = {
        x: content.x + (bounds.x || 0),
        y: content.y + (bounds.y || 0),
        width: bounds.width,
        height: bounds.height
      };
    }
  } catch {
  }
  openPetOverlay(screenBounds);
  return { ok: true, bounds: screenBounds };
});
ipcMain.handle("hermes:pet-overlay:close", async () => {
  closePetOverlay();
  return { ok: true };
});
ipcMain.on("hermes:pet-overlay:set-bounds", (_event, bounds) => {
  if (!petOverlayWindow || petOverlayWindow.isDestroyed() || !bounds) {
    return;
  }
  petOverlayWindow.setBounds({
    x: Math.round(bounds.x),
    y: Math.round(bounds.y),
    width: Math.max(80, Math.round(bounds.width)),
    height: Math.max(80, Math.round(bounds.height))
  });
});
ipcMain.on("hermes:pet-overlay:ignore-mouse", (_event, ignore) => {
  if (petOverlayWindow && !petOverlayWindow.isDestroyed()) {
    petOverlayWindow.setIgnoreMouseEvents(Boolean(ignore), { forward: true });
  }
});
ipcMain.on("hermes:pet-overlay:set-focusable", (_event, focusable) => {
  if (!petOverlayWindow || petOverlayWindow.isDestroyed()) {
    return;
  }
  petOverlayWindow.setFocusable(Boolean(focusable));
  if (focusable) {
    petOverlayWindow.focus();
  }
});
ipcMain.on("hermes:pet-overlay:state", (_event, payload) => {
  if (petOverlayWindow && !petOverlayWindow.isDestroyed()) {
    petOverlayWindow.webContents.send("hermes:pet-overlay:state", payload);
  }
});
ipcMain.on("hermes:pet-overlay:control", (_event, payload) => {
  if (!mainWindow || mainWindow.isDestroyed()) {
    return;
  }
  if (payload && payload.type === "toggle-app") {
    if (mainWindow.isMinimized() || !mainWindow.isVisible()) {
      mainWindow.show();
      mainWindow.focus();
    } else {
      mainWindow.minimize();
    }
    return;
  }
  if (payload && payload.type === "open-app") {
    if (mainWindow.isMinimized()) {
      mainWindow.restore();
    }
    mainWindow.show();
    mainWindow.focus();
  }
  mainWindow.webContents.send("hermes:pet-overlay:control", payload);
});
ipcMain.handle("hermes:bootstrap:reset", async () => {
  rememberLog("[bootstrap] reset requested by renderer; clearing latched failure");
  await teardownPrimaryBackendAndWait();
  bootstrapFailure = null;
  backendStartFailure = null;
  bootstrapState = {
    active: false,
    manifest: null,
    stages: {},
    error: null,
    log: [],
    startedAt: null,
    completedAt: null,
    unsupportedPlatform: null
  };
  return { ok: true };
});
ipcMain.handle("hermes:bootstrap:repair", async () => {
  rememberLog("[bootstrap] repair requested by renderer; clearing marker + latched failure");
  try {
    if (fileExists(BOOTSTRAP_COMPLETE_MARKER)) {
      fs.rmSync(BOOTSTRAP_COMPLETE_MARKER, { force: true });
    }
  } catch (error) {
    rememberLog(`[bootstrap] failed to remove marker during repair: ${error.message}`);
  }
  bootstrapFailure = null;
  backendStartFailure = null;
  resetHermesConnection();
  return { ok: true };
});
ipcMain.handle("hermes:bootstrap:cancel", async () => {
  if (bootstrapAbortController) {
    try {
      bootstrapAbortController.abort();
    } catch {
    }
    return { ok: true, cancelled: true };
  }
  return { ok: false, cancelled: false };
});
ipcMain.handle("hermes:boot-progress:get", async () => bootProgressState);
ipcMain.handle("hermes:bootstrap:get", async () => getBootstrapState());
ipcMain.handle(
  "hermes:connection-config:get",
  async (_event, profile) => sanitizeDesktopConnectionConfig(readDesktopConnectionConfig(), profile)
);
ipcMain.handle("hermes:connection-config:test", async (_event, payload) => testDesktopConnectionConfig(payload));
ipcMain.handle("hermes:connection-config:probe", async (_event, rawUrl) => probeRemoteAuthMode(rawUrl));
ipcMain.handle("hermes:connection-config:oauth-login", async (_event, rawUrl) => {
  const baseUrl = normalizeRemoteBaseUrl(rawUrl);
  await openOauthLoginWindow(baseUrl);
  return { ok: true, baseUrl, connected: await hasOauthSessionCookie(baseUrl) };
});
ipcMain.handle("hermes:connection-config:oauth-logout", async (_event, rawUrl) => {
  const baseUrl = rawUrl ? normalizeRemoteBaseUrl(rawUrl) : "";
  await clearOauthSession(baseUrl || void 0);
  return { ok: true, connected: baseUrl ? await hasLiveOauthSession(baseUrl) : false };
});
ipcMain.handle("hermes:connection-config:save", async (_event, payload) => {
  const config = coerceDesktopConnectionConfig(payload);
  writeDesktopConnectionConfig(config);
  return sanitizeDesktopConnectionConfig(config, payload?.profile);
});
ipcMain.handle("hermes:connection-config:apply", async (_event, payload) => {
  const config = coerceDesktopConnectionConfig(payload);
  writeDesktopConnectionConfig(config);
  const key = connectionScopeKey(payload?.profile);
  if (key && key !== primaryProfileKey()) {
    stopPoolBackend(key);
  } else {
    await teardownPrimaryBackendAndWait();
    mainWindow?.reload();
  }
  return sanitizeDesktopConnectionConfig(config, payload?.profile);
});
ipcMain.handle("hermes:profile:get", async () => ({ profile: readActiveDesktopProfile() }));
ipcMain.handle("hermes:profile:set", async (_event, name) => {
  const next = writeActiveDesktopProfile(name);
  await teardownPrimaryBackendAndWait();
  mainWindow?.reload();
  return { profile: next };
});
ipcMain.on("hermes:previewShortcutActive", (_event, active) => {
  previewShortcutActive = Boolean(active);
});
ipcMain.handle("hermes:requestMicrophoneAccess", async () => {
  if (!IS_MAC || typeof systemPreferences.askForMediaAccess !== "function") {
    return true;
  }
  return systemPreferences.askForMediaAccess("microphone");
});
async function interceptSessionRequestForRemote(request) {
  if (typeof request?.path !== "string") {
    return void 0;
  }
  const method = (request.method || "GET").toUpperCase();
  let parsed;
  try {
    parsed = new URL(request.path, "http://x");
  } catch {
    return void 0;
  }
  const { pathname, searchParams } = parsed;
  if (method === "GET" && pathname === "/api/profiles/sessions") {
    const remoteProfiles = configuredRemoteProfileNames();
    if (remoteProfiles.length === 0) {
      return void 0;
    }
    const requested = (searchParams.get("profile") || "all").trim() || "all";
    if (requested !== "all") {
      return profileHasRemoteOverride(requested) ? remoteSessionList(requested, searchParams) : void 0;
    }
    return mergeRemoteProfileSessions(searchParams, remoteProfiles);
  }
  if (/^\/api\/sessions\/[^/]+(\/messages)?$/.test(pathname)) {
    const profile = (searchParams.get("profile") || request.profile || "").trim();
    if (!profile) {
      return void 0;
    }
    if (profileHasRemoteOverride(profile)) {
      if (method === "GET") {
        return fetchJsonForProfile(profile, pathname);
      }
      const body = request.body && typeof request.body === "object" ? { ...request.body } : request.body;
      if (body) delete body.profile;
      return requestJsonForProfile(profile, pathname, method, body);
    }
    if (globalRemoteActive()) {
      const sep = pathname.includes("?") ? "&" : "?";
      const path2 = `${pathname}${sep}profile=${encodeURIComponent(profile)}`;
      if (method === "GET") {
        return fetchJsonForProfile(null, path2);
      }
      const body = request.body && typeof request.body === "object" ? { ...request.body, profile } : { profile };
      return requestJsonForProfile(null, path2, method, body);
    }
    return void 0;
  }
  return void 0;
}
var rowsOf = (data) => Array.isArray(data?.sessions) ? data.sessions : [];
async function remoteSessionList(profile, searchParams) {
  const qs = new URLSearchParams(searchParams);
  qs.delete("profile");
  const data = await fetchJsonForProfile(profile, `/api/sessions?${qs}`);
  for (const s of rowsOf(data)) {
    s.profile = profile;
    s.is_default_profile = false;
  }
  return { ...data, sessions: rowsOf(data) };
}
async function mergeRemoteProfileSessions(searchParams, remoteProfiles) {
  const limit = Math.max(1, Number(searchParams.get("limit")) || 20);
  const offset = Math.max(0, Number(searchParams.get("offset")) || 0);
  const order = searchParams.get("order") === "created" ? "started_at" : "last_active";
  const primary = await ensureBackend(null);
  const base = await fetchJson(`${primary.baseUrl}/api/profiles/sessions?${searchParams}`, primary.token, {
    method: "GET",
    timeoutMs: DEFAULT_FETCH_TIMEOUT_MS
  }).catch(() => ({ sessions: [], total: 0, profile_totals: {} }));
  const remoteParams = new URLSearchParams(searchParams);
  remoteParams.set("limit", String(limit + offset));
  remoteParams.set("offset", "0");
  const remoteSet = new Set(remoteProfiles);
  const merged = rowsOf(base).filter((s) => !remoteSet.has(s?.profile));
  const profileTotals = { ...base.profile_totals || {} };
  let total = (Number(base.total) || 0) - remoteProfiles.reduce((n, p) => n + (profileTotals[p] || 0), 0);
  await Promise.all(
    remoteProfiles.map(async (name) => {
      const list = await remoteSessionList(name, remoteParams).catch(() => null);
      if (!list) {
        delete profileTotals[name];
        return;
      }
      const rows = rowsOf(list);
      merged.push(...rows);
      profileTotals[name] = Number(list.total) || rows.length;
      total += profileTotals[name];
    })
  );
  const recency = (s) => s?.[order] ?? s?.started_at ?? 0;
  merged.sort((a, b) => recency(b) - recency(a));
  return { ...base, sessions: merged.slice(offset, offset + limit), total, profile_totals: profileTotals };
}
ipcMain.handle("hermes:api", async (_event, request) => {
  const rerouted = await interceptSessionRequestForRemote(request);
  if (rerouted !== void 0) {
    return rerouted;
  }
  await prepareProfileDeleteRequest(request);
  const profile = request?.profile;
  const connection = await ensureBackend(profile);
  const timeoutMs = resolveTimeoutMs(request?.timeoutMs, DEFAULT_FETCH_TIMEOUT_MS);
  const requestPath = pathWithGlobalRemoteProfile(request.path, profile, {
    globalRemote: globalRemoteActive(),
    profileRemoteOverride: profileHasRemoteOverride(profile)
  });
  const url = `${connection.baseUrl}${requestPath}`;
  if (connection.authMode === "oauth") {
    return fetchJsonViaOauthSession(url, {
      method: request?.method,
      body: request?.body,
      timeoutMs
    });
  }
  return fetchJson(url, connection.token, {
    method: request?.method,
    body: request?.body,
    timeoutMs
  });
});
ipcMain.handle("hermes:notify", (_event, payload) => {
  if (!Notification.isSupported()) return false;
  const actions = Array.isArray(payload?.actions) ? payload.actions : [];
  const notification = new Notification({
    title: payload?.title || "Hermes",
    body: payload?.body || "",
    silent: Boolean(payload?.silent),
    actions: actions.map((action) => ({ type: "button", text: String(action?.text || "") }))
  });
  notification.on("click", () => {
    if (!mainWindow || mainWindow.isDestroyed()) return;
    focusWindow(mainWindow);
    if (payload?.sessionId) {
      mainWindow.webContents.send("hermes:focus-session", payload.sessionId);
    }
  });
  notification.on("action", (_actionEvent, index) => {
    if (!mainWindow || mainWindow.isDestroyed()) return;
    const action = actions[index];
    if (action?.id) {
      mainWindow.webContents.send("hermes:notification-action", { sessionId: payload?.sessionId, actionId: action.id });
    }
  });
  notification.show();
  return true;
});
ipcMain.handle("hermes:readFileDataUrl", async (_event, filePath) => {
  const { resolvedPath } = await resolveReadableFileForIpc(filePath, {
    maxBytes: DATA_URL_READ_MAX_BYTES,
    purpose: "File preview"
  });
  const data = await fs.promises.readFile(resolvedPath);
  return `data:${mimeTypeForPath(resolvedPath)};base64,${data.toString("base64")}`;
});
ipcMain.handle("hermes:readFileText", async (_event, filePath) => {
  const { resolvedPath, stat } = await resolveReadableFileForIpc(filePath, {
    maxBytes: TEXT_PREVIEW_SOURCE_MAX_BYTES,
    purpose: "Text preview"
  });
  const ext = path.extname(resolvedPath).toLowerCase();
  const handle = await fs.promises.open(resolvedPath, "r");
  const bytesToRead = Math.min(stat.size, TEXT_PREVIEW_MAX_BYTES);
  try {
    const buffer = Buffer.alloc(bytesToRead);
    const { bytesRead } = await handle.read(buffer, 0, bytesToRead, 0);
    return {
      binary: looksBinary(buffer.subarray(0, Math.min(bytesRead, 4096))),
      byteSize: stat.size,
      language: PREVIEW_LANGUAGE_BY_EXT[ext] || "text",
      mimeType: mimeTypeForPath(resolvedPath),
      path: resolvedPath,
      text: buffer.subarray(0, bytesRead).toString("utf8"),
      truncated: stat.size > TEXT_PREVIEW_MAX_BYTES
    };
  } finally {
    await handle.close();
  }
});
ipcMain.handle("hermes:selectPaths", async (_event, options = {}) => {
  const properties = options?.directories ? ["openDirectory"] : ["openFile"];
  if (options?.multiple !== false) properties.push("multiSelections");
  let resolvedDefaultPath;
  if (options?.defaultPath) {
    try {
      resolvedDefaultPath = path.resolve(String(options.defaultPath));
    } catch {
      resolvedDefaultPath = void 0;
    }
  }
  const result = await dialog.showOpenDialog(mainWindow, {
    title: options?.title || "Add context",
    defaultPath: resolvedDefaultPath,
    properties,
    filters: Array.isArray(options?.filters) ? options.filters : void 0
  });
  if (result.canceled) return [];
  return result.filePaths;
});
ipcMain.handle("hermes:writeClipboard", (_event, text) => {
  clipboard.writeText(String(text || ""));
  return true;
});
ipcMain.handle("hermes:saveImageFromUrl", (_event, url) => saveImageFromUrl(String(url || "")));
ipcMain.handle("hermes:saveImageBuffer", async (_event, payload) => {
  const data = payload?.data;
  if (!data) throw new Error("saveImageBuffer: missing data");
  const buffer = Buffer.isBuffer(data) ? data : Buffer.from(data);
  return writeComposerImage(buffer, payload?.ext || ".png");
});
ipcMain.handle("hermes:saveClipboardImage", async () => {
  const image = clipboard.readImage();
  if (!image || image.isEmpty()) {
    return "";
  }
  return writeComposerImage(image.toPNG(), ".png");
});
ipcMain.handle(
  "hermes:normalizePreviewTarget",
  (_event, target, baseDir) => normalizePreviewTarget(String(target || ""), baseDir ? String(baseDir) : "")
);
ipcMain.handle("hermes:watchPreviewFile", (_event, url) => watchPreviewFile(String(url || "")));
ipcMain.handle("hermes:stopPreviewFileWatch", (_event, id) => stopPreviewFileWatch(String(id || "")));
ipcMain.on("hermes:titlebar-theme", (_event, payload) => {
  if (!payload || !isHexColor(payload.background) || !isHexColor(payload.foreground)) {
    return;
  }
  rendererTitleBarTheme = {
    background: payload.background,
    foreground: payload.foreground
  };
  mainWindow?.setTitleBarOverlay?.(getTitleBarOverlayOptions());
});
ipcMain.on("hermes:native-theme", (_event, mode) => {
  if (!THEME_SOURCES.has(mode)) {
    return;
  }
  if (nativeTheme.themeSource !== mode) {
    nativeTheme.themeSource = mode;
    writePersistedThemeSource(mode);
  }
});
ipcMain.on("hermes:translucency", (_event, payload) => {
  const next = clampIntensity(payload && payload.intensity);
  if (next === translucencyIntensity) {
    return;
  }
  translucencyIntensity = next;
  writePersistedTranslucency(next);
  for (const win of BrowserWindow.getAllWindows()) {
    applyWindowTranslucency(win);
  }
});
ipcMain.handle("hermes:openExternal", (_event, url) => {
  if (!openExternalUrl(url)) {
    throw new Error("Invalid external URL");
  }
});
ipcMain.handle("hermes:openPreviewInBrowser", async (_event, url) => {
  if (!await openPreviewInBrowser(url)) {
    throw new Error("Invalid preview URL");
  }
});
ipcMain.handle("hermes:setting:defaultProjectDir:get", async () => ({
  dir: readDefaultProjectDir(),
  defaultLabel: app.getPath("home"),
  resolvedCwd: resolveHermesCwd()
}));
ipcMain.handle("hermes:workspace:sanitize", async (_event, cwd) => sanitizeWorkspaceCwd(cwd));
ipcMain.handle("hermes:setting:defaultProjectDir:set", async (_event, dir) => {
  const next = typeof dir === "string" && dir.trim() ? dir.trim() : null;
  if (next) {
    try {
      fs.mkdirSync(next, { recursive: true });
    } catch (error) {
      throw new Error(`Could not create directory: ${error.message}`);
    }
  }
  writeDefaultProjectDir(next);
  return { dir: next };
});
ipcMain.handle("hermes:setting:defaultProjectDir:pick", async () => {
  const result = await dialog.showOpenDialog({
    title: "Choose default project directory",
    properties: ["openDirectory", "createDirectory"],
    defaultPath: readDefaultProjectDir() || app.getPath("home")
  });
  if (result.canceled || result.filePaths.length === 0) {
    return { canceled: true, dir: null };
  }
  return { canceled: false, dir: result.filePaths[0] };
});
ipcMain.handle("hermes:fetchLinkTitle", (_event, url) => fetchLinkTitle(url));
ipcMain.handle("hermes:logs:reveal", async () => {
  try {
    await fs.promises.mkdir(path.dirname(DESKTOP_LOG_PATH), { recursive: true });
    if (!fileExists(DESKTOP_LOG_PATH)) {
      await fs.promises.appendFile(DESKTOP_LOG_PATH, "");
    }
    shell.showItemInFolder(DESKTOP_LOG_PATH);
    return { ok: true, path: DESKTOP_LOG_PATH };
  } catch (error) {
    return { ok: false, path: DESKTOP_LOG_PATH, error: error.message };
  }
});
ipcMain.handle("hermes:logs:recent", async () => ({ path: DESKTOP_LOG_PATH, lines: hermesLog.slice(-200) }));
function isExecutableFile(filePath) {
  if (!filePath || !path.isAbsolute(filePath)) {
    return false;
  }
  try {
    fs.accessSync(filePath, fs.constants.X_OK);
    return true;
  } catch {
    return false;
  }
}
function posixShellSpec(shellPath) {
  const shellName = path.basename(shellPath);
  const interactiveArgs = shellName.includes("zsh") || shellName.includes("bash") ? ["-il"] : ["-i"];
  return { args: interactiveArgs, command: shellPath, name: shellName };
}
var spawnHelperChecked = false;
function ensureSpawnHelperExecutable() {
  if (spawnHelperChecked || IS_WINDOWS || !nodePtyDir) {
    return;
  }
  spawnHelperChecked = true;
  const arch = process.arch;
  const candidates = [
    path.join(nodePtyDir, "build", "Release", "spawn-helper"),
    path.join(nodePtyDir, "prebuilds", `${process.platform}-${arch}`, "spawn-helper")
  ];
  for (const helper of candidates) {
    try {
      const mode = fs.statSync(helper).mode;
      if ((mode & 73) !== 73) {
        fs.chmodSync(helper, mode | 493);
      }
    } catch {
    }
  }
}
function windowsPowerShellPath() {
  const systemRoot = process.env.SystemRoot || process.env.windir || "C:\\Windows";
  const builtin = path.join(systemRoot, "System32", "WindowsPowerShell", "v1.0", "powershell.exe");
  return isExecutableFile(builtin) ? builtin : findOnPath("powershell.exe");
}
function shellSpecFor(shellPath) {
  const name = path.basename(shellPath).toLowerCase();
  if (name.startsWith("pwsh") || name.startsWith("powershell")) {
    return { args: ["-NoLogo"], command: shellPath, name };
  }
  if (name.startsWith("cmd")) {
    return { args: [], command: shellPath, name };
  }
  return posixShellSpec(shellPath);
}
function windowsShellSpec() {
  const command = findOnPath("pwsh.exe") || findOnPath("pwsh") || windowsPowerShellPath() || process.env.COMSPEC || "cmd.exe";
  return shellSpecFor(command);
}
function terminalShellCommand() {
  const override = (process.env.HERMES_DESKTOP_SHELL || (IS_WINDOWS ? "" : process.env.SHELL) || "").trim();
  if (override) {
    const resolved = isExecutableFile(override) ? override : findOnPath(override);
    if (resolved) {
      return shellSpecFor(resolved);
    }
  }
  if (IS_WINDOWS) {
    return windowsShellSpec();
  }
  const shellPath = ["/bin/zsh", "/bin/bash", "/bin/sh"].find((candidate) => isExecutableFile(candidate));
  return posixShellSpec(shellPath || "/bin/sh");
}
function safeTerminalCwd(cwd) {
  const candidate = path.resolve(String(cwd || app.getPath("home")));
  try {
    const stat = fs.statSync(candidate);
    return stat.isDirectory() ? candidate : path.dirname(candidate);
  } catch {
    return app.getPath("home");
  }
}
function terminalShellEnv() {
  const env2 = { ...process.env };
  for (const key of Object.keys(env2)) {
    if (key === "npm_config_prefix" || key.startsWith("npm_config_") || key.startsWith("npm_package_")) {
      delete env2[key];
    }
  }
  delete env2.NO_COLOR;
  delete env2.FORCE_COLOR;
  delete env2.COLORFGBG;
  env2.COLORTERM = "truecolor";
  env2.LC_CTYPE = env2.LC_CTYPE || "UTF-8";
  env2.TERM = "xterm-256color";
  env2.TERM_PROGRAM = "Hermes";
  env2.TERM_PROGRAM_VERSION = app.getVersion();
  env2.HERMES_DESKTOP_TERMINAL = "1";
  return env2;
}
function terminalChannel(id, suffix) {
  return `hermes:terminal:${id}:${suffix}`;
}
function disposeTerminalSession(id) {
  const sessionInfo = terminalSessions.get(id);
  if (!sessionInfo) {
    return false;
  }
  terminalSessions.delete(id);
  try {
    sessionInfo.pty.kill();
  } catch {
  }
  return true;
}
ipcMain.handle("hermes:fs:readDir", async (_event, dirPath) => readDirForIpc(dirPath));
ipcMain.handle("hermes:fs:gitRoot", async (_event, startPath) => gitRootForIpc(startPath));
ipcMain.handle("hermes:fs:reveal", async (_event, targetPath) => {
  const target = String(targetPath || "").trim();
  if (!target) {
    return false;
  }
  try {
    shell.showItemInFolder(target);
    return true;
  } catch {
    return false;
  }
});
ipcMain.handle("hermes:fs:rename", async (_event, targetPath, newName) => {
  const src = String(targetPath || "").trim();
  const name = String(newName || "").trim();
  if (!src || !name || name === "." || name === ".." || name.includes("/") || name.includes("\\")) {
    throw new Error("Invalid rename");
  }
  const dst = path.join(path.dirname(src), name);
  if (dst === src) {
    return { path: dst };
  }
  if (fs.existsSync(dst)) {
    throw new Error(`"${name}" already exists`);
  }
  await fs.promises.rename(src, dst);
  return { path: dst };
});
ipcMain.handle("hermes:fs:writeText", async (_event, filePath, content) => {
  const raw = String(filePath || "").trim();
  if (!raw) {
    throw new Error("Invalid path");
  }
  const text = String(content ?? "");
  if (text.length > 1e6) {
    throw new Error("Content too large");
  }
  const resolved = resolveRequestedPathForIpc(expandUserPath(raw), { purpose: "Write text file" });
  if (!directoryExists(path.dirname(resolved))) {
    throw new Error("Parent directory does not exist");
  }
  await fs.promises.writeFile(resolved, text, "utf8");
  return { path: resolved };
});
ipcMain.handle("hermes:fs:trash", async (_event, targetPath) => {
  const target = String(targetPath || "").trim();
  if (!target) {
    throw new Error("Invalid delete");
  }
  await shell.trashItem(target);
  return true;
});
ipcMain.handle(
  "hermes:git:worktreeList",
  async (_event, repoPath) => listWorktrees(repoPath, resolveGitBinary())
);
ipcMain.handle(
  "hermes:git:worktreeAdd",
  async (_event, repoPath, options) => addWorktree(repoPath, options || {}, resolveGitBinary())
);
ipcMain.handle(
  "hermes:git:worktreeRemove",
  async (_event, repoPath, worktreePath, options) => removeWorktree(repoPath, worktreePath, options || {}, resolveGitBinary())
);
ipcMain.handle(
  "hermes:git:branchSwitch",
  async (_event, repoPath, branch) => switchBranch(repoPath, branch, resolveGitBinary())
);
ipcMain.handle(
  "hermes:git:branchList",
  async (_event, repoPath) => listBranches(repoPath, resolveGitBinary())
);
ipcMain.handle("hermes:git:repoStatus", async (_event, repoPath) => repoStatus(repoPath, resolveGitBinary()));
ipcMain.handle(
  "hermes:git:review:list",
  async (_event, repoPath, scope, baseRef) => reviewList(repoPath, scope, baseRef, resolveGitBinary())
);
ipcMain.handle(
  "hermes:git:review:diff",
  async (_event, repoPath, filePath, scope, baseRef, staged) => reviewDiff(repoPath, filePath, scope, baseRef, staged, resolveGitBinary())
);
ipcMain.handle(
  "hermes:git:fileDiff",
  async (_event, repoPath, filePath) => fileDiffVsHead(repoPath, filePath, resolveGitBinary())
);
ipcMain.handle(
  "hermes:git:review:stage",
  async (_event, repoPath, filePath) => reviewStage(repoPath, filePath ?? null, resolveGitBinary())
);
ipcMain.handle(
  "hermes:git:review:unstage",
  async (_event, repoPath, filePath) => reviewUnstage(repoPath, filePath ?? null, resolveGitBinary())
);
ipcMain.handle(
  "hermes:git:review:revert",
  async (_event, repoPath, filePath) => reviewRevert(repoPath, filePath ?? null, resolveGitBinary())
);
ipcMain.handle(
  "hermes:git:review:revParse",
  async (_event, repoPath, ref) => reviewRevParse(repoPath, ref, resolveGitBinary())
);
ipcMain.handle(
  "hermes:git:review:commit",
  async (_event, repoPath, message, push) => reviewCommit(repoPath, message, Boolean(push), resolveGitBinary())
);
ipcMain.handle(
  "hermes:git:review:commitContext",
  async (_event, repoPath) => reviewCommitContext(repoPath, resolveGitBinary())
);
ipcMain.handle("hermes:git:review:push", async (_event, repoPath) => reviewPush(repoPath, resolveGitBinary()));
ipcMain.handle("hermes:git:review:shipInfo", async (_event, repoPath) => reviewShipInfo(repoPath, resolveGhBinary()));
ipcMain.handle(
  "hermes:git:review:createPr",
  async (_event, repoPath) => reviewCreatePr(repoPath, resolveGitBinary(), resolveGhBinary())
);
ipcMain.handle("hermes:git:scanRepos", async (_event, roots, options) => {
  try {
    return await scanGitRepos(roots || [], options || {});
  } catch {
    return [];
  }
});
ipcMain.handle("hermes:terminal:start", async (event, payload = {}) => {
  if (!nodePty) {
    throw new Error("PTY support is unavailable. Reinstall desktop dependencies and restart Hermes.");
  }
  ensureSpawnHelperExecutable();
  const id = crypto.randomUUID();
  const { args, command, name } = terminalShellCommand();
  const cwd = safeTerminalCwd(payload?.cwd);
  const cols = Math.max(2, Number.parseInt(String(payload?.cols || 80), 10) || 80);
  const rows = Math.max(2, Number.parseInt(String(payload?.rows || 24), 10) || 24);
  const ptyProcess = nodePty.spawn(command, args, {
    cols,
    cwd,
    env: terminalShellEnv(),
    name: "xterm-256color",
    rows
  });
  terminalSessions.set(id, { pty: ptyProcess, webContentsId: event.sender.id });
  const send = (suffix, payload2) => {
    if (event.sender.isDestroyed()) {
      return;
    }
    event.sender.send(terminalChannel(id, suffix), payload2);
  };
  ptyProcess.onData((data) => send("data", data));
  ptyProcess.onExit(({ exitCode, signal }) => {
    terminalSessions.delete(id);
    send("exit", { code: exitCode, signal: signal || null });
  });
  event.sender.once("destroyed", () => disposeTerminalSession(id));
  return { cwd, id, shell: name };
});
ipcMain.handle("hermes:terminal:write", (_event, id, data) => {
  const sessionInfo = terminalSessions.get(String(id || ""));
  if (!sessionInfo) {
    return false;
  }
  sessionInfo.pty.write(String(data || ""));
  return true;
});
ipcMain.handle("hermes:terminal:resize", (_event, id, size = {}) => {
  const sessionInfo = terminalSessions.get(String(id || ""));
  if (!sessionInfo) {
    return false;
  }
  const cols = Math.max(2, Number.parseInt(String(size?.cols || 80), 10) || 80);
  const rows = Math.max(2, Number.parseInt(String(size?.rows || 24), 10) || 24);
  sessionInfo.pty.resize(cols, rows);
  return true;
});
ipcMain.handle("hermes:terminal:dispose", (_event, id) => disposeTerminalSession(String(id || "")));
ipcMain.handle(
  "hermes:updates:check",
  async () => checkUpdates().catch((error) => ({
    supported: true,
    branch: readDesktopUpdateConfig().branch,
    error: "check-failed",
    message: error?.message || String(error),
    fetchedAt: Date.now()
  }))
);
ipcMain.handle(
  "hermes:updates:apply",
  async (_event, payload) => applyUpdates(payload || {}).catch((error) => ({
    ok: false,
    error: "apply-failed",
    message: error?.message || String(error)
  }))
);
ipcMain.handle("hermes:updates:branch:get", async () => readDesktopUpdateConfig());
ipcMain.handle("hermes:updates:branch:set", async (_event, name) => {
  const branch = typeof name === "string" && name.trim() ? name.trim() : DEFAULT_UPDATE_BRANCH;
  writeDesktopUpdateConfig({ branch });
  return { branch };
});
function resolveHermesVersion() {
  try {
    const root = resolveUpdateRoot();
    const initPath = path.join(root, "hermes_cli", "__init__.py");
    if (fileExists(initPath)) {
      const raw = fs.readFileSync(initPath, "utf8");
      const match = raw.match(/__version__\s*=\s*["']([^"']+)["']/);
      if (match) {
        return match[1];
      }
    }
  } catch {
  }
  return app.getVersion();
}
function showAboutPanelFresh() {
  app.setAboutPanelOptions({
    applicationName: APP_NAME,
    applicationVersion: resolveHermesVersion(),
    copyright: "Copyright \xA9 2026 Nous Research"
  });
  app.showAboutPanel();
}
ipcMain.handle("hermes:version", async () => ({
  appVersion: resolveHermesVersion(),
  electronVersion: process.versions.electron,
  nodeVersion: process.versions.node,
  platform: process.platform,
  hermesRoot: resolveUpdateRoot()
}));
function uninstallVenvPython() {
  return getVenvPython(VENV_ROOT);
}
async function getUninstallSummary() {
  const py = uninstallVenvPython();
  const agentRoot = ACTIVE_HERMES_ROOT;
  const fallback = () => ({
    hermes_home: HERMES_HOME,
    agent_installed: isHermesSourceRoot(agentRoot) && fileExists(py),
    gui_installed: true,
    source_built_artifacts: [],
    packaged_app_paths: [],
    userdata_dir: app.getPath("userData"),
    userdata_exists: true,
    platform: process.platform,
    probe: "fallback"
  });
  if (!fileExists(py)) {
    return fallback();
  }
  return new Promise((resolve) => {
    let stdout = "";
    let settled = false;
    const done = (value) => {
      if (settled) return;
      settled = true;
      resolve(value);
    };
    try {
      const child = spawn(
        py,
        ["-m", "hermes_cli.main", "uninstall", "--gui-summary"],
        hiddenWindowsChildOptions({
          cwd: agentRoot,
          env: { ...process.env, HERMES_HOME, NO_COLOR: "1" },
          stdio: ["ignore", "pipe", "ignore"]
        })
      );
      child.stdout.on("data", (chunk) => {
        stdout += chunk.toString();
      });
      child.on("error", () => done(fallback()));
      child.on("exit", (code) => {
        if (code !== 0) return done(fallback());
        try {
          const line = stdout.trim().split("\n").filter(Boolean).pop() || "{}";
          const parsed = JSON.parse(line);
          parsed.running_app_path = resolveRemovableAppPath(process.execPath, process.platform, process.env);
          done(parsed);
        } catch {
          done(fallback());
        }
      });
      setTimeout(() => done(fallback()), 8e3);
    } catch {
      done(fallback());
    }
  });
}
async function runDesktopUninstall(mode) {
  let uninstallArgs;
  try {
    uninstallArgs = uninstallArgsForMode(mode);
  } catch (error) {
    return { ok: false, error: "invalid-mode", message: error.message };
  }
  const venvPy = uninstallVenvPython();
  if (!fileExists(venvPy)) {
    return {
      ok: false,
      error: "agent-missing",
      message: `Can't run the uninstaller: no Hermes agent venv at ${VENV_ROOT}.`
    };
  }
  let py = venvPy;
  let pythonPath = null;
  if (modeRemovesAgent(mode)) {
    const sysPy = findSystemPython();
    if (sysPy) {
      py = sysPy;
      pythonPath = ACTIVE_HERMES_ROOT;
    } else if (IS_WINDOWS) {
      rememberLog(
        "[uninstall] no system Python found for lite/full on Windows; falling back to the venv python \u2014 venv files locked by the running interpreter may remain and need manual deletion."
      );
    }
  }
  const appPath = resolveRemovableAppPath(process.execPath, process.platform, process.env);
  const removeBundle = shouldRemoveAppBundle(IS_PACKAGED, appPath) ? appPath : null;
  try {
    await releaseBackendLock(ACTIVE_HERMES_ROOT, "uninstall");
  } catch (error) {
    rememberLog(`[uninstall] backend teardown errored (continuing): ${error.message}`);
  }
  const scriptArgs = {
    desktopPid: process.pid,
    pythonExe: py,
    pythonPath,
    agentRoot: ACTIVE_HERMES_ROOT,
    uninstallArgs,
    appPath: removeBundle,
    hermesHome: HERMES_HOME
  };
  let scriptPath;
  let runner;
  let runnerArgs;
  try {
    if (IS_WINDOWS) {
      scriptPath = path.join(app.getPath("temp"), `hermes-uninstall-${Date.now()}.cmd`);
      fs.writeFileSync(scriptPath, buildWindowsCleanupScript(scriptArgs));
      runner = process.env.ComSpec || "cmd.exe";
      runnerArgs = ["/c", scriptPath];
    } else {
      scriptPath = path.join(app.getPath("temp"), `hermes-uninstall-${Date.now()}.sh`);
      fs.writeFileSync(scriptPath, buildPosixCleanupScript(scriptArgs), { mode: 493 });
      runner = "/bin/bash";
      runnerArgs = [scriptPath];
    }
  } catch (error) {
    return { ok: false, error: "script-write-failed", message: error.message };
  }
  try {
    const child = spawn(runner, runnerArgs, {
      detached: true,
      stdio: "ignore",
      windowsHide: true
    });
    child.unref();
  } catch (error) {
    return { ok: false, error: "spawn-failed", message: error.message };
  }
  rememberLog(
    `[uninstall] launched detached cleanup (${mode}): ${scriptPath} (removesAgent=${modeRemovesAgent(mode)} removesUserData=${modeRemovesUserData(mode)} bundle=${removeBundle || "none"})`
  );
  setTimeout(() => app.quit(), 800);
  return { ok: true, mode, willRemoveAppBundle: Boolean(removeBundle), scriptPath };
}
ipcMain.handle("hermes:uninstall:summary", async () => getUninstallSummary());
ipcMain.handle("hermes:uninstall:run", async (_event, payload) => {
  const mode = payload && typeof payload === "object" ? payload.mode : payload;
  return runDesktopUninstall(String(mode || ""));
});
ipcMain.handle("hermes:vscode-theme:fetch", async (_event, id) => fetchMarketplaceThemes(String(id || "")));
ipcMain.handle("hermes:vscode-theme:search", async (_event, query) => searchMarketplaceThemes(String(query || ""), 20));
var HERMES_PROTOCOL = "hermes";
var _pendingDeepLink = null;
var _rendererReadyForDeepLink = false;
function _extractDeepLink(argv) {
  if (!Array.isArray(argv)) return null;
  return argv.find((a) => typeof a === "string" && a.startsWith(`${HERMES_PROTOCOL}://`)) || null;
}
function handleDeepLink(url) {
  if (!url || typeof url !== "string") return;
  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    rememberLog(`[deeplink] ignoring malformed url: ${url}`);
    return;
  }
  const kind = parsed.hostname || "";
  const name = decodeURIComponent((parsed.pathname || "").replace(/^\//, ""));
  const params = {};
  parsed.searchParams.forEach((v, k) => {
    params[k] = v;
  });
  const payload = { kind, name, params };
  if (!_rendererReadyForDeepLink || !mainWindow || mainWindow.isDestroyed()) {
    _pendingDeepLink = payload;
    return;
  }
  try {
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.focus();
    mainWindow.webContents.send("hermes:deep-link", payload);
    rememberLog(`[deeplink] delivered ${kind}/${name}`);
  } catch (err) {
    rememberLog(`[deeplink] delivery failed: ${err.message}`);
  }
}
ipcMain.handle("hermes:deep-link-ready", () => {
  _rendererReadyForDeepLink = true;
  if (_pendingDeepLink) {
    const queued = _pendingDeepLink;
    _pendingDeepLink = null;
    handleDeepLink(
      `${HERMES_PROTOCOL}://${queued.kind}/${encodeURIComponent(queued.name)}` + (Object.keys(queued.params).length ? "?" + new URLSearchParams(queued.params).toString() : "")
    );
  }
  return { ok: true };
});
function registerDeepLinkProtocol() {
  try {
    if (process.defaultApp && process.argv.length >= 2) {
      app.setAsDefaultProtocolClient(HERMES_PROTOCOL, process.execPath, [path.resolve(process.argv[1])]);
    } else {
      app.setAsDefaultProtocolClient(HERMES_PROTOCOL);
    }
  } catch (err) {
    rememberLog(`[deeplink] protocol registration failed: ${err.message}`);
  }
}
var _gotSingleInstanceLock = app.requestSingleInstanceLock();
if (!_gotSingleInstanceLock) {
  app.quit();
} else {
  app.on("second-instance", (_event, argv) => {
    const url = _extractDeepLink(argv);
    if (url) handleDeepLink(url);
    else if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });
}
app.on("open-url", (event, url) => {
  event.preventDefault();
  handleDeepLink(url);
});
app.whenReady().then(() => {
  if (IS_MAC) {
    Menu.setApplicationMenu(buildApplicationMenu());
  } else {
    Menu.setApplicationMenu(null);
  }
  installMediaPermissions();
  registerMediaProtocol();
  registerDeepLinkProtocol();
  ensureWslWindowsFonts();
  configureSpellChecker();
  registerPowerResumeListeners();
  createWindow();
  const _coldStartLink = _extractDeepLink(process.argv);
  if (_coldStartLink) handleDeepLink(_coldStartLink);
  app.on("activate", () => {
    if (!mainWindow || mainWindow.isDestroyed()) {
      createWindow();
    } else {
      focusWindow(mainWindow);
    }
  });
});
function configureSpellChecker() {
  try {
    const defaultSession = session.defaultSession;
    if (!defaultSession || typeof defaultSession.setSpellCheckerLanguages !== "function") {
      return;
    }
    const available = defaultSession.availableSpellCheckerLanguages || [];
    const locale = app.getLocale && app.getLocale() || "en-US";
    const candidates = [locale, locale.split("-")[0], "en-US", "en"];
    const chosen = candidates.find((lang) => available.includes(lang)) || "en-US";
    defaultSession.setSpellCheckerLanguages([chosen]);
  } catch (error) {
    rememberLog(`Spellchecker setup failed: ${error.message}`);
  }
}
app.on("before-quit", () => {
  closePetOverlay();
  if (bootstrapAbortController) {
    try {
      bootstrapAbortController.abort();
    } catch {
    }
  }
  if (desktopLogFlushTimer) {
    clearTimeout(desktopLogFlushTimer);
    desktopLogFlushTimer = null;
  }
  flushDesktopLogBufferSync();
  closePreviewWatchers();
  for (const id of [...terminalSessions.keys()]) {
    disposeTerminalSession(id);
  }
  if (hermesProcess && !hermesProcess.killed) {
    hermesProcess.kill("SIGTERM");
  }
  stopAllPoolBackends();
});
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
