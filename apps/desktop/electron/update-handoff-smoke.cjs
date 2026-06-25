const fs = require('node:fs')
const net = require('node:net')
const os = require('node:os')
const path = require('node:path')
const { spawn, spawnSync } = require('node:child_process')

const DESKTOP_ROOT = path.resolve(__dirname, '..')
const REPO_ROOT = path.resolve(DESKTOP_ROOT, '..', '..')
const DEFAULT_APP_EXE = path.join(DESKTOP_ROOT, 'release', 'win-unpacked', 'Hermes.exe')
const POWERSHELL_EXE = path.join(
  process.env.WINDIR || 'C:\\Windows',
  'System32',
  'WindowsPowerShell',
  'v1.0',
  'powershell.exe'
)

function fail(message) {
  throw new Error(message)
}

function skip(message) {
  console.log(`Skipping desktop update handoff smoke: ${message}`)
}

function quotePowerShell(value) {
  return `'${String(value).replace(/'/g, "''")}'`
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    encoding: 'utf8',
    stdio: options.stdio || 'pipe',
    windowsHide: true,
    ...options
  })
  if (result.status !== 0) {
    fail(
      [
        `Command failed: ${command} ${args.join(' ')}`,
        result.stdout ? `stdout:\n${result.stdout}` : '',
        result.stderr ? `stderr:\n${result.stderr}` : ''
      ]
        .filter(Boolean)
        .join('\n')
    )
  }
  return result
}

function mkdirp(dir) {
  fs.mkdirSync(dir, { recursive: true })
}

function writeFile(file, contents) {
  mkdirp(path.dirname(file))
  fs.writeFileSync(file, contents)
}

function createFakeInstalledHermesRoot(updateRoot) {
  writeFile(path.join(updateRoot, 'hermes_cli', '__init__.py'), '')
  writeFile(
    path.join(updateRoot, 'hermes_cli', 'main.py'),
    [
      'import argparse',
      'import json',
      'from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer',
      '',
      'class Handler(BaseHTTPRequestHandler):',
      '    def do_GET(self):',
      '        if self.path.startswith("/api/status"):',
      '            body = json.dumps({"ok": True, "smoke": True}).encode("utf-8")',
      '            self.send_response(200)',
      '            self.send_header("Content-Type", "application/json")',
      '            self.send_header("Content-Length", str(len(body)))',
      '            self.end_headers()',
      '            self.wfile.write(body)',
      '            return',
      '        self.send_response(404)',
      '        self.end_headers()',
      '',
      '    def log_message(self, *_args):',
      '        return',
      '',
      'if __name__ == "__main__":',
      '    parser = argparse.ArgumentParser()',
      '    parser.add_argument("command", nargs="?")',
      '    parser.add_argument("--no-open", action="store_true")',
      '    parser.add_argument("--host", default="127.0.0.1")',
      '    parser.add_argument("--port", type=int, required=True)',
      '    args = parser.parse_args()',
      '    if args.command != "dashboard":',
      '        raise SystemExit(f"unsupported smoke command: {args.command}")',
      '    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()',
      ''
    ].join('\n')
  )
  writeFile(
    path.join(updateRoot, '.hermes-bootstrap-complete'),
    JSON.stringify(
      {
        schemaVersion: 1,
        pinnedCommit: '0000000',
        pinnedBranch: 'main',
        completedAt: new Date().toISOString(),
        desktopVersion: 'smoke'
      },
      null,
      2
    ) + '\n'
  )
}

function resolveSmokePython() {
  const candidates = [
    process.env.HERMES_DESKTOP_SMOKE_PYTHON,
    path.join(REPO_ROOT, 'venv', 'Scripts', 'python.exe')
  ].filter(Boolean)
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) return path.resolve(candidate)
  }

  const result = spawnSync('py', ['-3.11', '-c', 'import sys; print(sys.executable)'], {
    encoding: 'utf8',
    stdio: 'pipe',
    windowsHide: true
  })
  if (result.status === 0 && result.stdout.trim() && fs.existsSync(result.stdout.trim())) {
    return result.stdout.trim()
  }

  return null
}

function isLocked(file) {
  let fd
  try {
    fd = fs.openSync(file, 'r+')
    return false
  } catch {
    return fs.existsSync(file)
  } finally {
    if (fd !== undefined) fs.closeSync(fd)
  }
}

async function sleep(ms) {
  await new Promise(resolve => setTimeout(resolve, ms))
}

async function waitFor(predicate, { label, timeoutMs = 30_000, intervalMs = 100 } = {}) {
  const deadline = Date.now() + timeoutMs
  let lastError = null
  while (Date.now() < deadline) {
    try {
      const result = await predicate()
      if (result) return result
    } catch (error) {
      lastError = error
    }
    await sleep(intervalMs)
  }
  if (lastError) {
    fail(`Timed out waiting for ${label}: ${lastError.message}`)
  }
  fail(`Timed out waiting for ${label}`)
}

function waitForExit(child, timeoutMs, label) {
  if (child.exitCode !== null || child.signalCode !== null) {
    return Promise.resolve({ code: child.exitCode, signal: child.signalCode })
  }
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`${label} did not exit within ${timeoutMs}ms`))
    }, timeoutMs)
    child.once('exit', (code, signal) => {
      clearTimeout(timer)
      resolve({ code, signal })
    })
  })
}

function taskkill(pid) {
  if (!Number.isInteger(pid) || pid <= 0) return
  spawnSync('taskkill', ['/PID', String(pid), '/T', '/F'], {
    stdio: 'ignore',
    windowsHide: true
  })
}

function isProcessRunning(pid) {
  if (!Number.isInteger(pid) || pid <= 0) return false
  const result = spawnSync(
    'powershell.exe',
    [
      '-NoProfile',
      '-ExecutionPolicy',
      'Bypass',
      '-Command',
      `$p = Get-CimInstance Win32_Process -Filter "ProcessId=${pid}"; if ($p) { '1' }`
    ],
    {
      encoding: 'utf8',
      stdio: 'pipe',
      timeout: 5000,
      windowsHide: true
    }
  )
  return result.status === 0 && result.stdout.trim() === '1'
}

function smokeAppPath() {
  const entries = [
    path.join(process.env.WINDIR || 'C:\\Windows', 'System32'),
    process.env.WINDIR || 'C:\\Windows',
    path.join(process.env.WINDIR || 'C:\\Windows', 'System32', 'Wbem'),
    path.join(process.env.WINDIR || 'C:\\Windows', 'System32', 'WindowsPowerShell', 'v1.0')
  ]
  return entries.join(path.delimiter)
}

function dotnetTargetFramework() {
  const result = spawnSync('dotnet', ['--list-sdks'], {
    encoding: 'utf8',
    stdio: 'pipe',
    windowsHide: true
  })
  if (result.error) {
    if (result.error.code === 'ENOENT') return null
    fail(`Could not run dotnet --list-sdks: ${result.error.message}`)
  }
  if (result.status !== 0) return null

  const majors = result.stdout
    .split(/\r?\n/)
    .map(line => Number.parseInt(line.split('.')[0], 10))
    .filter(Number.isInteger)
  const major = Math.max(...majors)
  if (!Number.isFinite(major) || major <= 0) return null
  return `net${major}.0`
}

function compileFakeUpdater(homeDir, sourceDir, targetFramework) {
  const projectDir = path.join(sourceDir, 'FakeHermesUpdater')
  const publishDir = path.join(projectDir, 'publish')
  mkdirp(projectDir)

  writeFile(
    path.join(projectDir, 'FakeHermesUpdater.csproj'),
    `<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>${targetFramework}</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>
</Project>
`
  )

  writeFile(
    path.join(projectDir, 'Program.cs'),
    `using System.Diagnostics;
using System.Text.Json;

var log = Environment.GetEnvironmentVariable("HERMES_FAKE_UPDATER_LOG");
if (string.IsNullOrWhiteSpace(log))
{
    log = Path.Combine(Environment.GetEnvironmentVariable("HERMES_HOME") ?? AppContext.BaseDirectory, "fake-updater.json");
}
Directory.CreateDirectory(Path.GetDirectoryName(log)!);

int? relaunchPid = null;
var relaunch = Environment.GetEnvironmentVariable("HERMES_FAKE_UPDATER_RELAUNCH");
if (!string.IsNullOrWhiteSpace(relaunch) && File.Exists(relaunch))
{
    var psi = new ProcessStartInfo(relaunch)
    {
        UseShellExecute = false,
        WorkingDirectory = Path.GetDirectoryName(relaunch)!
    };
    psi.Environment["HERMES_FAKE_RELAUNCHED"] = "1";
    foreach (var key in new[] {
        "HERMES_HOME",
        "HERMES_DESKTOP_USER_DATA_DIR",
        "HERMES_DESKTOP_HERMES_ROOT",
        "HERMES_DESKTOP_BOOT_FAKE",
        "HERMES_DESKTOP_BOOT_FAKE_STEP_MS",
        "HERMES_DESKTOP_DISABLE_GPU"
    })
    {
        var value = Environment.GetEnvironmentVariable(key);
        if (value is not null) psi.Environment[key] = value;
    }
    relaunchPid = Process.Start(psi)?.Id;
}

var payload = new
{
    pid = Environment.ProcessId,
    args,
    cwd = Environment.CurrentDirectory,
    hermesHome = Environment.GetEnvironmentVariable("HERMES_HOME"),
    path = Environment.GetEnvironmentVariable("PATH"),
    relaunchPid,
    startedAtUtc = DateTimeOffset.UtcNow
};
File.WriteAllText(log, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
`
  )

  run(
    'dotnet',
    [
      'publish',
      projectDir,
      '-c',
      'Release',
      '-r',
      'win-x64',
      '--self-contained',
      'false',
      '-p:PublishSingleFile=true',
      '-p:DebugType=None',
      '-p:DebugSymbols=false',
      '-o',
      publishDir
    ],
    {
      env: {
        ...process.env,
        DOTNET_CLI_TELEMETRY_OPTOUT: '1',
        DOTNET_NOLOGO: '1',
        DOTNET_SKIP_FIRST_TIME_EXPERIENCE: '1'
      }
    }
  )

  const builtExe = path.join(publishDir, 'FakeHermesUpdater.exe')
  const updater = path.join(homeDir, 'hermes-setup.exe')
  fs.copyFileSync(builtExe, updater)
  return updater
}

function createLockedRuntime(updateRoot, shimPath) {
  const lockHolder = path.join(updateRoot, 'hermes_cli', 'lock-holder.js')
  mkdirp(path.dirname(shimPath))
  writeFile(shimPath, 'fake shim locked by update smoke\n')

  const lockScript = [
    `$stream = [System.IO.File]::Open(${quotePowerShell(shimPath)}, [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)`,
    'try { Start-Sleep -Seconds 120 } finally { $stream.Dispose() }'
  ].join('; ')
  writeFile(
    lockHolder,
    `const { spawn } = require('node:child_process')

const child = spawn(${JSON.stringify(POWERSHELL_EXE)}, [
  '-NoProfile',
  '-ExecutionPolicy',
  'Bypass',
  '-Command',
  ${JSON.stringify(lockScript)}
], {
  stdio: 'ignore',
  windowsHide: true
})

child.once('exit', (code) => process.exit(code ?? 0))
process.once('exit', () => {
  try {
    child.kill()
  } catch {
    // Best effort; the update smoke kills the whole process tree.
  }
})
setInterval(() => {}, 1000)
`
  )

  return spawn(
    process.execPath,
    [lockHolder, 'hermes_cli.main'],
    {
      cwd: updateRoot,
      stdio: 'ignore',
      windowsHide: true
    }
  )
}

function getFreePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer()
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const address = server.address()
      const port = typeof address === 'object' && address ? address.port : null
      server.close(() => {
        if (!port) reject(new Error('Could not allocate a port'))
        else resolve(port)
      })
    })
  })
}

async function fetchJson(url) {
  const response = await fetch(url)
  if (!response.ok) fail(`${url} returned HTTP ${response.status}`)
  return await response.json()
}

async function waitForRendererTarget(port) {
  return await waitFor(
    async () => {
      const targets = await fetchJson(`http://127.0.0.1:${port}/json/list`)
      return targets.find(target => target.type === 'page' && target.webSocketDebuggerUrl)
    },
    { label: 'Electron renderer debug target', timeoutMs: 45_000, intervalMs: 250 }
  )
}

function cdpCall(ws, id, method, params) {
  return new Promise((resolve, reject) => {
    const handleMessage = event => {
      const message = JSON.parse(String(event.data))
      if (message.id !== id) return
      ws.removeEventListener('message', handleMessage)
      if (message.error) reject(new Error(`${method} failed: ${JSON.stringify(message.error)}`))
      else resolve(message)
    }
    ws.addEventListener('message', handleMessage)
    ws.send(JSON.stringify({ id, method, params }))
  })
}

async function evaluateInRenderer(webSocketDebuggerUrl, expression) {
  if (typeof WebSocket !== 'function') fail('This smoke requires a Node runtime with global WebSocket support')

  const ws = new WebSocket(webSocketDebuggerUrl)
  await new Promise((resolve, reject) => {
    ws.addEventListener('open', resolve, { once: true })
    ws.addEventListener('error', reject, { once: true })
  })

  try {
    await cdpCall(ws, 1, 'Runtime.enable')
    const response = await cdpCall(ws, 2, 'Runtime.evaluate', {
      awaitPromise: true,
      expression,
      returnByValue: true,
      timeout: 20_000
    })
    if (response.result.exceptionDetails) {
      fail(JSON.stringify(response.result.exceptionDetails, null, 2))
    }
    return response.result.result.value
  } finally {
    ws.close()
  }
}

function isExpectedHandoffRendererShutdown(error) {
  return /Execution context was destroyed/i.test(String(error?.message || error))
}

async function main() {
  if (process.platform !== 'win32') {
    skip('Windows-only smoke is not available on this platform.')
    return
  }
  if (typeof WebSocket !== 'function') {
    skip('Node runtime does not provide global WebSocket support.')
    return
  }

  const appExe = path.resolve(process.env.HERMES_DESKTOP_SMOKE_APP || DEFAULT_APP_EXE)
  if (!fs.existsSync(appExe)) {
    fail(`Missing built desktop app at ${appExe}. Run "python -m hermes_cli.main desktop --build-only" first.`)
  }
  if (!fs.existsSync(POWERSHELL_EXE)) fail(`Missing PowerShell at ${POWERSHELL_EXE}`)

  const targetFramework = dotnetTargetFramework()
  if (!targetFramework) {
    skip('dotnet SDK is not available to build the fake updater.')
    return
  }
  const smokePython = resolveSmokePython()
  if (!smokePython) {
    skip('Python 3.11 is not available for the fake dashboard backend.')
    return
  }

  const stamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\..+/, '')
  const smokeRoot = path.resolve(
    process.env.HERMES_DESKTOP_UPDATE_SMOKE_ROOT ||
      path.join(os.tmpdir(), `hermes-desktop-update-handoff-${stamp}`)
  )
  const homeDir = path.join(smokeRoot, 'home')
  const userDataDir = path.join(smokeRoot, 'user-data')
  const updateRoot = path.join(smokeRoot, 'hermes-agent')
  const shimPath = path.join(updateRoot, 'venv', 'Scripts', 'hermes.exe')
  const updaterLog = path.join(smokeRoot, 'fake-updater.json')
  const desktopLog = path.join(homeDir, 'logs', 'desktop.log')
  let relaunchedPid = null

  mkdirp(homeDir)
  mkdirp(userDataDir)
  mkdirp(path.dirname(shimPath))
  createFakeInstalledHermesRoot(updateRoot)
  compileFakeUpdater(homeDir, smokeRoot, targetFramework)

  const runtime = createLockedRuntime(updateRoot, shimPath)
  await waitFor(() => isLocked(shimPath), {
    label: 'fake Hermes venv shim lock',
    timeoutMs: 10_000,
    intervalMs: 100
  })

  const port = await getFreePort()
  const app = spawn(appExe, [`--remote-debugging-port=${port}`], {
    cwd: path.dirname(appExe),
    env: {
      ...process.env,
      HERMES_DESKTOP_BOOT_FAKE: '1',
      HERMES_DESKTOP_BOOT_FAKE_STEP_MS: '120',
      HERMES_DESKTOP_DISABLE_GPU: '1',
      HERMES_DESKTOP_HERMES_ROOT: updateRoot,
      HERMES_DESKTOP_PYTHON: smokePython,
      HERMES_DESKTOP_USER_DATA_DIR: userDataDir,
      HERMES_FAKE_UPDATER_LOG: updaterLog,
      HERMES_FAKE_UPDATER_RELAUNCH: appExe,
      HERMES_HOME: homeDir,
      PATH: smokeAppPath()
    },
    stdio: 'ignore',
    windowsHide: false
  })

  try {
    const target = await waitForRendererTarget(port)
    let result = null
    let rendererClosedDuringHandoff = false
    try {
      result = await evaluateInRenderer(
        target.webSocketDebuggerUrl,
        `(async () => {
          const start = Date.now()
          while (!window.hermesDesktop?.updates?.apply) {
            if (Date.now() - start > 15000) throw new Error('updates bridge was not exposed')
            await new Promise(resolve => setTimeout(resolve, 100))
          }
          window.__hermesUpdateProgress = []
          window.hermesDesktop.updates.onProgress?.(payload => window.__hermesUpdateProgress.push(payload))
          return await window.hermesDesktop.updates.apply({})
        })()`
      )
    } catch (error) {
      if (!isExpectedHandoffRendererShutdown(error)) throw error
      rendererClosedDuringHandoff = true
    }

    if (!rendererClosedDuringHandoff && (!result || result.ok !== true || result.handedOff !== true || result.manual)) {
      fail(`Expected handed-off update result, got ${JSON.stringify(result)}`)
    }

    await waitFor(() => fs.existsSync(updaterLog), {
      label: 'fake updater log',
      timeoutMs: 20_000,
      intervalMs: 100
    })
    await waitForExit(app, 15_000, 'Hermes desktop')
    await waitForExit(runtime, 10_000, 'locked fake runtime')

    if (isLocked(shimPath)) fail('The fake venv shim is still locked after handoff')

    const fakeUpdater = JSON.parse(fs.readFileSync(updaterLog, 'utf8'))
    const args = Array.isArray(fakeUpdater.args) ? fakeUpdater.args : []
    if (!args.includes('--update')) fail(`Fake updater did not receive --update: ${JSON.stringify(args)}`)
    if (!args.includes('--branch')) fail(`Fake updater did not receive --branch: ${JSON.stringify(args)}`)
    if (!args.includes('main')) fail(`Fake updater did not receive main branch: ${JSON.stringify(args)}`)
    relaunchedPid = Number(fakeUpdater.relaunchPid)
    if (!Number.isInteger(relaunchedPid) || relaunchedPid <= 0) {
      fail(`Fake updater did not relaunch Hermes: ${JSON.stringify(fakeUpdater)}`)
    }
    await waitFor(() => isProcessRunning(relaunchedPid), {
      label: 'relaunched Hermes desktop process',
      timeoutMs: 10_000,
      intervalMs: 250
    })

    const log = fs.existsSync(desktopLog) ? fs.readFileSync(desktopLog, 'utf8') : ''
    for (const needle of [
      'stopped 1 Hermes runtime process(es) before update',
      'venv shim unlocked; safe to proceed',
      'launched updater:'
    ]) {
      if (!log.includes(needle)) fail(`Desktop log is missing "${needle}"`)
    }

    console.log('Desktop update handoff smoke passed.')
    console.log(`  smoke root: ${smokeRoot}`)
    console.log(`  updater log: ${updaterLog}`)
    console.log(`  desktop log: ${desktopLog}`)
    taskkill(relaunchedPid)
  } catch (error) {
    try {
      if (!relaunchedPid && fs.existsSync(updaterLog)) {
        relaunchedPid = Number(JSON.parse(fs.readFileSync(updaterLog, 'utf8')).relaunchPid)
      }
    } catch {
      relaunchedPid = null
    }
    taskkill(relaunchedPid)
    taskkill(app.pid)
    taskkill(runtime.pid)
    console.error(`Desktop update handoff smoke failed. Smoke root preserved at ${smokeRoot}`)
    throw error
  }
}

main().catch(error => {
  console.error(error.stack || error.message || String(error))
  process.exit(1)
})
