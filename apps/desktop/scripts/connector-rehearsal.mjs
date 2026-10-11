#!/usr/bin/env node
// Launch the source desktop against the real backend with an isolated identity.
// The rehearsal never copies credentials and never registers a protocol handler.
//
// Usage: node apps/desktop/scripts/connector-rehearsal.mjs [sandbox-directory] [--cdp-port <port>]
// Requires HERMES_DESKTOP_PYTHON and a dev Electron bundle from
// `node apps/desktop/scripts/bundle-electron-main.mjs --dev`.

import { randomUUID } from 'node:crypto'
import { execFileSync, spawn } from 'node:child_process'
import { createRequire } from 'node:module'
import fs from 'node:fs'
import net from 'node:net'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const require = createRequire(import.meta.url)
const PORTS = Object.freeze({ renderer: 5194 })
const REHEARSAL_MARKER = '.connector-rehearsal.json'

function normalizeCdpPort(value = 'off') {
  if (typeof value === 'string' && value.trim().toLowerCase() === 'off') return 'off'
  const port = Number(value)
  if (!Number.isInteger(port) || port < 1024 || port > 65535) {
    throw new Error('CDP port must be "off" or an integer from 1024 to 65535.')
  }
  return port
}

/** Accept one optional sandbox path and an explicit, opt-in CDP port. */
export function parseRehearsalArgs(args) {
  let sandbox
  let cdpPort = 'off'
  let sawCdpPort = false

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index]
    if (arg === '--cdp-port') {
      if (sawCdpPort || args[index + 1] === undefined) {
        throw new Error('Use --cdp-port <port> once, with a port from 1024 to 65535.')
      }
      sawCdpPort = true
      cdpPort = normalizeCdpPort(args[++index])
    } else if (arg.startsWith('--')) {
      throw new Error(`Unknown option: ${arg}`)
    } else if (sandbox !== undefined) {
      throw new Error('Provide at most one sandbox-directory path.')
    } else {
      sandbox = arg
    }
  }

  return { sandbox, cdpPort }
}

export function rehearsalPortChecks({ rendererPort, cdpPort }) {
  return [
    { label: 'Renderer', port: rendererPort },
    ...(cdpPort === 'off' ? [] : [{ label: 'CDP', port: cdpPort }])
  ]
}

/** Keep Windows notifications separate and leave registered consent handlers alone. */
export function configureRehearsalApp(app, desktop, platform = process.platform) {
  app.setAsDefaultProtocolClient = () => false
  app.setAppPath(desktop)
  if (platform === 'win32') {
    const setAppUserModelId = app.setAppUserModelId.bind(app)
    app.setAppUserModelId = () => setAppUserModelId('com.nousresearch.hermes.connector-rehearsal')
  }
}

/** Never adopt a nonempty user profile or arbitrary directory as a rehearsal. */
export function assertRehearsalDirectory(sandbox) {
  if (!fs.existsSync(sandbox) || fs.readdirSync(sandbox).length === 0) return
  try {
    if (JSON.parse(fs.readFileSync(path.join(sandbox, REHEARSAL_MARKER), 'utf8')).schema === 1) return
  } catch {}
  throw new Error('Choose an empty directory or an existing connector rehearsal directory.')
}

const WINDOWS_ENV_KEYS = new Set([
  'PATH',
  'PATHEXT',
  'SYSTEMROOT',
  'WINDIR',
  'COMSPEC',
  'OS',
  'SYSTEMDRIVE',
  'PROGRAMDATA',
  'PROCESSOR_ARCHITECTURE',
  'PROCESSOR_IDENTIFIER',
  'PROCESSOR_LEVEL',
  'PROCESSOR_REVISION',
  'NUMBER_OF_PROCESSORS'
])
const POSIX_ENV_KEYS = new Set([
  'PATH',
  'LANG',
  'LANGUAGE',
  'LC_ALL',
  'LC_CTYPE',
  'SHELL',
  'DISPLAY',
  'WAYLAND_DISPLAY',
  'XDG_RUNTIME_DIR',
  'DBUS_SESSION_BUS_ADDRESS',
  'XAUTHORITY'
])

function copyAllowedEnvironment(sourceEnv, platform) {
  const allowed = platform === 'win32' ? WINDOWS_ENV_KEYS : POSIX_ENV_KEYS
  const env = {}

  for (const [key, value] of Object.entries(sourceEnv)) {
    const name = platform === 'win32' ? key.toUpperCase() : key
    if (value !== undefined && allowed.has(name)) {
      env[name] = value
    }
  }

  return env
}

/** Prepare isolated paths and a credential-free child environment without side effects. */
export function prepareRehearsal({
  sandbox,
  desktop,
  root,
  python,
  sourceEnv = process.env,
  platform = process.platform,
  rendererPort = PORTS.renderer,
  cdpPort = 'off'
}) {
  const paths = platform === 'win32' ? path.win32 : path
  const isolatedSandbox = paths.resolve(sandbox)
  const isolatedDesktop = paths.resolve(desktop)
  const hermesHome = paths.join(isolatedSandbox, '.hermes')
  const runtimeDir = paths.join(isolatedSandbox, 'runtime')
  const userDataDir = paths.join(isolatedSandbox, 'electron')
  const workingDirectory = paths.join(isolatedSandbox, 'work')
  const tempDir = paths.join(isolatedSandbox, 'tmp')
  const resolvedCdpPort = normalizeCdpPort(cdpPort)
  const appDataDir = platform === 'win32' ? paths.join(isolatedSandbox, 'AppData', 'Roaming') : undefined
  const localAppDataDir = platform === 'win32' ? paths.join(isolatedSandbox, 'AppData', 'Local') : undefined
  const url = `http://127.0.0.1:${rendererPort}`
  const env = copyAllowedEnvironment(sourceEnv, platform)

  Object.assign(env, {
    HOME: isolatedSandbox,
    TMPDIR: tempDir,
    HERMES_HOME: hermesHome,
    HERMES_SHARED_AUTH_DIR: paths.join(hermesHome, 'shared'),
    HERMES_RUNTIME_DIR: runtimeDir,
    HERMES_DESKTOP_USER_DATA_DIR: userDataDir,
    HERMES_DESKTOP_CWD: workingDirectory,
    HERMES_DESKTOP_HERMES_ROOT: paths.resolve(root),
    HERMES_DESKTOP_PYTHON: paths.resolve(python),
    HERMES_DESKTOP_IGNORE_EXISTING: '1',
    HERMES_DESKTOP_APP_NAME: 'Hermes Connector Rehearsal',
    HERMES_GUEST_ONBOARDING: '1',
    HERMES_DESKTOP_DEV_SERVER: url,
    HERMES_DESKTOP_CDP_PORT: String(resolvedCdpPort)
  })

  if (platform === 'win32') {
    Object.assign(env, {
      USERPROFILE: isolatedSandbox,
      TEMP: tempDir,
      TMP: tempDir,
      APPDATA: appDataDir,
      LOCALAPPDATA: localAppDataDir
    })

    const drive = paths.parse(isolatedSandbox).root.slice(0, 2)
    if (/^[A-Za-z]:$/.test(drive)) {
      env.HOMEDRIVE = drive
      env.HOMEPATH = isolatedSandbox.slice(drive.length)
    }
  }

  return {
    sandbox: isolatedSandbox,
    desktop: isolatedDesktop,
    root: paths.resolve(root),
    hermesHome,
    sharedAuthDir: paths.join(hermesHome, 'shared'),
    runtimeDir,
    userDataDir,
    workingDirectory,
    tempDir,
    appDataDir,
    localAppDataDir,
    rendererPort,
    cdpPort: resolvedCdpPort,
    url,
    env
  }
}

/** Refuse to start if any fixed rehearsal port is already bound on localhost. */
export async function assertPortsAvailable(ports, host = '127.0.0.1') {
  for (const { label, port } of ports) {
    const probe = net.createServer()

    try {
      await new Promise((resolve, reject) => {
        probe.once('error', reject)
        probe.listen(port, host, () => {
          probe.removeAllListeners('error')
          probe.close(error => (error ? reject(error) : resolve()))
        })
      })
    } catch (error) {
      throw new Error(`${label} port ${port} is unavailable; free it before starting the connector rehearsal.`, {
        cause: error
      })
    }
  }
}

function waitForSpawn(child, label, onRuntimeError) {
  return new Promise((resolve, reject) => {
    let spawned = false
    child.once('spawn', () => {
      spawned = true
      resolve()
    })
    child.on('error', error => {
      const failure = new Error(`${label} ${spawned ? 'process error' : 'failed to start'}.`, { cause: error })
      if (spawned) onRuntimeError(failure)
      else reject(failure)
    })
  })
}

function resolveViteEntry() {
  const packagePath = require.resolve('vite/package.json')
  const pkg = JSON.parse(fs.readFileSync(packagePath, 'utf8'))
  const bin = typeof pkg.bin === 'string' ? pkg.bin : pkg.bin?.vite

  if (!bin) {
    throw new Error('Could not resolve the Vite CLI from vite/package.json.')
  }

  return path.resolve(path.dirname(packagePath), bin)
}

async function launch() {
  const desktop = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
  const root = path.resolve(desktop, '../..')
  const args = parseRehearsalArgs(process.argv.slice(2))
  const sandbox = args.sandbox
    ? path.resolve(args.sandbox)
    : path.join(os.tmpdir(), `hermes-connectors-${randomUUID()}`)
  const python = process.env.HERMES_DESKTOP_PYTHON

  if (!python || !fs.existsSync(python)) {
    throw new Error('Set HERMES_DESKTOP_PYTHON to your installed Hermes Python executable.')
  }
  if (!fs.existsSync(path.join(desktop, 'dist/electron-main.mjs'))) {
    throw new Error('Run node apps/desktop/scripts/bundle-electron-main.mjs --dev first.')
  }

  const rehearsal = prepareRehearsal({ sandbox, desktop, root, python, cdpPort: args.cdpPort })
  assertRehearsalDirectory(rehearsal.sandbox)
  await assertPortsAvailable(rehearsalPortChecks(rehearsal))

  fs.mkdirSync(rehearsal.sandbox, { recursive: true })
  fs.writeFileSync(path.join(rehearsal.sandbox, REHEARSAL_MARKER), JSON.stringify({ schema: 1 }) + '\n')

  for (const dir of [
    rehearsal.hermesHome,
    rehearsal.sharedAuthDir,
    rehearsal.runtimeDir,
    rehearsal.userDataDir,
    rehearsal.workingDirectory,
    rehearsal.tempDir,
    rehearsal.appDataDir,
    rehearsal.localAppDataDir
  ]) {
    if (dir) fs.mkdirSync(dir, { recursive: true })
  }

  // Use the normal atomic config writer, preserving a reopened rehearsal's other settings.
  execFileSync(python, ['-m', 'hermes_cli.main', 'config', 'set', 'tools.connectors.browser_return', 'false'], {
    cwd: root,
    env: rehearsal.env,
    stdio: 'inherit'
  })

  const entry = path.join(desktop, 'dist/connector-rehearsal.mjs')
  fs.writeFileSync(
    entry,
    `import { app } from 'electron'\nimport { configureRehearsalApp } from ${JSON.stringify(pathToFileURL(fileURLToPath(import.meta.url)).href)}\nconfigureRehearsalApp(app, ${JSON.stringify(desktop)})\nawait import('./electron-main.mjs')\n`
  )

  let vite
  let electron
  let stopping = false
  const stop = () => {
    if (stopping) return
    stopping = true
    electron?.kill('SIGTERM')
    vite?.kill('SIGTERM')
  }

  process.on('SIGINT', stop)
  process.on('SIGTERM', stop)
  try {
    vite = spawn(
      process.execPath,
      [resolveViteEntry(), '--host', '127.0.0.1', '--port', String(rehearsal.rendererPort), '--strictPort'],
      {
        cwd: desktop,
        env: rehearsal.env,
        stdio: 'inherit'
      }
    )
    vite.on('exit', stop)
    await waitForSpawn(vite, 'Renderer server', error => {
      console.error(error)
      stop()
    })

    const deadline = Date.now() + 30000
    while (true) {
      if (vite.exitCode !== null) throw new Error('Renderer server exited.')
      try {
        if ((await fetch(rehearsal.url)).ok) break
      } catch {}
      if (Date.now() > deadline) throw new Error('Renderer did not become ready.')
      await new Promise(resolve => setTimeout(resolve, 250))
    }

    // The electron package resolves its supported platform-specific executable.
    electron = spawn(require('electron'), [entry], { cwd: desktop, env: rehearsal.env, stdio: 'inherit' })
    electron.on('exit', stop)
    await waitForSpawn(electron, 'Electron', error => {
      console.error(error)
      stop()
    })
    console.log(
      `Connector rehearsal: ${sandbox}\nRenderer: ${rehearsal.url}\nReal service; OAuth requires your approval. Reuse the sandbox path to reopen this identity.`
    )
  } catch (error) {
    stop()
    throw error
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await launch()
}
