#!/usr/bin/env node
// Real desktop + real backend. Isolated identity; no copied credentials.
// Adds a fixed Vite port and CDP (9344) so the run can be driven and read
// from outside. When you only want to click through it yourself:
//   D=$(mktemp -d) && env -u HERMES_SHARED_AUTH_DIR HERMES_GUEST_ONBOARDING=1 \
//     HERMES_HOME="$D/home" HERMES_DESKTOP_USER_DATA_DIR="$D/userdata" hermes desktop
import { spawn } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import net from 'node:net'

const desktop = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const root = path.resolve(desktop, '../..')
const sandbox = process.argv[2] ? path.resolve(process.argv[2]) : fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-connectors-'))
const port = 5194
const url = `http://127.0.0.1:${port}`
const python = process.env.HERMES_DESKTOP_PYTHON
if (!python || !fs.existsSync(python)) throw new Error('Set HERMES_DESKTOP_PYTHON to your installed Hermes Python executable.')
if (!fs.existsSync(path.join(desktop, 'dist/electron-main.mjs'))) throw new Error('Run node scripts/bundle-electron-main.mjs --dev first.')
await new Promise((resolve, reject) => {
  const probe = net.createServer()
  probe.once('error', () => reject(new Error(`Port ${port} is occupied; refusing to use another worktree server.`)))
  probe.listen(port, '127.0.0.1', () => probe.close(resolve))
})
for (const dir of ['.hermes/shared', 'electron', 'work']) fs.mkdirSync(path.join(sandbox, dir), { recursive: true })
const env = Object.fromEntries(['PATH', 'TMPDIR', 'LANG', 'SHELL'].filter(key => process.env[key]).map(key => [key, process.env[key]]))
Object.assign(env, {
  HOME: sandbox, HERMES_HOME: path.join(sandbox, '.hermes'), HERMES_SHARED_AUTH_DIR: path.join(sandbox, '.hermes/shared'),
  HERMES_DESKTOP_USER_DATA_DIR: path.join(sandbox, 'electron'), HERMES_DESKTOP_CWD: path.join(sandbox, 'work'),
  HERMES_DESKTOP_HERMES_ROOT: root, HERMES_DESKTOP_PYTHON: python,
  HERMES_DESKTOP_IGNORE_EXISTING: '1', HERMES_DESKTOP_APP_NAME: 'Hermes Connector Rehearsal',
  HERMES_GUEST_ONBOARDING: '1', HERMES_DESKTOP_DEV_SERVER: url, HERMES_DESKTOP_CDP_PORT: '9344'
})
const entry = path.join(desktop, 'dist/connector-rehearsal.mjs')
fs.writeFileSync(entry, `import { app } from 'electron'\napp.setAsDefaultProtocolClient = () => false\napp.setAppPath(${JSON.stringify(desktop)})\nawait import('./electron-main.mjs')\n`)
const vite = spawn(path.join(root, 'node_modules/.bin/vite'), ['--host', '127.0.0.1', '--port', String(port), '--strictPort'], { cwd: desktop, env, stdio: 'inherit' })
let electron
let stopping = false
const stop = () => {
  if (stopping) return
  stopping = true
  electron?.kill('SIGTERM')
  vite.kill('SIGTERM')
}
process.on('SIGINT', stop)
process.on('SIGTERM', stop)
vite.on('exit', stop)
try {
  const deadline = Date.now() + 30000
  while (true) {
    if (vite.exitCode !== null) throw new Error('Renderer server exited')
    try { if ((await fetch(url)).ok) break } catch {}
    if (Date.now() > deadline) throw new Error('Renderer did not become ready')
    await new Promise(resolve => setTimeout(resolve, 250))
  }
  const binary = path.join(desktop, 'node_modules/electron/dist', process.platform === 'darwin' ? 'Electron.app/Contents/MacOS/Electron' : process.platform === 'win32' ? 'electron.exe' : 'electron')
  electron = spawn(binary, [entry], { cwd: desktop, env, stdio: 'inherit' })
  console.log(`Connector rehearsal: ${sandbox}\nRenderer: ${url}\nReal service; OAuth requires your approval. Reuse the sandbox path to reopen this identity.`)
  electron.on('exit', stop)
} catch (error) { stop(); throw error }
