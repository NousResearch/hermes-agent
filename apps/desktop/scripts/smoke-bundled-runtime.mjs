import fs from 'node:fs'
import { randomUUID } from 'node:crypto'
import path from 'node:path'
import { spawn } from 'node:child_process'
const root = path.resolve(process.argv[2] || 'build/runtime')
const home = fs.mkdtempSync(path.join(process.env.TEMP || '.', 'czesiek-runtime-smoke-'))
const token = randomUUID()
const guard = path.join(home, 'offline-guard')
fs.mkdirSync(guard)
fs.writeFileSync(
  path.join(guard, 'sitecustomize.py'),
  `import sys\ndef audit(event,args):\n if event == 'socket.connect':\n  address=args[1]\n  if isinstance(address,tuple) and address[0] not in ('127.0.0.1','::1','localhost'):\n   raise RuntimeError('Offline smoke test blocked external connection')\nsys.addaudithook(audit)\n`
)
const env = Object.fromEntries(
  Object.entries(process.env).filter(([key]) => !/KEY|TOKEN|SECRET|HERMES|PYTHON|VIRTUAL_ENV|^PATH$/i.test(key))
)
Object.assign(env, {
  HERMES_HOME: home,
  HERMES_DASHBOARD_SESSION_TOKEN: token,
  HERMES_SERVE_HEADLESS: '1',
  HERMES_DESKTOP: '1',
  PYTHONPATH: guard + ';' + root + '/agent',
  PYTHONNOUSERSITE: '1',
  PYTHONDONTWRITEBYTECODE: '1',
  PYTHONUTF8: '1',
  PATH: root + '/python;C:/Windows/System32;C:/Windows',
  TEMP: home,
  TMP: home
})
const child = spawn(
  root + '/python/python.exe',
  ['-m', 'hermes_cli.main', 'serve', '--host', '127.0.0.1', '--port', '0'],
  { env, cwd: home, windowsHide: true }
)
let output = ''
let checking = false
const timer = setTimeout(() => {
  console.error(output.slice(-6000))
  child.kill()
  process.exitCode = 1
}, 120000)
const onData = async data => {
  output += data.toString()
  fs.writeFileSync(path.join(home, 'smoke.log'), output)
  const match = output.match(/HERMES_BACKEND_READY port=(\d+)/)
  if (!match || checking) return
  checking = true
  try {
    const response = await fetch('http://127.0.0.1:' + match[1] + '/api/health')
    const voice = await fetch('http://127.0.0.1:' + match[1] + '/api/voice/realtime/status', { headers: { 'X-Hermes-Session-Token': token } })
    const session = await fetch('http://127.0.0.1:' + match[1] + '/api/voice/realtime/session', { method: 'POST', headers: { 'X-Hermes-Session-Token': token } })
    if (voice.status !== 200 || session.status !== 400) throw new Error(`Live routes: status=${voice.status}, session=${session.status}`)
    console.log(
      JSON.stringify({
        status: response.status,
        liveStatus: voice.status,
        liveSessionWithoutKey: session.status,
        profile: home,
        python: root + '/python/python.exe',
        externalPythonNetwork: 'blocked',
        systemPythonOnPath: false
      })
    )
    if (!response.ok) process.exitCode = 1
  } catch (error) {
    console.error(error)
    process.exitCode = 1
  } finally {
    clearTimeout(timer)
    child.kill()
  }
}
child.stdout.on('data', onData)
child.stderr.on('data', onData)
child.on('exit', code => {
  clearTimeout(timer)
  if (!checking) {
    console.error(output.slice(-6000))
    process.exitCode = code || 1
  }
})
