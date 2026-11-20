// Real backend + production READY parser; no Electron renderer is exercised.
import { spawn } from 'node:child_process'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { randomUUID } from 'node:crypto'
import { setTimeout as delay } from 'node:timers/promises'
import { waitForDashboardPortAnnouncement } from '../../apps/desktop/electron/backend-ready.ts'

const repo = fileURLToPath(new URL('../../', import.meta.url))
const home = mkdtempSync(join(tmpdir(), 'hermes-native-ready-'))
const token = randomUUID()
const env = {
  PATH: process.env.PATH, HOME: home, HERMES_HOME: home,
  LANG: 'C.UTF-8', PYTHONUNBUFFERED: '1', HERMES_NONINTERACTIVE: '1',
  HERMES_DASHBOARD_SESSION_TOKEN: token, HERMES_SERVE_HEADLESS: '1',
  HERMES_PARENT_PID: String(process.pid),
}
const child = spawn(process.argv[2] || join(repo, '.venv/bin/python'),
  ['-m', 'hermes_cli.main', 'serve', '--host', '127.0.0.1', '--port', '0', '--isolated'],
  { cwd: repo, env, stdio: ['ignore', 'pipe', 'pipe'] })
let tail = ''
for (const stream of [child.stdout, child.stderr]) {
  stream.on('data', chunk => { tail = (tail + chunk.toString()).slice(-32768) })
}
// Match production's synchronous attach-then-wait order, without synthetic output.
const announcement = waitForDashboardPortAnnouncement(child, { bufferedOutput: () => tail })
try {
  const port = await announcement
  console.log(JSON.stringify({ platform: process.platform, arch: process.arch, readyPort: port, home }))
  const ws = new WebSocket(`ws://127.0.0.1:${port}/api/ws?token=${token}`)
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('WS open timeout')), 15000)
    ws.addEventListener('open', () => { clearTimeout(timer); resolve() }, { once: true })
    ws.addEventListener('error', () => { clearTimeout(timer); reject(new Error('WS rejected')) }, { once: true })
  })
  for (let i = 0; i < 20; i++) {
    for (const path of ['/api/status', '/api/profiles']) {
      const response = await fetch(`http://127.0.0.1:${port}${path}`, { headers: { 'X-Hermes-Session-Token': token } })
      if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`)
      await response.json()
    }
    if (child.exitCode !== null || ws.readyState !== WebSocket.OPEN) throw new Error('Backend/session lost')
    await delay(5000)
  }
  ws.close()
  console.log('PASS: production READY parser + real backend HTTP/profiles and WS stayed healthy beyond 90s; not full Electron UI proof')
} catch (error) {
  console.error(tail)
  throw error
} finally {
  child.kill('SIGTERM')
}
