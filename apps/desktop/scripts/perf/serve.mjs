// Launch a standalone, fully isolated perf instance and leave it running so you
// can attach the harness (`npm run perf`) or DevTools to it. Ctrl-C tears it
// down and removes its temp dirs.
//
//   npm run perf:serve                 # :9222, temp HERMES_HOME + user-data-dir
//   PERF_PORT=9333 npm run perf:serve  # custom CDP port
//
// This is the isolation seam: because it uses its own --user-data-dir the
// Electron single-instance lock never collides with a running `hgui`.

import { startIsolatedInstance } from './lib/launch.mjs'

const port = Number(process.env.PERF_PORT ?? 9222)
const devPort = Number(process.env.PERF_DEV_PORT ?? 5174)

console.log(`[perf:serve] starting isolated instance (CDP :${port}, dev :${devPort})…`)

const instance = await startIsolatedInstance({
  port,
  devPort,
  hermesHome: process.env.PERF_HERMES_HOME,
  userDataDir: process.env.PERF_USER_DATA
})

console.log(`[perf:serve] READY — attach with: npm run perf -- --port ${port}`)

let closing = false
const shutdown = () => {
  if (closing) {
    return
  }

  closing = true
  console.log('\n[perf:serve] tearing down…')
  instance.teardown()
  process.exit(0)
}

process.on('SIGINT', shutdown)
process.on('SIGTERM', shutdown)
