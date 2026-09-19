import { app } from 'electron'

import { WSLG_X11_FALLBACK_EXIT_CODE, wslgLaunchArgs, wslgX11FallbackArgs } from './wslg-launch'
import { spawnWslgLaunch } from './wslg-launch-process'

const args = wslgLaunchArgs(process.argv.slice(1), process.env, process.platform)

if (args) {
  // Keep the launcher alive until the child exits: npm's concurrently must not
  // tear down Vite during this handoff. No backend, windows or single-instance
  // lock are created in this parent. The child has an explicit platform flag,
  // so it goes straight into main on its first pass.
  let supervisedChild = spawnWslgLaunch(args)
  let fallbackUsed = false

  supervisedChild.once('error', error => {
    console.error('[hermes] WSLg launch failed:', error)
    app.exit(1)
  })
  supervisedChild.once('exit', code => {
    // Keep this supervisor alive across the one bounded fallback. In dev,
    // concurrently treats an Electron parent exit as a reason to stop Vite.
    const fallbackArgs = !fallbackUsed && code === WSLG_X11_FALLBACK_EXIT_CODE ? wslgX11FallbackArgs(args) : null

    if (!fallbackArgs) {
      app.exit(code ?? 1)

      return
    }

    fallbackUsed = true
    supervisedChild = spawnWslgLaunch(fallbackArgs)
    supervisedChild.once('error', error => {
      console.error('[hermes] WSLg X11 fallback launch failed:', error)
      app.exit(1)
    })
    supervisedChild.once('exit', fallbackCode => app.exit(fallbackCode ?? 1))
  })

  for (const signal of ['SIGINT', 'SIGTERM'] as const) {
    process.once(signal, () => supervisedChild.kill(signal))
  }
} else {
  await import('./main')
}
