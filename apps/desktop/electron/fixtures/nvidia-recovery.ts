import { appendFileSync, existsSync, readFileSync } from 'node:fs'

// No windows, gateway, credentials, or real Hermes home. This fixture exercises
// native Electron locking with the production Linux recovery policy and test driver facts.
import { app } from 'electron'

import { INSTALL_STAMP } from '../install-stamp'
import { launchBuildIdentity } from '../launch-build-identity'
import { createLaunchMarkerWriter } from '../launch-marker-writer'
import { readNvidiaEglMarker } from '../linux-nvidia-egl-fallback'
import { createNvidiaEglRecovery } from '../linux-nvidia-egl-recovery'

const argument = process.argv.find(arg => arg.startsWith('--recovery-fixture='))

if (!argument) {
  throw new Error('Missing isolated fixture options')
}

const options: { userData: string; eventsFile: string; commandFile: string; quitBeforeReveal?: boolean } = JSON.parse(
  argument.slice('--recovery-fixture='.length)
)

app.setPath('userData', options.userData)
app.disableHardwareAcceleration()
const writer = createLaunchMarkerWriter()

const report = (event: string, data: Record<string, unknown> = {}) => {
  appendFileSync(options.eventsFile, `${JSON.stringify({ event, ...data })}\n`, 'utf8')
}

const recovery = createNvidiaEglRecovery({
  userData: options.userData,
  appVersion: app.getVersion(),
  buildIdentity: launchBuildIdentity({ appVersion: app.getVersion(), installStamp: INSTALL_STAMP }),
  platform: 'linux',
  driverMajor: 600,
  driverVersion: '600.1.2',
  env: {},
  queueMarker: write => writer.queue(write),
  relaunch: () => report('relaunch'),
  log: message => report('log', { message })
})

const primary = app.requestSingleInstanceLock()
writer.flush(primary)

if (!primary) {
  report('secondary', { marker: readNvidiaEglMarker(options.userData) })
  app.exit(0)
} else {
  app.on('before-quit', () => recovery.cleanExit())
  void app.whenReady().then(() => {
    if (options.quitBeforeReveal) {
      recovery.cleanExit()
      report('early-exit', { marker: readNvidiaEglMarker(options.userData) })
      app.exit(0)

      return
    }

    recovery.healthy()
    report('ready', { software: recovery.decision.enable, marker: readNvidiaEglMarker(options.userData) })

    const control = setInterval(() => {
      if (existsSync(options.commandFile) && readFileSync(options.commandFile, 'utf8') === 'quit') {
        clearInterval(control)
        app.quit()
      }
    }, 50)
  })
}
