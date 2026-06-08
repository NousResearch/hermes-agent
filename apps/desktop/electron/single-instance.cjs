const fs = require('node:fs')
const path = require('node:path')
const { spawn } = require('node:child_process')

function focusPrimaryWindow(window) {
  if (!window || (typeof window.isDestroyed === 'function' && window.isDestroyed())) {
    return false
  }

  if (typeof window.isMinimized === 'function' && window.isMinimized()) {
    window.restore()
  }

  if (typeof window.isVisible === 'function' && !window.isVisible()) {
    window.show()
  }

  if (typeof window.focus === 'function') {
    window.focus()
  }

  return true
}

function resolveMacAppBundle(executablePath) {
  const parts = String(executablePath || '').split(path.sep)
  const appIndex = parts.findIndex(part => part.endsWith('.app'))

  if (appIndex === -1) {
    return null
  }

  return parts.slice(0, appIndex + 1).join(path.sep) || path.sep
}

function focusRunningAppBundle({ executablePath = process.execPath, platform = process.platform, spawnProcess = spawn } = {}) {
  if (platform !== 'darwin') {
    return false
  }

  const appBundle = resolveMacAppBundle(executablePath)
  if (!appBundle) {
    return false
  }

  const child = spawnProcess('open', [appBundle], {
    detached: true,
    stdio: 'ignore'
  })
  child.unref?.()
  return true
}

function isProcessAlive(pid, processApi = process) {
  if (!Number.isInteger(pid) || pid <= 0) {
    return false
  }

  try {
    processApi.kill(pid, 0)
    return true
  } catch (error) {
    return error?.code === 'EPERM'
  }
}

function readLockPid(lockDir, fsApi = fs) {
  try {
    const raw = fsApi.readFileSync(path.join(lockDir, 'owner.json'), 'utf8')
    const parsed = JSON.parse(raw)
    return Number.isInteger(parsed?.pid) ? parsed.pid : null
  } catch {
    return null
  }
}

function writeLockOwner(lockDir, processApi = process, fsApi = fs) {
  fsApi.writeFileSync(
    path.join(lockDir, 'owner.json'),
    JSON.stringify(
      {
        executable: processApi.execPath,
        pid: processApi.pid,
        startedAt: new Date().toISOString()
      },
      null,
      2
    )
  )
}

function acquireDirectoryLock(lockDir, { fsApi = fs, processApi = process } = {}) {
  const tryAcquire = () => {
    fsApi.mkdirSync(lockDir, { mode: 0o700 })
    writeLockOwner(lockDir, processApi, fsApi)
    return true
  }

  try {
    tryAcquire()
  } catch (error) {
    if (error?.code !== 'EEXIST') {
      throw error
    }

    const ownerPid = readLockPid(lockDir, fsApi)
    if (isProcessAlive(ownerPid, processApi)) {
      return {
        acquired: false,
        ownerPid,
        release: () => {}
      }
    }

    fsApi.rmSync(lockDir, { force: true, recursive: true })
    try {
      tryAcquire()
    } catch (retryError) {
      if (retryError?.code !== 'EEXIST') {
        throw retryError
      }

      return {
        acquired: false,
        ownerPid: readLockPid(lockDir, fsApi),
        release: () => {}
      }
    }
  }

  let released = false
  return {
    acquired: true,
    ownerPid: processApi.pid,
    release: () => {
      if (released) return
      released = true
      fsApi.rmSync(lockDir, { force: true, recursive: true })
    }
  }
}

function exitDuplicateApp(app) {
  if (typeof app.exit === 'function') {
    app.exit(0)
    return
  }

  app.quit()
}

function installSingleInstanceGuard({
  app,
  createWindow,
  getMainWindow,
  lockDir,
  log = () => {},
  processApi = process,
  spawnProcess = spawn
}) {
  const resolvedLockDir = lockDir || path.join(app.getPath('userData'), 'hermes-desktop-single-instance.lock')
  const osLock = acquireDirectoryLock(resolvedLockDir, { processApi })

  if (!osLock.acquired) {
    log(`[hermes] another Hermes Desktop instance is already running (pid ${osLock.ownerPid || 'unknown'}); exiting duplicate launch`)
    focusRunningAppBundle({ executablePath: processApi.execPath, platform: processApi.platform, spawnProcess })
    exitDuplicateApp(app)
    return false
  }

  const releaseOsLock = () => osLock.release()
  processApi.once?.('exit', releaseOsLock)

  const hasElectronLock = app.requestSingleInstanceLock()

  if (!hasElectronLock) {
    log('[hermes] another Hermes Desktop instance is already running; exiting duplicate launch')
    osLock.release()
    focusRunningAppBundle({ executablePath: processApi.execPath, platform: processApi.platform, spawnProcess })
    exitDuplicateApp(app)
    return false
  }

  app.on('second-instance', () => {
    log('[hermes] duplicate Hermes Desktop launch detected; focusing the running window')

    const focusOrCreateWindow = () => {
      if (!focusPrimaryWindow(getMainWindow())) {
        createWindow()
      }
    }

    if (typeof app.isReady === 'function' && !app.isReady()) {
      app.once('ready', focusOrCreateWindow)
      return
    }

    focusOrCreateWindow()
  })

  return true
}

module.exports = {
  acquireDirectoryLock,
  focusPrimaryWindow,
  focusRunningAppBundle,
  isProcessAlive,
  installSingleInstanceGuard
}
