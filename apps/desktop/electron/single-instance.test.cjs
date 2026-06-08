const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')

const {
  acquireDirectoryLock,
  focusPrimaryWindow,
  focusRunningAppBundle,
  installSingleInstanceGuard
} = require('./single-instance.cjs')

function tempLockPath(t) {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-single-instance-test-'))
  t.after(() => fs.rmSync(tempDir, { force: true, recursive: true }))
  return path.join(tempDir, 'desktop.lock')
}

function makeApp({ hasLock = true, ready = true, userData = os.tmpdir() } = {}) {
  const listeners = new Map()
  const onceListeners = new Map()
  const app = {
    exitCalls: [],
    quitCalls: 0,
    requestSingleInstanceLock: () => hasLock,
    quit: () => {
      app.quitCalls += 1
    },
    exit: code => {
      app.exitCalls.push(code)
    },
    getPath: name => {
      assert.equal(name, 'userData')
      return userData
    },
    isReady: () => ready,
    on: (event, handler) => {
      listeners.set(event, handler)
    },
    once: (event, handler) => {
      onceListeners.set(event, handler)
    },
    emit: event => {
      listeners.get(event)?.()
    },
    emitOnce: event => {
      onceListeners.get(event)?.()
    },
    hasListener: event => listeners.has(event),
    hasOnceListener: event => onceListeners.has(event)
  }

  return app
}

function makeWindow({ destroyed = false, minimized = false, visible = true } = {}) {
  return {
    focusCalls: 0,
    restoreCalls: 0,
    showCalls: 0,
    isDestroyed: () => destroyed,
    isMinimized: () => minimized,
    isVisible: () => visible,
    restore() {
      this.restoreCalls += 1
      minimized = false
    },
    show() {
      this.showCalls += 1
      visible = true
    },
    focus() {
      this.focusCalls += 1
    }
  }
}

function makeProcess({ alivePids = new Set(), pid = 1000, platform = 'darwin' } = {}) {
  return {
    execPath: '/Applications/Hermes.app/Contents/MacOS/Hermes',
    pid,
    platform,
    kill: targetPid => {
      if (!alivePids.has(targetPid)) {
        const error = new Error('No such process')
        error.code = 'ESRCH'
        throw error
      }
    },
    once: () => {}
  }
}

test('focusPrimaryWindow restores shows and focuses an existing window', () => {
  const window = makeWindow({ minimized: true, visible: false })

  assert.equal(focusPrimaryWindow(window), true)
  assert.equal(window.restoreCalls, 1)
  assert.equal(window.showCalls, 1)
  assert.equal(window.focusCalls, 1)
})

test('focusPrimaryWindow rejects missing or destroyed windows', () => {
  assert.equal(focusPrimaryWindow(null), false)
  assert.equal(focusPrimaryWindow(makeWindow({ destroyed: true })), false)
})

test('focusRunningAppBundle opens the containing macOS .app bundle', () => {
  const calls = []

  const focused = focusRunningAppBundle({
    executablePath: '/Applications/Hermes.app/Contents/MacOS/Hermes',
    platform: 'darwin',
    spawnProcess: (command, args, options) => {
      calls.push({ args, command, options })
      return { unref: () => {} }
    }
  })

  assert.equal(focused, true)
  assert.equal(calls[0].command, 'open')
  assert.deepEqual(calls[0].args, ['/Applications/Hermes.app'])
  assert.equal(calls[0].options.detached, true)
})

test('focusRunningAppBundle ignores non-macOS processes', () => {
  assert.equal(
    focusRunningAppBundle({
      executablePath: '/opt/Hermes/Hermes',
      platform: 'linux',
      spawnProcess: () => {
        throw new Error('spawn should not be called')
      }
    }),
    false
  )
})

test('acquireDirectoryLock creates and releases the owner lock', t => {
  const lockDir = tempLockPath(t)
  const processApi = makeProcess({ pid: 42 })

  const lock = acquireDirectoryLock(lockDir, { processApi })

  assert.equal(lock.acquired, true)
  assert.equal(JSON.parse(fs.readFileSync(path.join(lockDir, 'owner.json'), 'utf8')).pid, 42)

  lock.release()
  assert.equal(fs.existsSync(lockDir), false)
})

test('acquireDirectoryLock rejects a live owner process', t => {
  const lockDir = tempLockPath(t)
  fs.mkdirSync(lockDir)
  fs.writeFileSync(path.join(lockDir, 'owner.json'), JSON.stringify({ pid: 123 }))

  const lock = acquireDirectoryLock(lockDir, {
    processApi: makeProcess({ alivePids: new Set([123]), pid: 456 })
  })

  assert.equal(lock.acquired, false)
  assert.equal(lock.ownerPid, 123)
})

test('acquireDirectoryLock replaces a stale owner process', t => {
  const lockDir = tempLockPath(t)
  fs.mkdirSync(lockDir)
  fs.writeFileSync(path.join(lockDir, 'owner.json'), JSON.stringify({ pid: 123 }))

  const lock = acquireDirectoryLock(lockDir, {
    processApi: makeProcess({ alivePids: new Set(), pid: 456 })
  })

  assert.equal(lock.acquired, true)
  assert.equal(JSON.parse(fs.readFileSync(path.join(lockDir, 'owner.json'), 'utf8')).pid, 456)
})

test('installSingleInstanceGuard exits duplicate process when OS lock has a live owner', t => {
  const app = makeApp()
  const logs = []
  const lockDir = tempLockPath(t)
  const spawned = []
  let created = 0
  fs.mkdirSync(lockDir)
  fs.writeFileSync(path.join(lockDir, 'owner.json'), JSON.stringify({ pid: 123 }))

  const installed = installSingleInstanceGuard({
    app,
    createWindow: () => {
      created += 1
    },
    getMainWindow: () => null,
    lockDir,
    log: message => logs.push(message),
    processApi: makeProcess({ alivePids: new Set([123]), pid: 456 }),
    spawnProcess: (command, args) => {
      spawned.push({ args, command })
      return { unref: () => {} }
    }
  })

  assert.equal(installed, false)
  assert.deepEqual(app.exitCalls, [0])
  assert.equal(created, 0)
  assert.equal(app.hasListener('second-instance'), false)
  assert.match(logs.join('\n'), /already running/)
  assert.deepEqual(spawned, [{ args: ['/Applications/Hermes.app'], command: 'open' }])
})

test('installSingleInstanceGuard exits duplicate process when Electron lock is unavailable', t => {
  const app = makeApp({ hasLock: false })
  const lockDir = tempLockPath(t)

  const installed = installSingleInstanceGuard({
    app,
    createWindow: () => {},
    getMainWindow: () => null,
    lockDir,
    processApi: makeProcess()
  })

  assert.equal(installed, false)
  assert.deepEqual(app.exitCalls, [0])
  assert.equal(fs.existsSync(lockDir), false)
})

test('installSingleInstanceGuard focuses existing window on a second launch', t => {
  const app = makeApp()
  const window = makeWindow()
  const lockDir = tempLockPath(t)
  let created = 0

  const installed = installSingleInstanceGuard({
    app,
    createWindow: () => {
      created += 1
    },
    getMainWindow: () => window,
    lockDir
  })

  assert.equal(installed, true)
  app.emit('second-instance')
  assert.equal(window.focusCalls, 1)
  assert.equal(created, 0)
})

test('installSingleInstanceGuard creates a window when second launch finds none', t => {
  const app = makeApp()
  const lockDir = tempLockPath(t)
  let created = 0

  installSingleInstanceGuard({
    app,
    createWindow: () => {
      created += 1
    },
    getMainWindow: () => null,
    lockDir
  })

  app.emit('second-instance')
  assert.equal(created, 1)
})

test('installSingleInstanceGuard waits for app readiness before creating a missing window', t => {
  const app = makeApp({ ready: false })
  const lockDir = tempLockPath(t)
  let created = 0

  installSingleInstanceGuard({
    app,
    createWindow: () => {
      created += 1
    },
    getMainWindow: () => null,
    lockDir
  })

  app.emit('second-instance')
  assert.equal(created, 0)
  assert.equal(app.hasOnceListener('ready'), true)

  app.emitOnce('ready')
  assert.equal(created, 1)
})
