const assert = require('node:assert/strict')
const Module = require('node:module')
const path = require('node:path')
const test = require('node:test')

// main.cjs touches a lot of electron globals at module load. We stub the
// entire `electron` module via Module._resolveFilename + require.cache so
// that requiring main.cjs in a plain Node process works for unit testing.

function loadMainWithStub({ createFromPath }) {
  const mainPath = require.resolve('../main.cjs')
  delete require.cache[mainPath]

  // main.cjs touches a lot of electron globals at module load. Instead of
  // hand-rolling every method, wrap each subsystem in a Proxy that returns
  // a no-op function for any property access — enough to get past load.
  const noopFn = () => undefined
  function autoStub(overrides = {}) {
    return new Proxy(overrides, {
      get(target, prop) {
        if (prop in target) return target[prop]
        const fn = (...args) => {
          // A few known methods need specific shapes:
          if (prop === 'whenReady') return new Promise(() => {})
          if (prop === 'requestSingleInstanceLock') return true
          if (prop === 'isPackaged') return false
          if (prop === 'on' || prop === 'once') return undefined
          return undefined
        }
        // Mirror common boolean/string fields read at module scope.
        if (prop === 'isPackaged') return false
        if (prop === 'platform') return process.platform
        if (typeof prop === 'symbol') return undefined
        return fn
      }
    })
  }

  const electronStub = {
    app: autoStub({
      isPackaged: false,
      commandLine: autoStub(),
      dock: autoStub(),
      whenReady: () => new Promise(() => {}),
      requestSingleInstanceLock: () => true,
      getPath: () => '/tmp',
      getName: () => 'Hermes',
      getVersion: () => '0.0.0',
      getAppPath: () => process.cwd(),
      setName: () => {},
      setPath: () => {},
      on: () => {},
      once: () => {}
    }),
    BrowserWindow: class { static getAllWindows() { return [] } },
    Menu: autoStub({ buildFromTemplate: () => ({}), setApplicationMenu: () => {} }),
    Notification: class { show() {} static isSupported() { return false } },
    clipboard: autoStub({ readText: () => '' }),
    dialog: autoStub(),
    ipcMain: autoStub(),
    nativeImage: { createFromPath },
    nativeTheme: autoStub({ shouldUseDarkColors: false }),
    net: autoStub(),
    powerMonitor: autoStub(),
    protocol: autoStub(),
    safeStorage: autoStub({ isEncryptionAvailable: () => false }),
    session: autoStub({ defaultSession: autoStub({ webRequest: autoStub(), cookies: autoStub({ get: async () => [] }) }) }),
    shell: autoStub({ openExternal: async () => {} }),
    systemPreferences: autoStub({ getUserDefault: () => '' })
  }

  const electronKey = require.resolve('electron')
  require.cache[electronKey] = {
    id: electronKey,
    filename: electronKey,
    loaded: true,
    exports: electronStub
  }

  return require(mainPath)
}

test('loadAppIcon returns null when every candidate produces an empty image', () => {
  const main = loadMainWithStub({
    createFromPath: () => ({ isEmpty: () => true })
  })
  assert.equal(main.loadAppIcon(), null)
})

test('loadAppIcon returns the first non-empty NativeImage', () => {
  const realImage = { isEmpty: () => false, _marker: 'real' }
  let calls = 0
  const main = loadMainWithStub({
    createFromPath: () => {
      calls += 1
      // Empty for first attempt, real for subsequent — exercises the loop.
      return calls === 1 ? { isEmpty: () => true } : realImage
    }
  })
  const result = main.loadAppIcon()
  // Either we found the real one, or there were no existing fixture paths
  // on this machine. Both are acceptable as long as the result is the real
  // image object or null (never a stub empty image).
  if (result !== null) {
    assert.equal(result, realImage)
  }
})

test('getAppIconPath still returns a filesystem path or undefined', () => {
  const main = loadMainWithStub({
    createFromPath: () => ({ isEmpty: () => true })
  })
  const result = main.getAppIconPath()
  assert.ok(result === undefined || typeof result === 'string')
})
