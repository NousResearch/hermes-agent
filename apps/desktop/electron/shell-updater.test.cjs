'use strict'

const assert = require('node:assert/strict')
const test = require('node:test')
const { EventEmitter } = require('node:events')

// hc-473: keep this suite hermetic regardless of how it's invoked (npm run
// test:desktop:platforms already sets this too, but this file must not rely
// on that -- a bare `node --test electron/shell-updater.test.cjs` must never
// let createShellUpdater's default sendTelemetry touch the real network).
// Tests that assert on beacon content inject their own fake sendTelemetry,
// which always takes priority over this env var.
process.env.APEXNODES_TELEMETRY = 'off'

const {
  SHELL_UPDATE_EVENT_CHANNEL,
  SHELL_UPDATE_FEED_BASE,
  createShellUpdater,
  shellUpdateFeedUrl
} = require('./shell-updater.cjs')

// ---------------------------------------------------------------------------
// 假件:ipcMain 只要 .handle;autoUpdater 是 EventEmitter + 可数的 stub。
// ---------------------------------------------------------------------------

function fakeIpcMain() {
  const handlers = new Map()
  return {
    handle: (channel, fn) => handlers.set(channel, fn),
    invoke: (channel, ...args) => {
      const fn = handlers.get(channel)
      if (!fn) throw new Error(`no handler for ${channel}`)
      return fn({}, ...args)
    },
    handlers
  }
}

function fakeAutoUpdater() {
  const updater = new EventEmitter()
  updater.autoDownload = false
  updater.autoInstallOnAppQuit = false
  updater.allowDowngrade = true
  updater.feed = null
  updater.checkCalls = 0
  updater.installCalls = []
  updater.setFeedURL = options => {
    updater.feed = options
  }
  updater.checkForUpdates = async () => {
    updater.checkCalls += 1
  }
  updater.quitAndInstall = (...args) => {
    updater.installCalls.push(args)
  }
  return updater
}

function harness({ isPackaged = true, autoUpdater = fakeAutoUpdater(), ...rest } = {}) {
  const ipcMain = fakeIpcMain()
  const broadcasts = []
  const logs = []
  const updater = createShellUpdater({
    autoUpdater,
    ipcMain,
    isPackaged,
    log: msg => logs.push(String(msg)),
    broadcast: (channel, payload) => broadcasts.push({ channel, payload }),
    platform: 'darwin',
    arch: 'arm64',
    // 长到测试内永不触发;定时行为单独用短值测。
    initialDelayMs: 60_000,
    recheckIntervalMs: 60_000,
    ...rest
  })
  return { updater, ipcMain, broadcasts, logs, autoUpdater }
}

const flushImmediate = () => new Promise(resolve => setImmediate(resolve))

async function waitUntil(predicate, timeoutMs = 2000) {
  const deadline = Date.now() + timeoutMs
  while (!predicate()) {
    if (Date.now() > deadline) throw new Error('waitUntil timed out')
    await new Promise(resolve => setTimeout(resolve, 5))
  }
}

// ---------------------------------------------------------------------------
// shellUpdateFeedUrl:平台-架构分目录(mac 双架构 latest-mac.yml 同名,平铺会互覆)
// ---------------------------------------------------------------------------

test('shellUpdateFeedUrl keys the feed by platform-arch under the COS base', () => {
  assert.equal(shellUpdateFeedUrl({ platform: 'darwin', arch: 'arm64' }), `${SHELL_UPDATE_FEED_BASE}/mac-arm64`)
  assert.equal(shellUpdateFeedUrl({ platform: 'darwin', arch: 'x64' }), `${SHELL_UPDATE_FEED_BASE}/mac-x64`)
  assert.equal(shellUpdateFeedUrl({ platform: 'win32', arch: 'x64' }), `${SHELL_UPDATE_FEED_BASE}/win-x64`)
  assert.equal(shellUpdateFeedUrl({ platform: 'linux', arch: 'x64' }), `${SHELL_UPDATE_FEED_BASE}/linux-x64`)
  assert.equal(shellUpdateFeedUrl({ base: 'https://cdn.example/d', platform: 'win32', arch: 'arm64' }), 'https://cdn.example/d/win-arm64')
})

// hc-435 命门:feed 必须落在 <base>/<os>-<arch> 子目录,永远比 base 深一层。
// 根 desktop/ 下故意为空(404);一旦 feed 塌回 base,updater 就去读 404 的根
// feed → 自更新静默失效。这条守住「绝不等于 base、也绝不是 base 的直接前缀段」。
test('shellUpdateFeedUrl never collapses to the bare (empty) root base', () => {
  for (const arch of ['arm64', 'x64']) {
    for (const platform of ['darwin', 'win32', 'linux']) {
      const url = shellUpdateFeedUrl({ platform, arch })
      assert.notEqual(url, SHELL_UPDATE_FEED_BASE, `${platform}/${arch} must not equal the root base`)
      assert.ok(url.startsWith(`${SHELL_UPDATE_FEED_BASE}/`), `${platform}/${arch} must live under the base`)
      assert.match(url, /\/(mac|win|linux)-(arm64|x64)$/, `${platform}/${arch} must end with an os-arch segment`)
    }
  }
})

// 空 arch(理论回归:某处把 arch 显式传成空串,或 process.arch 拿到空值)不能
// 悄悄拼出 base+空段的根 feed —— 直接抛,让上层兜成 disabled 而不是去读根 404。
// (arch 属性缺省/undefined 走默认 process.arch,真机上非空,不在此守;这里守的
// 是「拿到了但为空」这个会塌向根目录的危险值。)
test('shellUpdateFeedUrl throws on an empty arch instead of yielding a root-ish feed', () => {
  assert.throws(() => shellUpdateFeedUrl({ platform: 'darwin', arch: '' }), /missing arch/)
  assert.throws(() => shellUpdateFeedUrl({ platform: 'win32', arch: '' }), /missing arch/)
})

// ---------------------------------------------------------------------------
// dev / 未打包:整体停用,但 IPC 面仍在(renderer 无需探测)
// ---------------------------------------------------------------------------

test('unpackaged build: disabled state, install refused, autoUpdater untouched', async () => {
  const { updater, ipcMain, broadcasts, autoUpdater } = harness({ isPackaged: false })

  assert.equal(updater.getState().phase, 'disabled')
  assert.deepEqual(await ipcMain.invoke('hermes:shell-update:get'), {
    phase: 'disabled',
    version: null,
    percent: null,
    error: null
  })

  const install = await ipcMain.invoke('hermes:shell-update:install')
  assert.deepEqual(install, { ok: false, error: 'disabled' })
  await flushImmediate()
  assert.equal(autoUpdater.installCalls.length, 0)
  // 停用态没配置过 updater,也不该有任何状态广播。
  assert.equal(autoUpdater.feed, null)
  assert.equal(broadcasts.length, 0)
})

test('missing autoUpdater (require failed) degrades to disabled instead of throwing', async () => {
  const { updater, ipcMain } = harness({ autoUpdater: null })

  assert.equal(updater.getState().phase, 'disabled')
  assert.deepEqual(await ipcMain.invoke('hermes:shell-update:install'), { ok: false, error: 'disabled' })
})

// ---------------------------------------------------------------------------
// packaged:接线配置(静默下载 + 退出即装 + per-arch generic feed)
// ---------------------------------------------------------------------------

test('packaged build wires silent-download config and the per-arch generic feed', () => {
  const { autoUpdater } = harness()

  assert.equal(autoUpdater.autoDownload, true)
  assert.equal(autoUpdater.autoInstallOnAppQuit, true)
  assert.equal(autoUpdater.allowDowngrade, false)
  assert.deepEqual(autoUpdater.feed, { provider: 'generic', url: `${SHELL_UPDATE_FEED_BASE}/mac-arm64` })
})

// hc-435:packaged 但 arch 拿不到时,宁可整体停用也不 setFeedURL 到根 feed,
// 更不能把 initShellUpdater 带崩(startup 契约:自更新故障绝不拦启动)。
test('packaged build with an unresolvable arch degrades to disabled, never feeds the root', () => {
  const { updater, autoUpdater, logs } = harness({ arch: '' })

  assert.equal(updater.getState().phase, 'disabled')
  // 从未配过 feed(没有塌向根目录),也没崩。
  assert.equal(autoUpdater.feed, null)
  assert.ok(logs.some(line => line.includes('cannot resolve per-arch feed URL')))
})

// ---------------------------------------------------------------------------
// 事件流 → 状态机 + IPC 广播
// ---------------------------------------------------------------------------

test('autoUpdater event flow drives the state machine and broadcasts each transition', async () => {
  const { updater, ipcMain, broadcasts, autoUpdater } = harness()

  autoUpdater.emit('checking-for-update')
  assert.equal(updater.getState().phase, 'checking')

  autoUpdater.emit('update-available', { version: '0.16.1' })
  assert.deepEqual(updater.getState(), { phase: 'available', version: '0.16.1', percent: null, error: null })

  autoUpdater.emit('download-progress', { percent: 42.5 })
  assert.equal(updater.getState().phase, 'downloading')
  assert.equal(updater.getState().percent, 42.5)

  autoUpdater.emit('update-downloaded', { version: '0.16.1' })
  assert.deepEqual(updater.getState(), { phase: 'downloaded', version: '0.16.1', percent: 100, error: null })

  // 每次迁移都推 renderer;载荷是快照不是引用。
  assert.deepEqual(
    broadcasts.map(b => b.channel),
    Array(4).fill(SHELL_UPDATE_EVENT_CHANNEL)
  )
  assert.deepEqual(
    broadcasts.map(b => b.payload.phase),
    ['checking', 'available', 'downloading', 'downloaded']
  )
  broadcasts[3].payload.phase = 'tampered'
  assert.equal(updater.getState().phase, 'downloaded')

  // renderer 挂载后主动拉一次,拿到的是当前快照。
  assert.deepEqual(await ipcMain.invoke('hermes:shell-update:get'), {
    phase: 'downloaded',
    version: '0.16.1',
    percent: 100,
    error: null
  })
})

test('update-not-available returns to idle silently', () => {
  const { updater, autoUpdater } = harness()

  autoUpdater.emit('checking-for-update')
  autoUpdater.emit('update-not-available', { version: '0.16.0' })

  assert.equal(updater.getState().phase, 'idle')
  assert.equal(updater.getState().error, null)
})

test('updater errors stay silent: logged + state only, no throw', () => {
  const { updater, logs, autoUpdater } = harness()

  // 没挂 error 监听的话这里会直接 throw(EventEmitter 语义),等于主进程崩溃。
  autoUpdater.emit('error', new Error('ECONNRESET'))

  assert.equal(updater.getState().phase, 'error')
  assert.equal(updater.getState().error, 'ECONNRESET')
  assert.ok(logs.some(line => line.includes('[shell-update] error (silent): ECONNRESET')))
})

// ---------------------------------------------------------------------------
// quitAndInstall IPC
// ---------------------------------------------------------------------------

test('install IPC refuses before downloaded and installs (silent+relaunch) after', async () => {
  const { ipcMain, autoUpdater } = harness()

  assert.deepEqual(await ipcMain.invoke('hermes:shell-update:install'), { ok: false, error: 'not_downloaded' })
  await flushImmediate()
  assert.equal(autoUpdater.installCalls.length, 0)

  autoUpdater.emit('update-downloaded', { version: '0.16.1' })
  assert.deepEqual(await ipcMain.invoke('hermes:shell-update:install'), { ok: true })
  await flushImmediate()
  // win: isSilent + isForceRunAfter;mac(Squirrel.Mac)忽略参数。
  assert.deepEqual(autoUpdater.installCalls, [[true, true]])
})

test('a throwing quitAndInstall is caught and surfaced as error state, not a crash', async () => {
  const { updater, ipcMain, logs, autoUpdater } = harness()
  autoUpdater.quitAndInstall = () => {
    throw new Error('spawn failed')
  }

  autoUpdater.emit('update-downloaded', { version: '0.16.1' })
  assert.deepEqual(await ipcMain.invoke('hermes:shell-update:install'), { ok: true })
  await flushImmediate()

  assert.equal(updater.getState().phase, 'error')
  assert.ok(logs.some(line => line.includes('quitAndInstall failed: spawn failed')))
})

// ---------------------------------------------------------------------------
// 检查调度:延迟首查 + 周期重查;失败静默
// ---------------------------------------------------------------------------

test('schedules the delayed initial check and periodic rechecks; dispose stops them', async () => {
  const { updater, autoUpdater } = harness({ initialDelayMs: 1, recheckIntervalMs: 5 })

  await waitUntil(() => autoUpdater.checkCalls >= 2)

  updater.dispose()
  const settled = autoUpdater.checkCalls
  await new Promise(resolve => setTimeout(resolve, 25))
  assert.equal(autoUpdater.checkCalls, settled)
})

test('a rejecting checkForUpdates is swallowed and logged (no unhandled rejection)', async () => {
  const { updater, logs, autoUpdater } = harness()
  autoUpdater.checkForUpdates = async () => {
    throw new Error('feed unreachable')
  }

  await updater.checkNow()

  assert.ok(logs.some(line => line.includes('[shell-update] check failed (silent): feed unreachable')))
})

// ---------------------------------------------------------------------------
// hc-473: anonymous shell-update telemetry
// ---------------------------------------------------------------------------

function telemetryHarness(extra = {}) {
  const telemetryEvents = []
  const h = harness({ appVersion: '0.16.7', sendTelemetry: ev => telemetryEvents.push(ev), ...extra })
  return { ...h, telemetryEvents }
}

test('update-available fires a shell_update start beacon', () => {
  const { autoUpdater, telemetryEvents } = telemetryHarness()

  autoUpdater.emit('update-available', { version: '0.16.1' })

  assert.deepEqual(telemetryEvents, [
    { platform: 'mac', arch: 'arm64', app_version: '0.16.7', stage: 'shell_update', status: 'start' }
  ])
})

test('update-downloaded fires a shell_update success beacon', () => {
  const { autoUpdater, telemetryEvents } = telemetryHarness()

  autoUpdater.emit('update-available', { version: '0.16.1' })
  autoUpdater.emit('update-downloaded', { version: '0.16.1' })

  assert.deepEqual(
    telemetryEvents.map(ev => ev.status),
    ['start', 'success']
  )
  assert.ok(telemetryEvents.every(ev => ev.stage === 'shell_update'))
})

test('an updater error fires a shell_update failure beacon with a categorized error_code', () => {
  const { autoUpdater, telemetryEvents } = telemetryHarness()

  autoUpdater.emit('error', new Error('ECONNRESET'))

  assert.deepEqual(telemetryEvents, [
    {
      platform: 'mac',
      arch: 'arm64',
      app_version: '0.16.7',
      stage: 'shell_update',
      status: 'failure',
      error_code: 'shell_update:network'
    }
  ])
})

test('checking-for-update / update-not-available / download-progress fire NO telemetry (high-frequency, non-actionable)', () => {
  const { autoUpdater, telemetryEvents } = telemetryHarness()

  autoUpdater.emit('checking-for-update')
  autoUpdater.emit('update-not-available', { version: '0.16.0' })
  autoUpdater.emit('download-progress', { percent: 42.5 })

  assert.deepEqual(telemetryEvents, [])
})

test('quitAndInstall requested fires a shell_update_apply start beacon (no success beacon is possible — the process exits)', async () => {
  const { ipcMain, autoUpdater, telemetryEvents } = telemetryHarness()

  autoUpdater.emit('update-downloaded', { version: '0.16.1' })
  telemetryEvents.length = 0 // drop the update-downloaded beacon, isolate the install call
  assert.deepEqual(await ipcMain.invoke('hermes:shell-update:install'), { ok: true })
  await flushImmediate()

  assert.deepEqual(telemetryEvents, [
    { platform: 'mac', arch: 'arm64', app_version: '0.16.7', stage: 'shell_update_apply', status: 'start' }
  ])
})

test('a throwing quitAndInstall ALSO fires a shell_update_apply failure beacon', async () => {
  const { ipcMain, autoUpdater, telemetryEvents } = telemetryHarness()
  autoUpdater.quitAndInstall = () => {
    throw new Error('spawn failed')
  }

  autoUpdater.emit('update-downloaded', { version: '0.16.1' })
  telemetryEvents.length = 0
  assert.deepEqual(await ipcMain.invoke('hermes:shell-update:install'), { ok: true })
  await flushImmediate()

  assert.deepEqual(
    telemetryEvents.map(ev => ev.status),
    ['start', 'failure']
  )
  const failure = telemetryEvents.find(ev => ev.status === 'failure')
  assert.equal(failure.error_code, 'shell_update_apply:unknown')
})

test('install IPC refused before downloaded fires no telemetry at all', async () => {
  const { ipcMain, telemetryEvents } = telemetryHarness()

  assert.deepEqual(await ipcMain.invoke('hermes:shell-update:install'), { ok: false, error: 'not_downloaded' })
  await flushImmediate()

  assert.deepEqual(telemetryEvents, [])
})

test('a disabled (unpackaged) updater never touches telemetry', async () => {
  const { ipcMain, telemetryEvents } = telemetryHarness({ isPackaged: false })

  await ipcMain.invoke('hermes:shell-update:install')

  assert.deepEqual(telemetryEvents, [])
})

test('omitting sendTelemetry still drives the state machine correctly (real default, network suppressed by APEXNODES_TELEMETRY=off)', () => {
  const { updater, autoUpdater } = harness()

  autoUpdater.emit('update-available', { version: '0.16.1' })

  assert.equal(updater.getState().phase, 'available')
})
