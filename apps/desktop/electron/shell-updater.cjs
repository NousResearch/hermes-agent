'use strict'

// 壳自更新(electron-updater 接线)— 更新的是 Electron 壳本体(dmg/zip/exe),
// 和引擎(runtime)的 opt-in 更新(apex-runtime-latest.cjs)是两条互不相扰的
// 通道。策略:全程静默 —— 启动 60s 后首查、之后每 6h 重查;发现新版自动下载
// (autoDownload),下载完成只把状态推给 renderer(侧栏胶囊出「重启以更新」),
// 绝不弹窗;用户点击才 quitAndInstall,不点则退出时自动装(autoInstallOnAppQuit)。
// 检查/下载失败一律记日志吞掉,离线用户零打扰。
//
// 更新源(generic provider)按 平台-架构 分目录:
//   <COS>/desktop/mac-arm64/  <COS>/desktop/mac-x64/  <COS>/desktop/win-x64/
// 而不是平铺在 desktop/ 下 —— mac 两个架构在 CI 里是两个独立矩阵 job,各自
// 产出一份同名 latest-mac.yml,平铺上传会互相覆盖,后传的架构把先传的 feed
// 冲掉(x64 更新器会在 arm64 的 files 列表里找不到 zip)。package.json 里
// build.publish 只配到 desktop/ 基址(让 electron-builder 生成 yml/blockmap),
// 真正生效的 feed URL 以这里 shellUpdateFeedUrl() 运行时 setFeedURL 为准。
//
// 依赖注入设计(同 apex-runtime-latest.cjs):模块不 require electron /
// electron-updater,autoUpdater、ipcMain、broadcast 全部由 main.cjs 传入,
// 这样 node --test 能用假件直接测事件流→IPC 推送,不需要 electron 环境。
//
// hc-473: 同样的注入设计延伸到匿名遥测 —— sendTelemetry 缺省即真实发射器
// (apexnodes-telemetry.cjs),main.cjs 不需要任何接线改动;测试注入假件截获
// 事件而不碰网络。四个上报点对应 dispatch 的 update-available/downloaded/
// applied/error:'shell_update' 一个 stage 覆盖 available(start)→downloaded
// (success)/error(failure)的检查+下载生命周期,'shell_update_apply' 另一个
// stage 覆盖用户点击「重启以更新」之后(只有 start —— quitAndInstall 一旦成功
// 进程即退出重启,这一侧永远看不到 success,是已知且预期的不对称)。
// checking-for-update / update-not-available / download-progress 故意不打点
// (高频 + 无操作性,不在 dispatch 的四项清单里)。

const {
  sendDesktopTelemetry,
  fireTelemetry,
  classifyErrorCategory,
  normalizeDesktopPlatform,
  STATUS_START,
  STATUS_SUCCESS,
  STATUS_FAILURE
} = require('./apexnodes-telemetry.cjs')

const SHELL_UPDATE_EVENT_CHANNEL = 'hermes:shell-update:event'
const SHELL_UPDATE_FEED_BASE = 'https://apexnodes-runtime-202606250443-1300912302.cos.ap-guangzhou.myqcloud.com/desktop'
const SHELL_UPDATE_INITIAL_DELAY_MS = 60_000
const SHELL_UPDATE_RECHECK_INTERVAL_MS = 6 * 60 * 60 * 1000

// 平台-架构 → feed 子目录。linux 目前没有发布通道,给个规整命名兜底(真开
// linux 通道时 CI 侧同样按这个前缀上传即可)。
//
// ★ 这个 `/${os}-${arch}` 子目录是自更新能不能生效的命门(hc-435):CI 把每个
// 架构的同名 latest-mac.yml/latest.yml 传到各自子目录,根 desktop/ 下故意为空
// (curl 根目录必 404)。运行时这里算出的 URL 交给 setFeedURL,electron-updater
// 的 GenericProvider 会取 `<这里的URL>/latest-mac.yml`,所以只要这里塌回根基址
// (os/arch 缺失被拼成空段),updater 就会去读 404 的根 feed → 自更新静默失效。
// 因此空 os/arch 直接抛,绝不让 feed 退化成根目录。
function shellUpdateFeedUrl({ base = SHELL_UPDATE_FEED_BASE, platform = process.platform, arch = process.arch } = {}) {
  const os = platform === 'darwin' ? 'mac' : platform === 'win32' ? 'win' : 'linux'
  if (!arch) {
    throw new Error(`shellUpdateFeedUrl: missing arch (platform=${platform}) — feed would collapse to the empty root desktop/ prefix`)
  }
  return `${base}/${os}-${arch}`
}

// renderer 可见的状态机。phase 单向推进为主,error/idle 可回到 checking:
//   disabled → (终态,dev/无 updater)
//   idle → checking → available → downloading → downloaded(终态,等重启)
//                   ↘ idle(update-not-available)          ↘ error → checking(下轮)
function initialState(disabled) {
  return { phase: disabled ? 'disabled' : 'idle', version: null, percent: null, error: null }
}

/**
 * 装配壳自更新。返回 { getState, checkNow, dispose } —— main.cjs 只管调一次,
 * 返回值主要给测试用(dispose 清定时器/摘监听)。
 *
 * @param {object} options
 * @param {object|null} options.autoUpdater  electron-updater 的 autoUpdater;dev 传 null
 * @param {object} options.ipcMain           electron ipcMain(或测试假件,只需 .handle)
 * @param {boolean} options.isPackaged       app.isPackaged;false 时整体停用
 * @param {(msg: string) => void} [options.log]        接 main.cjs rememberLog
 * @param {(channel: string, payload: object) => void} [options.broadcast]  推给所有窗口
 * @param {string} [options.feedBase]        缺省 COS desktop/ 基址
 * @param {string} [options.platform]        缺省 process.platform
 * @param {string} [options.arch]            缺省 process.arch
 * @param {number} [options.initialDelayMs]  首查延迟,缺省 60s
 * @param {number} [options.recheckIntervalMs] 重查周期,缺省 6h
 * @param {(event:object) => any} [options.sendTelemetry] hc-473 匿名遥测发射
 *   函数,缺省真实 apexnodes-telemetry.cjs;测试传假件截获事件。
 * @param {string|null} [options.appVersion] 当前壳版本(app.getVersion());
 *   缺省 null——遥测里的 app_version 字段本就是可选的。
 */
function createShellUpdater(options) {
  const {
    autoUpdater,
    ipcMain,
    isPackaged,
    log = () => {},
    broadcast = () => {},
    feedBase = SHELL_UPDATE_FEED_BASE,
    platform = process.platform,
    arch = process.arch,
    initialDelayMs = SHELL_UPDATE_INITIAL_DELAY_MS,
    recheckIntervalMs = SHELL_UPDATE_RECHECK_INTERVAL_MS,
    sendTelemetry = sendDesktopTelemetry,
    appVersion = null
  } = options

  const disabled = !isPackaged || !autoUpdater
  const state = initialState(disabled)
  // hc-473: one {platform, arch, app_version} shape reused by every beacon
  // this updater fires; `stage`/`status`/`error_code` vary per call site.
  const telemetryBase = { platform: normalizeDesktopPlatform(platform), arch, app_version: appVersion }

  function setState(patch) {
    Object.assign(state, patch)
    try {
      broadcast(SHELL_UPDATE_EVENT_CHANNEL, { ...state })
    } catch (error) {
      log(`[shell-update] broadcast failed (ignored): ${error && error.message}`)
    }
  }

  // IPC 面在 dev/packaged 两种模式下都注册,renderer 不需要探测 —— dev 里
  // get 返回 disabled,install 拒绝,胶囊自然不出现。
  ipcMain.handle('hermes:shell-update:get', async () => ({ ...state }))
  ipcMain.handle('hermes:shell-update:install', async () => {
    if (disabled || state.phase !== 'downloaded') {
      return { ok: false, error: disabled ? 'disabled' : 'not_downloaded' }
    }
    log(`[shell-update] quitAndInstall requested (version=${state.version || '?'})`)
    // hc-473: 'start' only, by design — a successful quitAndInstall quits
    // this very process to relaunch the updated one, so there is no code
    // path left here to ever observe/report a 'success'. The process exiting
    // cleanly is itself the (unobservable-from-here) success signal.
    fireTelemetry(sendTelemetry, { ...telemetryBase, stage: 'shell_update_apply', status: STATUS_START })
    // setImmediate:先让 IPC 应答回到 renderer 再拆窗口,避免 renderer 在
    // await 上挂到进程退出。win 上 (true, true) = 静默装 + 装完拉起;mac 的
    // Squirrel.Mac 忽略参数,quit 后换包自动重启。
    setImmediate(() => {
      try {
        autoUpdater.quitAndInstall(true, true)
      } catch (error) {
        log(`[shell-update] quitAndInstall failed: ${error && error.message}`)
        setState({ phase: 'error', error: (error && error.message) || String(error) })
        fireTelemetry(sendTelemetry, {
          ...telemetryBase,
          stage: 'shell_update_apply',
          status: STATUS_FAILURE,
          error_code: `shell_update_apply:${classifyErrorCategory(error)}`
        })
      }
    })
    return { ok: true }
  })

  if (disabled) {
    log('[shell-update] disabled (dev / unpackaged build)')
    return { getState: () => ({ ...state }), checkNow: async () => {}, dispose: () => {} }
  }

  // feed URL 塌向根目录 = 自更新静默失效(hc-435)。shellUpdateFeedUrl 对空
  // os/arch 会抛;这里兜住,宁可整体停用(状态=disabled,胶囊不出)也不去读
  // 404 的根 feed、更不因一个异常 arch 把主进程启动带崩。
  let feedUrl
  try {
    feedUrl = shellUpdateFeedUrl({ base: feedBase, platform, arch })
  } catch (error) {
    log(`[shell-update] disabled: cannot resolve per-arch feed URL: ${error && error.message}`)
    setState({ phase: 'disabled' })
    return { getState: () => ({ ...state }), checkNow: async () => {}, dispose: () => {} }
  }
  autoUpdater.autoDownload = true
  autoUpdater.autoInstallOnAppQuit = true
  autoUpdater.allowDowngrade = false
  // electron-updater 内部日志也进 desktop log,出问题时有迹可循;debug 太吵,丢弃。
  autoUpdater.logger = {
    info: msg => log(`[shell-update] ${msg}`),
    warn: msg => log(`[shell-update] warn: ${msg}`),
    error: msg => log(`[shell-update] error: ${msg}`),
    debug: () => {}
  }
  autoUpdater.setFeedURL({ provider: 'generic', url: feedUrl })
  log(`[shell-update] enabled: feed=${feedUrl}`)

  // 事件 → 状态机。'error' 监听必须挂上:EventEmitter 没有 error 监听时 emit
  // 会直接 throw,把静默失败变成主进程崩溃。
  const listeners = [
    // hc-473: checking-for-update fires every ~6h and mostly finds nothing —
    // deliberately NOT telemetered (high-frequency, non-actionable; not one
    // of the dispatch's four beacon points).
    ['checking-for-update', () => setState({ phase: 'checking', error: null })],
    [
      'update-available',
      info => {
        setState({ phase: 'available', version: (info && info.version) || null, error: null })
        log(`[shell-update] update available: ${(info && info.version) || '?'} (auto-downloading)`)
        // hc-473: 'start' of the shell_update funnel — found + autoDownload
        // begins right after this listener returns.
        fireTelemetry(sendTelemetry, { ...telemetryBase, stage: 'shell_update', status: STATUS_START })
      }
    ],
    // hc-473: the common/expected outcome of a check — deliberately silent.
    ['update-not-available', () => setState({ phase: 'idle', percent: null, error: null })],
    [
      'download-progress',
      progress => {
        // 静默下载:只记状态(renderer 胶囊对 downloading 不渲染),不打扰。
        // hc-473: per-tick progress is deliberately NOT telemetered (would
        // flood the beacon with many events per download for no signal).
        setState({ phase: 'downloading', percent: progress && typeof progress.percent === 'number' ? progress.percent : null })
      }
    ],
    [
      'update-downloaded',
      info => {
        setState({ phase: 'downloaded', version: (info && info.version) || state.version, percent: 100, error: null })
        log(`[shell-update] update downloaded: ${(info && info.version) || '?'} (waiting for restart)`)
        // hc-473: 'success' of the shell_update funnel (electron-updater has
        // already verified the download internally by this point).
        fireTelemetry(sendTelemetry, { ...telemetryBase, stage: 'shell_update', status: STATUS_SUCCESS })
      }
    ],
    [
      'error',
      error => {
        // 静默失败:记日志 + 状态,绝不弹 UI。下一轮周期检查会自动重试。
        const message = (error && error.message) || String(error)
        setState({ phase: 'error', error: message })
        log(`[shell-update] error (silent): ${message}`)
        // hc-473: covers BOTH a check failure and a download failure — the
        // underlying autoUpdater API collapses both into this one 'error'
        // event, so this beacon can't distinguish which sub-stage failed any
        // more finely than the existing state machine already does.
        fireTelemetry(sendTelemetry, {
          ...telemetryBase,
          stage: 'shell_update',
          status: STATUS_FAILURE,
          error_code: `shell_update:${classifyErrorCategory(error)}`
        })
      }
    ]
  ]
  for (const [event, handler] of listeners) {
    autoUpdater.on(event, handler)
  }

  async function checkNow() {
    try {
      await autoUpdater.checkForUpdates()
    } catch (error) {
      // checkForUpdates 的失败通常也会走 'error' 事件;这里兜同步 throw /
      // rejection,保证后台定时器永远打不出 unhandled rejection。
      log(`[shell-update] check failed (silent): ${(error && error.message) || error}`)
    }
  }

  // 首查延迟 60s:避开 app 启动的 gateway 拉起/会话 hydrate 高峰(和引擎胶囊
  // 的 30s 静默检查同思路)。定时器 unref,不影响进程退出,也免得测试挂住。
  const initialTimer = setTimeout(() => void checkNow(), initialDelayMs)
  if (typeof initialTimer.unref === 'function') initialTimer.unref()
  const recheckTimer = setInterval(() => void checkNow(), recheckIntervalMs)
  if (typeof recheckTimer.unref === 'function') recheckTimer.unref()

  function dispose() {
    clearTimeout(initialTimer)
    clearInterval(recheckTimer)
    for (const [event, handler] of listeners) {
      autoUpdater.removeListener(event, handler)
    }
  }

  return { getState: () => ({ ...state }), checkNow, dispose }
}

module.exports = {
  SHELL_UPDATE_EVENT_CHANNEL,
  SHELL_UPDATE_FEED_BASE,
  SHELL_UPDATE_INITIAL_DELAY_MS,
  SHELL_UPDATE_RECHECK_INTERVAL_MS,
  createShellUpdater,
  shellUpdateFeedUrl
}
