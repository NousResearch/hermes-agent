import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

import type { App, IpcMain } from 'electron'

import { detectRemoteDisplay, resolveLinuxPasswordStore } from './bootstrap-platform'
import { describeDevCdpDecision, resolveDevCdpPort } from './dev-cdp'
import {
  alreadyHasNoSandbox,
  buildNoSandboxRelaunchArgs,
  decideWindowsSandboxLaunch,
  fallbackMarker,
  grantAllApplicationPackagesAcl,
  readSandboxMarker,
  type SandboxFallbackReason,
  shouldAttemptAclRepair,
  shouldRelaunchForGpuSandboxCrash,
  writeSandboxMarker
} from './windows-sandbox-fallback'

export interface DesktopPlatformPreflightOptions {
  app: App
  ipcMain: IpcMain
  devServer: string | undefined
  env?: NodeJS.ProcessEnv
  argv?: string[]
  execPath?: string
  exitAfterBackendShutdown: (code: number) => Promise<void>
  isPackaged: boolean
  isWindows: boolean
  isWsl: boolean
  platform?: NodeJS.Platform
}

// All switches and the sandbox marker must be installed before Electron's
// ready event. Keep their original order and return one mutable sandbox state
// shared by startup, renderer-crash recovery, and quit finalization.
export function installDesktopPlatformPreflightRuntime(options: DesktopPlatformPreflightOptions) {
  const { app, ipcMain, devServer, exitAfterBackendShutdown, isPackaged, isWindows, isWsl } = options
  const env = options.env ?? process.env
  const argv = options.argv ?? process.argv
  const execPath = options.execPath ?? process.execPath
  const platform = options.platform ?? process.platform

  // Remote displays (SSH X11 forwarding, VNC, RDP) make Chromium's GPU
  // compositor flicker — accelerated layers can't be presented cleanly over the
  // wire, so the window flashes during scroll/streaming/animation. Local
  // Windows/macOS (and WSLg, which renders locally via vGPU) composite on the
  // GPU and never see it. Fall back to software rendering when a remote display
  // is detected; it's rock-steady over the wire and the CPU cost is negligible
  // next to the connection's latency. Must run before app `ready` — these
  // switches only apply pre-launch. Override with HERMES_DESKTOP_DISABLE_GPU
  // (1/true → always disable, 0/false → keep GPU on).
  const remoteDisplayReason = detectRemoteDisplay({ env, platform })

  if (remoteDisplayReason) {
    app.disableHardwareAcceleration()
    // Belt-and-suspenders for X11/VNC, where the Viz compositor can still glitch
    // with only --disable-gpu: force compositing onto the CPU too.
    app.commandLine.appendSwitch('disable-gpu-compositing')
    console.log(
      `[hermes] remote display detected (${remoteDisplayReason}); disabling GPU hardware acceleration to prevent flicker`
    )
  }

  // Renderer debugging port. On for dev-server runs (`hgui` / `npm run dev`) so
  // the CDP tooling in scripts/ can attach; never for a packaged build — see
  // electron/dev-cdp.ts. Must run before app `ready` like the switches above;
  // Chromium binds it at launch.
  const devCdp = resolveDevCdpPort({ env, isPackaged, devServer })

  if (devCdp.port) {
    app.commandLine.appendSwitch('remote-debugging-port', String(devCdp.port))
    // Loopback only. Chromium already defaults to 127.0.0.1, but say it out loud
    // so a future edit can't widen it by omission.
    app.commandLine.appendSwitch('remote-debugging-address', '127.0.0.1')
    console.log(
      `[hermes] renderer debugging on http://127.0.0.1:${devCdp.port} — anything that can reach it ` +
        'can run code in the renderer. HERMES_DESKTOP_CDP_PORT=off to disable.'
    )
  } else {
    const why = describeDevCdpDecision(devCdp)

    if (why) {
      console.warn(`[hermes] ${why}`)
    }
  }

  // WSLg: Chromium blocklists the Mesa vGPU → software compositing → typing lag.
  // /dev/dxg means a real GPU is available; un-blocklist it. Skipped when a remote
  // display already forced software (SSH'd-into-WSL).
  if (isWsl && !remoteDisplayReason && fs.existsSync('/dev/dxg')) {
    app.commandLine.appendSwitch('ignore-gpu-blocklist')
    app.commandLine.appendSwitch('enable-gpu-rasterization')
    app.commandLine.appendSwitch('enable-zero-copy')
    console.log('[hermes] WSL GPU passthrough (/dev/dxg) detected; enabling GPU acceleration')
  }

  // Linux: point Chromium at the session's keychain backend so safeStorage can
  // encrypt remote gateway tokens (hardening.ts refuses to persist them without
  // it). The value arrives via HERMES_DESKTOP_PASSWORD_STORE, bridged by the
  // `hermes desktop` launcher from detection or `desktop.password_store` in
  // config.yaml. Must run before app `ready` — the switch only applies pre-launch.
  const passwordStore = resolveLinuxPasswordStore({ env, platform })

  if (passwordStore.warning) {
    console.warn(`[hermes] ${passwordStore.warning}`)
  }

  if (passwordStore.store) {
    app.commandLine.appendSwitch('password-store', passwordStore.store)
    console.log(`[hermes] using password-store backend: ${passwordStore.store}`)
  }

  // Windows sandbox / GPU breakpoint crash recovery (#38216).
  // Some hosts kill Chromium's sandboxed GPU/renderer children with 0x80000003.
  // The sticky marker recovers shortcut launches that bypass `hermes desktop`.
  const sandboxState: {
    fallbackActive: boolean
    fallbackSticky: boolean
    fallbackReason: SandboxFallbackReason
    noSandboxRelaunchAttempted: boolean
  } = {
    fallbackActive: false,
    fallbackSticky: false,
    fallbackReason: 'boot-loop',
    noSandboxRelaunchAttempted: false
  }

  if (isWindows) {
    const windowsUserData = app.getPath('userData')
    const priorMarker = readSandboxMarker(windowsUserData)

    // Best-effort ACL repair only after an aborted boot or engaged fallback.
    // The install dir receives AppContainer read access; userData never does.
    if (shouldAttemptAclRepair(priorMarker)) {
      const exeDir = path.dirname(execPath)
      const acl = grantAllApplicationPackagesAcl(exeDir, { execFileSync })

      if (acl.ok) {
        console.log(`[hermes] granted ALL APPLICATION PACKAGES RX on ${exeDir} (#38216)`)
      } else if (acl.error && acl.error !== 'missing-target-or-exec') {
        console.warn(`[hermes] AppContainer ACL grant failed on ${exeDir}: ${acl.error}`)
      }
    }

    const sandboxDecision = decideWindowsSandboxLaunch({
      platform,
      argv,
      env,
      marker: priorMarker,
      appVersion: app.getVersion()
    })

    sandboxState.fallbackActive = sandboxDecision.enable
    sandboxState.fallbackSticky = sandboxDecision.nextMarker.state === 'fallback'

    if (sandboxDecision.nextMarker.state === 'fallback' && sandboxDecision.nextMarker.reason) {
      sandboxState.fallbackReason = sandboxDecision.nextMarker.reason
    }

    if (sandboxDecision.enable && sandboxDecision.reason !== 'already-enabled') {
      app.commandLine.appendSwitch('no-sandbox')
      env.ELECTRON_DISABLE_SANDBOX = '1'
      console.log(
        `[hermes] Windows sandbox fallback enabled (${sandboxDecision.reason}); launching with --no-sandbox (#38216)`
      )
    }

    writeSandboxMarker(windowsUserData, sandboxDecision.nextMarker)

    // Catch the first GPU breakpoint death and relaunch before Chromium's
    // "GPU process isn't usable" FATAL abort ends the process with no recovery.
    app.on('child-process-gone', (_event, details) => {
      if (
        !shouldRelaunchForGpuSandboxCrash({
          platform,
          details,
          alreadyNoSandbox: sandboxState.fallbackActive || alreadyHasNoSandbox(argv, env),
          relaunchAttempted: sandboxState.noSandboxRelaunchAttempted
        })
      ) {
        return
      }

      sandboxState.noSandboxRelaunchAttempted = true
      sandboxState.fallbackActive = true
      sandboxState.fallbackSticky = true
      sandboxState.fallbackReason = 'gpu-breakpoint'

      try {
        writeSandboxMarker(app.getPath('userData'), fallbackMarker('gpu-breakpoint', app.getVersion()))
      } catch {
        void 0
      }

      console.warn(
        `[hermes] Windows GPU sandbox crashed (exit=${details?.exitCode}); relaunching once with --no-sandbox (#38216)`
      )

      try {
        app.relaunch({ args: buildNoSandboxRelaunchArgs(argv.slice(1)) })
        void exitAfterBackendShutdown(0)
      } catch (error) {
        console.error(`[hermes] --no-sandbox relaunch failed: ${error?.message || error}`)
      }
    })
  }

  ipcMain.handle('hermes:get-remote-display-reason', () => remoteDisplayReason)

  // Keep the renderer's PROCESS priority normal while its windows are hidden.
  // Timer throttling remains governed by the streaming-scoped runtime dial.
  app.commandLine.appendSwitch('disable-renderer-backgrounding')

  return { remoteDisplayReason, sandboxState }
}
