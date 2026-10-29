import type * as childProcess from 'node:child_process'
import { spawn } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { describe, expect, it, vi } from 'vitest'

import {
  clearLinuxGpuBootMarker,
  createLinuxGpuBootWindow,
  decideLinuxGpuLaunch,
  linuxGpuFallbackMarker,
  readLinuxGpuMarker,
  writeLinuxGpuMarker
} from './linux-gpu-fallback'
import { createNvidiaEglRecovery } from './linux-nvidia-egl-recovery'
import { linuxNoSandboxRelaunch } from './linux-sandbox-relaunch'

vi.mock('node:child_process', async importOriginal => {
  const actual = await importOriginal<typeof childProcess>()

  return { ...actual, spawn: vi.fn(actual.spawn) }
})

const argv = ['/opt/Hermes/hermes', '--ozone-platform=x11', 'hermes://session/abc']

function restoreEnv(saved: NodeJS.ProcessEnv) {
  for (const key of Object.keys(process.env)) {
    if (!(key in saved)) {
      delete process.env[key]
    }
  }

  Object.assign(process.env, saved)
}

describe('Linux runtime --no-sandbox relaunch', () => {
  it('moves a runtime-only --no-sandbox onto argv exactly once', () => {
    const relaunch = linuxNoSandboxRelaunch({
      platform: 'linux',
      runtimeNoSandbox: true,
      argv,
      env: {},
      launchedWithoutSandbox: false
    })

    expect(relaunch).toEqual({
      args: ['--ozone-platform=x11', 'hermes://session/abc', '--no-sandbox'],
      target: process.execPath
    })

    // The relaunched process started unsandboxed, so it never restarts again.
    expect(
      linuxNoSandboxRelaunch({
        platform: 'linux',
        runtimeNoSandbox: true,
        argv: ['/opt/Hermes/hermes', ...relaunch!.args],
        env: {},
        launchedWithoutSandbox: true
      })
    ).toBeNull()
  })

  it('leaves sandboxed launches, launch-time opt-outs and other platforms alone', () => {
    expect(
      linuxNoSandboxRelaunch({
        platform: 'linux',
        runtimeNoSandbox: false,
        argv,
        env: {},
        launchedWithoutSandbox: false
      })
    ).toBeNull()
    expect(
      linuxNoSandboxRelaunch({ platform: 'linux', runtimeNoSandbox: true, argv, env: {}, launchedWithoutSandbox: true })
    ).toBeNull()
    // Windows reads the switch per child launch, so its runtime fallback works in place.
    expect(
      linuxNoSandboxRelaunch({
        platform: 'win32',
        runtimeNoSandbox: true,
        argv,
        env: {},
        launchedWithoutSandbox: false
      })
    ).toBeNull()
  })

  // main.ts wires the NVIDIA recovery's cleanExit in: the restart happens after
  // createNvidiaEglRecovery wrote its `booting` witness, and the next launch
  // must not read that deliberate exit as a GPU abort mid-boot.
  it.skipIf(process.platform !== 'linux')('restarts without leaving the next launch a GPU-abort witness', async () => {
    const saved = { ...process.env }
    const userData = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-sandbox-relaunch-gpu-'))

    const launch = () =>
      createNvidiaEglRecovery({
        bootWindow: createLinuxGpuBootWindow({ platform: 'linux' }),
        userData,
        driverMajor: 600,
        driverVersion: '600.1.2',
        appVersion: '0.0.0',
        platform: 'linux',
        env: {},
        primary: true,
        relaunch: () => undefined,
        log: () => undefined
      })

    try {
      delete process.env.ELECTRON_DISABLE_SANDBOX
      delete process.env.APPIMAGE
      vi.resetModules()
      const { relaunchIfNoSandboxCameTooLate } = await import('./linux-sandbox-relaunch')
      const recovery = launch()

      const relaunched = relaunchIfNoSandboxCameTooLate(
        {
          commandLine: { hasSwitch: name => name === 'no-sandbox' },
          relaunch: () => undefined,
          exit: () => undefined
        },
        () => recovery.cleanExit()
      )

      expect(relaunched).toBe(true)
      expect(launch().decision.enable).toBe(false)
    } finally {
      restoreEnv(saved)
      fs.rmSync(userData, { recursive: true, force: true })
    }
  })

  // main.ts wires the generic ladder's cleanup in too: its `booting` witness was
  // written before the restart, and a re-probe after an update or one earlier
  // interrupted boot would otherwise turn the deliberate exit into sticky
  // software rendering on any Linux GPU.
  it.skipIf(process.platform !== 'linux')(
    'restarts without leaving the generic GPU ladder a boot-abort witness',
    async () => {
      const saved = { ...process.env }
      const userData = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-sandbox-relaunch-linux-gpu-'))

      const launch = () => {
        const decision = decideLinuxGpuLaunch({
          argv: ['/opt/Hermes/hermes'],
          env: {},
          marker: readLinuxGpuMarker(userData),
          appVersion: '2.0.0',
          remoteDisplayReason: null,
          nvidiaFallbackActive: false
        })

        writeLinuxGpuMarker(userData, decision.nextMarker)

        return decision
      }

      try {
        delete process.env.ELECTRON_DISABLE_SANDBOX
        delete process.env.APPIMAGE
        vi.resetModules()
        const { relaunchIfNoSandboxCameTooLate } = await import('./linux-sandbox-relaunch')

        // An update re-probing an older version's fallback; one boot interrupted earlier;
        // and a genuine fallback of this version, which must stay sticky.
        for (const [prior, engagedAfter] of [
          [linuxGpuFallbackMarker('gpu-crash', '1.0.0'), false],
          [{ state: 'booting' as const }, false],
          [linuxGpuFallbackMarker('gpu-crash', '2.0.0'), true]
        ] as const) {
          writeLinuxGpuMarker(userData, prior)
          const decision = launch()

          relaunchIfNoSandboxCameTooLate(
            {
              commandLine: { hasSwitch: name => name === 'no-sandbox' },
              relaunch: () => undefined,
              exit: () => undefined
            },
            () => clearLinuxGpuBootMarker(userData, decision.nextMarker.state === 'fallback')
          )

          expect(launch().enable).toBe(engagedAfter)
        }
      } finally {
        restoreEnv(saved)
        fs.rmSync(userData, { recursive: true, force: true })
      }
    }
  )

  // Electron's relauncher execs its target with PR_SET_NO_NEW_PRIVS: the image's
  // runtime then cannot FUSE-mount, and the old mount dies with this process.
  it.skipIf(process.platform !== 'linux')(
    'starts a mounted AppImage again as a detached child once this process is gone',
    async () => {
      const saved = { ...process.env }
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-appimage-relaunch-'))
      const report = path.join(dir, 'report')
      const image = path.join(dir, 'Hermes.AppImage')
      const shortDir = '/run/user/1000/hermes-desktop'
      const events: string[] = []

      fs.writeFileSync(
        image,
        `#!/bin/sh\nprintf '%s\\n' "$*" "$(cut -d' ' -f6 /proc/$$/stat) $$" "$TMPDIR" "\${HERMES_DESKTOP_READY_FD:-unset}" > "${report}.part"\nmv "${report}.part" "${report}"\n`,
        { mode: 0o755 }
      )

      try {
        process.env.TMPDIR = shortDir
        process.env.HERMES_DESKTOP_TMPDIR = dir
        process.env.HERMES_DESKTOP_READY_FD = '3'
        delete process.env.ELECTRON_DISABLE_SANDBOX
        vi.resetModules()
        const { relaunchIfNoSandboxCameTooLate } = await import('./linux-sandbox-relaunch')
        const { createAppRelauncher } = await import('./app-relaunch')

        process.env.TMPDIR = dir
        delete process.env.HERMES_DESKTOP_TMPDIR
        process.env.APPIMAGE = image
        process.env.APPDIR = path.dirname(process.execPath)

        const relauncher = createAppRelauncher(
          {
            commandLine: { hasSwitch: name => name === 'no-sandbox' },
            relaunch: () => events.push('electron relaunch'),
            releaseSingleInstanceLock: () => events.push('release'),
            exit: code => events.push(`exit ${code}`)
          },
          'linux'
        )

        // The helper waits while it is still this process's child; a shell that starts it in the
        // background (in a session of its own, as the detached spawn gives it) and exits stands in.
        const realSpawn = (await vi.importActual<typeof childProcess>('node:child_process')).spawn
        vi.mocked(spawn).mockImplementationOnce((file, args, options) =>
          realSpawn('/bin/sh', ['-c', 'exec 4<&0; setsid "$@" <&4 4<&- &', 'sh', file, ...args], options)
        )
        relaunchIfNoSandboxCameTooLate(relauncher, () => events.push('beforeExit'))
        expect(events).toEqual(['beforeExit', 'release', 'exit 0'])
        // This process is gone: the waiting helper's stdin closes.
        vi.mocked(spawn).mock.results.at(-1)!.value.stdin.end()

        await vi.waitFor(() => expect(fs.existsSync(report)).toBe(true), { timeout: 5000 })
        const [args, session, tmpdir, readyFd] = fs.readFileSync(report, 'utf8').split('\n')
        const [sid, pid] = session.split(' ')

        expect(args.split(' ').at(-1)).toBe('--no-sandbox')
        // Its own session: nothing that ends this process's session takes it down.
        expect(sid).toBe(pid)
        expect(tmpdir).toBe(shortDir)
        expect(readyFd).toBe('unset')
      } finally {
        restoreEnv(saved)
        fs.rmSync(dir, { recursive: true, force: true })
      }
    }
  )

  // --appimage-extract-and-run users run without FUSE, which the image needs;
  // the extracted directory is a plain one Electron can relaunch from.
  it('relaunches the extracted executable of --appimage-extract-and-run, not the image', () => {
    const appDir = '/run/user/1000/appimage_extracted_0123456789abcdef'

    expect(
      linuxNoSandboxRelaunch({
        platform: 'linux',
        runtimeNoSandbox: true,
        argv: [`${appDir}/hermes`],
        execPath: `${appDir}/hermes`,
        env: { APPIMAGE: '/home/u/Apps/Hermes.AppImage', APPDIR: appDir },
        launchedWithoutSandbox: false
      })
    ).toEqual({ args: ['--no-sandbox'], target: `${appDir}/hermes` })
  })

  it('relaunches the AppImage itself rather than its vanishing mount', () => {
    const relaunch = linuxNoSandboxRelaunch({
      platform: 'linux',
      runtimeNoSandbox: true,
      argv: ['/tmp/.mount_HermesX/hermes'],
      execPath: '/tmp/.mount_HermesX/hermes',
      env: { APPIMAGE: '/home/u/Apps/Hermes.AppImage', APPDIR: '/tmp/.mount_HermesX' },
      launchedWithoutSandbox: false
    })

    expect(relaunch).toEqual({
      args: ['--no-sandbox'],
      target: '/home/u/Apps/Hermes.AppImage'
    })
  })

  // The type-2 runtime mounts under $TMPDIR/.mount_<name><random>; a launch that
  // lost APPDIR still runs from that mount, so APPIMAGE is still ours.
  it('relaunches the AppImage when only APPIMAGE survives but this runs from its mount', () => {
    const relaunch = linuxNoSandboxRelaunch({
      platform: 'linux',
      runtimeNoSandbox: true,
      argv: ['/run/user/1000/.mount_HermesQ7x2Lp/hermes'],
      execPath: '/run/user/1000/.mount_HermesQ7x2Lp/hermes',
      env: { APPIMAGE: '/home/u/Apps/Hermes.AppImage' },
      launchedWithoutSandbox: false
    })

    expect(relaunch).toEqual({
      args: ['--no-sandbox'],
      target: '/home/u/Apps/Hermes.AppImage'
    })
  })

  // AppImage apps (terminals, IDEs) export APPIMAGE/APPDIR to everything they
  // start; Hermes launched from one must relaunch itself, not that app.
  it('relaunches this executable when APPIMAGE was inherited from another AppImage', () => {
    for (const env of [
      { APPIMAGE: '/home/u/Apps/Terminal.AppImage' },
      { APPIMAGE: '/home/u/Apps/Terminal.AppImage', APPDIR: '/tmp/.mount_TermY' },
      { APPIMAGE: '/home/u/Apps/Terminal.AppImage', APPDIR: '/opt/Herm' }
    ]) {
      expect(
        linuxNoSandboxRelaunch({
          platform: 'linux',
          runtimeNoSandbox: true,
          argv,
          execPath: '/opt/Hermes/hermes',
          env,
          launchedWithoutSandbox: false
        })
      ).toEqual({
        args: ['--ozone-platform=x11', 'hermes://session/abc', '--no-sandbox'],
        target: '/opt/Hermes/hermes'
      })
    }
  })
})
