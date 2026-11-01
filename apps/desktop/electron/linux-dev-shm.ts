/**
 * Linux /dev/shm usability probe for #138277.
 *
 * Chromium's renderer processes allocate their shared memory under /dev/shm.
 * On hosts where that mount refuses creates — unmounted, a size-0 tmpfs, or
 * policy-denied even though `ls -ld /dev/shm` looks healthy (#138277:
 * CentOS Stream 9, `drwxrwxrwt`, every renderer dies pre-main at
 * `platform_shared_memory_region_posix` with "Creating shared memory in
 * /dev/shm/... failed: No such file or directory") — the renderer
 * crash-loops and the app never reaches a usable UI.
 *
 * Probe the mount with a real create+write+unlink before `app` is ready: a
 * stat or permission check passes on exactly the hosts that still refuse
 * creates, only a real open reproduces what the renderer is about to do.
 * When the probe cannot complete, tell Chromium to back its shared memory
 * with temp files instead (`--disable-dev-shm-usage`, the standard remedy for
 * shm-less containers). Healthy hosts keep POSIX shm and its zero-copy
 * performance — the fallback costs a temp-file-backed path only where
 * /dev/shm is already unusable.
 *
 * Every step is best-effort, like the crash-diagnostics wiring beside it: a
 * probe that throws degrades to Chromium's default behavior, never to a
 * desktop that dies before app readiness.
 */

import fs from 'node:fs'
import path from 'node:path'

export const DEV_SHM_DIR = '/dev/shm'
export const DEV_SHM_FALLBACK_SWITCH = 'disable-dev-shm-usage'

export interface DevShmProbeOutcome {
  ok: boolean
  /** Failure detail for the desktop log when the mount is not usable. */
  error?: string
}

/**
 * Create, write and unlink a probe file in *shmDir*. Writing matters: a
 * size-0 tmpfs admits the create but fails the write that a renderer's
 * shared region would need. The probe name carries pid+ms so a crashed
 * prior probe can never make the next one fail-create with EEXIST.
 */
export function probeDevShmUsable(shmDir: string = DEV_SHM_DIR): DevShmProbeOutcome {
  const probePath = path.join(shmDir, `.hermes-shm-probe-${process.pid}-${Date.now()}`)

  try {
    const fd = fs.openSync(probePath, 'wx', 0o600)

    try {
      fs.writeFileSync(fd, 'hermes /dev/shm probe\n')
    } finally {
      fs.closeSync(fd)
    }

    // A create+write that succeeded means the mount is usable for exactly
    // what renderers do; an unlink the host refuses must not misread as an
    // unusable mount (one stale probe file is harmless either way).
    try {
      fs.unlinkSync(probePath)
    } catch {
      // Keep { ok: true } — the renderer never needs to unlink its region.
    }

    return { ok: true }
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error)

    return { ok: false, error: detail }
  }
}

/** The side effects the fallback needs, injected so the failure paths are provable. */
export interface DevShmHost {
  /** Run the real /dev/shm probe. */
  probe(): DevShmProbeOutcome
  /** `app.commandLine.appendSwitch` — must run before `app` is ready. */
  appendSwitch(name: string): void
  /** Desktop log line, mirroring the sandbox-fallback announcement. */
  log(message: string): void
}

export function applyLinuxDevShmFallback(platform: NodeJS.Platform, host: DevShmHost): void {
  if (platform !== 'linux') {
    return
  }

  try {
    const outcome = host.probe()

    if (outcome.ok) {
      return
    }

    host.appendSwitch(DEV_SHM_FALLBACK_SWITCH)
    host.log(
      `[hermes] /dev/shm is not usable (${outcome.error ?? 'unknown error'}); ` +
        `launching with --${DEV_SHM_FALLBACK_SWITCH} (#138277)`
    )
  } catch {
    // Ignore — an unprobeable mount keeps Chromium's default behavior.
  }
}
