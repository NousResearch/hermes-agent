/**
 * Does this Linux host run Chromium under a user-namespace sandbox?
 *
 * The launcher already prefers `--disable-setuid-sandbox` (userns) over
 * setuid. On such a host the renderer gets a chroot where `/dev/shm` is
 * reachable, so the sandboxed path renders. `--no-sandbox` skips the broker
 * and the chroot, and the renderer then cannot create its shared memory and
 * aborts in a SIGILL loop — so on these hosts dropping the sandbox is not a
 * recovery step but the crash itself, and once the sticky marker engages it
 * never clears (the app reports the same `0.0.0` version on every source
 * build, so the ladder never re-probes).
 *
 * The kernel tells us directly: `kernel.unprivileged_userns_clone` (Debian and
 * derivatives) or `user.max_user_namespaces`. Both default to enabled and are
 * the switches distros flip to break unprivileged userns, so "not disabled"
 * is the signal — an unreadable or absent file means the kernel is not
 * restricting userns.
 *
 * Pure and dependency-free so it can be unit-tested without Electron.
 */

import fs from 'node:fs'

/** Value of a sysctl that means "unprivileged namespaces are switched off". */
const DISABLED = new Set(['0', 'n', 'no', 'off', 'false'])

/**
 * sysctls that gate unprivileged user namespaces, in the order they are
 * consulted. Debian-derived kernels gate the clone() syscall; upstream kernels
 * gate the namespace count instead.
 */
const USERNS_GATES: readonly string[] = [
  '/proc/sys/kernel/unprivileged_userns_clone',
  '/proc/sys/user/max_user_namespaces'
]

export function isUserNamespaceRestriction(readValue: (path: string) => string | null): boolean {
  for (const gate of USERNS_GATES) {
    let value: string | null

    try {
      value = readValue(gate)
    } catch {
      value = null
    }

    if (value === null) {
      // Not this kernel's gate (or not readable as this user) — keep looking.
      continue
    }

    const normalized = String(value).trim().toLowerCase()

    if (!normalized) {
      continue
    }

    // A positive count or a non-zero clone switch means userns is available.
    return DISABLED.has(normalized)
  }

  return false
}

export function detectUserNamespaceSandbox({
  platform = process.platform,
  readFileSync = fs.readFileSync
}: {
  platform?: NodeJS.Platform | string
  readFileSync?: (path: string, encoding: 'utf8') => string
} = {}): boolean {
  if (platform !== 'linux') {
    return false
  }

  return !isUserNamespaceRestriction(path => {
    try {
      return readFileSync(path, 'utf8')
    } catch {
      return null
    }
  })
}
