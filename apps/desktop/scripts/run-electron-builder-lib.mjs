// Pure helpers for run-electron-builder.mjs — kept import-safe for unit tests
// (the CLI entry spawns electron-builder on load).

import fs from "node:fs"

import { normalizeCpuArch, readPeArch } from "./pe-arch.mjs"

/**
 * Best-effort target arch for this electron-builder invocation.
 * Prefer an explicit `--x64` / `--arm64` / `--ia32` flag; otherwise host arch.
 */
export function resolveBuilderTargetArch(argv = [], hostArch = "x64") {
  for (const arg of argv) {
    if (arg === "--x64" || arg === "--arm64" || arg === "--ia32") {
      return normalizeCpuArch(arg.slice(2))
    }
    if (arg.startsWith("--arch=")) {
      return normalizeCpuArch(arg.slice("--arch=".length))
    }
  }
  for (let i = 0; i < argv.length - 1; i++) {
    if (argv[i] === "--arch") {
      return normalizeCpuArch(argv[i + 1])
    }
  }
  return normalizeCpuArch(hostArch)
}

/**
 * Decide whether the local electron dist is safe to pass as electronDist.
 */
export function shouldReuseElectronDist({
  platform = "win32",
  distDir = null,
  binaryPath = null,
  targetArch = null,
  hostArch = "x64",
  peArchReader = readPeArch,
  existsSync = fs.existsSync,
} = {}) {
  if (!distDir || !binaryPath || !existsSync(binaryPath)) {
    return { reuse: false, reason: "missing" }
  }
  // PE Machine is a Windows concept. On other hosts, existence is enough
  // (cross-OS reuse is a separate concern — see #66345).
  if (platform !== "win32") {
    return { reuse: true, reason: "non-windows" }
  }
  const want = normalizeCpuArch(targetArch) || normalizeCpuArch(hostArch)
  if (!want) {
    return { reuse: true, reason: "unknown-arch" }
  }
  const got = peArchReader(binaryPath)
  if (!got) {
    return { reuse: false, reason: "unreadable-pe", got, want }
  }
  if (got !== want) {
    return { reuse: false, reason: "arch-mismatch", got, want }
  }
  return { reuse: true, reason: "arch-match", got, want }
}
