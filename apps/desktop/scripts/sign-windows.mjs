#!/usr/bin/env node
// sign-windows.mjs — electron-builder `customSign` hook for Windows Authenticode
// signing.  Bypasses electron-builder's built-in winCodeSign path (which fetches
// winCodeSign-2.6.0.7z and crashes 7-Zip on non-admin Windows because the
// archive contains macOS symlinks that require SeCreateSymbolicLinkPrivilege).
//
// HOW IT WORKS
// ------------
// electron-builder calls this script with a `configuration` object when
// `build.win.customSign` points here.  We call `signtool.exe` from the
// Windows SDK directly — no downloaded helper, no symlink extraction.
//
// CERT CONFIGURATION (CI)
// -----------------------
// Set these environment variables in your signing environment:
//
//   WIN_CSC_LINK           — path to a PFX file, or a base-64-encoded PFX
//   WIN_CSC_KEY_PASSWORD   — PFX password
//   SIGNTOOL_PATH          — (optional) full path to signtool.exe if it is not
//                            on PATH or under the default Windows SDK locations
//
// When none of these are set this hook is a no-op, so local developer builds
// are unchanged (no cert = no signing, same as before).
//
// LOCAL BUILDS
// ------------
// No cert vars → hook exits 0 immediately.  Hermes.exe is stamped with rcedit
// (icon + PE metadata) by afterPack / set-exe-identity.mjs as always, but
// carries no Authenticode signature.  Windows Smart App Control will block
// unsigned executables on machines with SAC in enforcement mode (#70544).
//
// REFERENCES
// ----------
// electron-builder customSign: https://www.electron.build/code-signing
// signtool docs: https://learn.microsoft.com/en-us/windows/win32/seccrypto/signtool

import { execFileSync } from 'node:child_process'
import { existsSync, writeFileSync, unlinkSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { randomBytes } from 'node:crypto'

// Well-known Windows SDK signtool locations (newest SDKs first).
const SDK_SIGNTOOL_CANDIDATES = [
  'C:\\Program Files (x86)\\Windows Kits\\10\\bin\\10.0.26100.0\\x64\\signtool.exe',
  'C:\\Program Files (x86)\\Windows Kits\\10\\bin\\10.0.22621.0\\x64\\signtool.exe',
  'C:\\Program Files (x86)\\Windows Kits\\10\\bin\\10.0.19041.0\\x64\\signtool.exe',
  'C:\\Program Files (x86)\\Windows Kits\\10\\bin\\x64\\signtool.exe',
]

function resolveSigntool() {
  if (process.env.SIGNTOOL_PATH) {
    return process.env.SIGNTOOL_PATH
  }

  for (const candidate of SDK_SIGNTOOL_CANDIDATES) {
    if (existsSync(candidate)) {
      return candidate
    }
  }

  // Last resort: rely on PATH.
  return 'signtool.exe'
}

/**
 * electron-builder `customSign` entry point.
 * @param {object} configuration  — { path: string, hash: string, isNest: boolean }
 */
export default async function signWindows(configuration) {
  const pfxSource = process.env.WIN_CSC_LINK
  const pfxPassword = process.env.WIN_CSC_KEY_PASSWORD

  if (!pfxSource) {
    // No cert configured — local build.  Skip signing silently.
    console.log('[sign-windows] No WIN_CSC_LINK set — skipping Authenticode signing (local build).')
    return
  }

  const filePath = configuration.path
  const signtool = resolveSigntool()

  console.log(`[sign-windows] Signing: ${filePath}`)
  console.log(`[sign-windows] signtool: ${signtool}`)

  // Decode base-64 PFX if WIN_CSC_LINK is not a file path.
  let pfxPath = pfxSource
  let tempPfx = null

  if (!existsSync(pfxSource)) {
    // Treat as base-64 encoded PFX (common in CI secret stores).
    const pfxBuf = Buffer.from(pfxSource, 'base64')
    tempPfx = join(tmpdir(), `hermes-sign-${randomBytes(8).toString('hex')}.pfx`)
    writeFileSync(tempPfx, pfxBuf)
    pfxPath = tempPfx
    console.log('[sign-windows] Decoded base-64 PFX to temporary file.')
  }

  try {
    const args = [
      'sign',
      '/fd', 'sha256',
      '/td', 'sha256',
      '/tr', 'http://timestamp.digicert.com',
      '/f', pfxPath,
    ]

    if (pfxPassword) {
      args.push('/p', pfxPassword)
    }

    args.push(filePath)

    execFileSync(signtool, args, { stdio: 'inherit', windowsHide: true })
    console.log(`[sign-windows] ✓ Signed: ${filePath}`)
  } finally {
    if (tempPfx) {
      try { unlinkSync(tempPfx) } catch { /* best-effort cleanup */ }
    }
  }
}
