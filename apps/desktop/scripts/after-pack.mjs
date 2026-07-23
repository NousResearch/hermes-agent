/**
 * after-pack.mjs — electron-builder afterPack hook.
 *
 * Stamps the Hermes icon + identity onto the packed Windows Hermes.exe via
 * rcedit (delegated to set-exe-identity.mjs). This runs for EVERY packed build
 * — first install, `hermes desktop`, the installer's --update rebuild, and a
 * dev's manual `npm run pack` — so the branded exe can never silently revert
 * to the stock "Electron" icon/name (the bug when the stamp lived only in
 * install.ps1, which the update path doesn't use).
 *
 * Also validates the Windows executable's PE architecture header to catch
 * wrong-arch builds at pack time instead of letting the user hit Windows'
 * "This app can't run on your computer" error at launch (#69179). The PE
 * machine field is read from the COFF header: 0x8664 = x64, 0x014c = x86,
 * 0xAA64 = ARM64. If the detected arch does not match the target arch, the
 * build is failed immediately with a clear diagnostic.
 *
 * Windows-only: rcedit edits PE resources, irrelevant on macOS/Linux where the
 * app identity comes from the bundle Info.plist / desktop entry. Best-effort:
 * a stamp failure must never fail an otherwise-good build (worst case is the
 * stock icon, not a broken app), so we log and resolve rather than throw.
 * Architecture validation, however, IS a hard failure — a wrong-arch binary
 * is a broken binary, and it is far better to fail here than ship junk.
 *
 * electron-builder passes a context with:
 *   - electronPlatformName: 'win32' | 'darwin' | 'linux'
 *   - appOutDir:            the unpacked app directory for this target
 *   - arch:                 Arch enum (0=ia32, 1=x64, 2=armv7l, 3=arm64, 4=universal)
 *   - packager.appInfo.productFilename: the exe basename (e.g. 'Hermes')
 */

import fs from 'node:fs'
import path from 'node:path'
import { Arch } from 'electron-builder'

import { stampExeIdentity } from './set-exe-identity.mjs'

// PE COFF machine type constants (IMAGE_FILE_MACHINE_*).
const PE_MACHINE_X64 = 0x8664
const PE_MACHINE_X86 = 0x014c
const PE_MACHINE_ARM64 = 0xaa64

const PE_MACHINE_NAME = {
  [PE_MACHINE_X64]: 'x64',
  [PE_MACHINE_X86]: 'x86 (ia32)',
  [PE_MACHINE_ARM64]: 'ARM64'
}

/**
 * Read the PE machine type from a Windows executable.
 *
 * Parses the DOS header → PE signature → COFF header to extract the 2-byte
 * Machine field. Returns the numeric machine type, or null if the file is
 * not a valid PE executable.
 */
function readPeMachineType(filePath) {
  let fd
  try {
    const buf = Buffer.alloc(512)
    fd = fs.openSync(filePath, 'r')
    const bytesRead = fs.readSync(fd, buf, 0, 512, 0)
    if (bytesRead < 512) {
      return null
    }

    // DOS header: e_lfanew is at offset 0x3C (4-byte little-endian)
    const peOffset = buf.readUInt32LE(0x3c)
    if (peOffset === 0 || peOffset > bytesRead - 4) {
      return null
    }

    // PE signature must be "PE\0\0"
    if (
      buf[peOffset] !== 0x50 || // 'P'
      buf[peOffset + 1] !== 0x45 || // 'E'
      buf[peOffset + 2] !== 0x00 ||
      buf[peOffset + 3] !== 0x00
    ) {
      return null
    }

    // COFF header starts right after the signature; Machine is the first 2 bytes
    const machineType = buf.readUInt16LE(peOffset + 4)
    return machineType
  } catch {
    return null
  } finally {
    if (fd !== undefined) {
      try {
        fs.closeSync(fd)
      } catch {
        /* ignore */
      }
    }
  }
}

/**
 * Map electron-builder's Arch enum to the PE machine type we expect.
 * Returns null for universal/unknown (no hard check possible).
 */
function expectedPeMachine(arch) {
  const archName = typeof arch === 'number' ? Arch[arch] : undefined

  switch (archName) {
    case 'x64':
      return PE_MACHINE_X64
    case 'ia32':
      return PE_MACHINE_X86
    case 'arm64':
      return PE_MACHINE_ARM64
    default:
      return null // universal or unspecified — can't validate
  }
}

export default async function afterPack(context) {
  if (context.electronPlatformName !== 'win32') {
    return
  }

  const productName = context.packager?.appInfo?.productFilename || 'Hermes'
  const exe = path.join(context.appOutDir, `${productName}.exe`)
  const desktopRoot = path.resolve(import.meta.dirname, '..')

  // ── Architecture validation (#69179) ──────────────────────────────
  const expected = expectedPeMachine(context.arch)
  if (expected !== null) {
    const actual = readPeMachineType(exe)
    const actualName = actual != null ? PE_MACHINE_NAME[actual] || `unknown (0x${actual.toString(16)})` : 'not-a-valid-PE'

    if (actual === null) {
      // Hard fail: the exe is not a valid PE — it will not launch on any Windows.
      throw new Error(
        `[after-pack] ${exe} is not a valid PE executable (detected arch: ${actualName}). ` +
          `The built Hermes.exe will NOT run on Windows. The Electron binary download may be corrupted — ` +
          `clear the Electron cache and retry.`
      )
    }

    if (actual !== expected) {
      const expectedName = PE_MACHINE_NAME[expected]
      throw new Error(
        `[after-pack] Architecture mismatch: built ${actualName}, expected ${expectedName}. ` +
          `Users will see "This app can't run on your computer" when launching Hermes.exe. ` +
          `Target arch: ${Arch[context.arch]}. Check the Electron download cache and electron-builder configuration.`
      )
    }

    console.log(`[after-pack] ✓ Hermes.exe PE arch validated: ${actualName}`)
  }

  // ── Icon / identity stamp (best-effort) ────────────────────────────
  try {
    await stampExeIdentity(exe, desktopRoot)
  } catch (err) {
    // Never fail the build over a cosmetic stamp.
    console.warn(`[after-pack] exe identity stamp failed (${err.message}); Hermes.exe keeps the stock Electron icon`)
  }
}
