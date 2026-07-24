/**
 * pe-arch.mjs — read a Windows PE file's COFF Machine field.
 *
 * Used to stop packaging / launching a Hermes.exe (or electron.exe) whose
 * CPU architecture does not match the host. Windows surfaces that mismatch
 * as the modal "This app can't run on your PC" / 「此应用无法在你的电脑上运行」
 * (#69179) — there is no useful stderr, so we have to catch it before launch.
 *
 * PE layout (little-endian):
 *   offset 0x00: "MZ"
 *   offset 0x3C: DWORD e_lfanew → PE header
 *   PE+0:        "PE\0\0"
 *   PE+4:        WORD Machine
 *
 * Machine values we care about:
 *   0x014c IMAGE_FILE_MACHINE_I386
 *   0x8664 IMAGE_FILE_MACHINE_AMD64
 *   0xAA64 IMAGE_FILE_MACHINE_ARM64
 */

import fs from 'node:fs'

export const PE_MACHINE = Object.freeze({
  ia32: 0x014c,
  x64: 0x8664,
  arm64: 0xaa64
})

const MACHINE_TO_ARCH = Object.freeze({
  [PE_MACHINE.ia32]: 'ia32',
  [PE_MACHINE.x64]: 'x64',
  [PE_MACHINE.arm64]: 'arm64'
})

/**
 * Map electron-builder / Node arch names onto the PE Machine vocabulary.
 * Accepts Arch enum strings (`x64`, `arm64`, `ia32`) and Node's `process.arch`.
 */
export function normalizeCpuArch(arch) {
  if (!arch || typeof arch !== 'string') return null
  const a = arch.toLowerCase()
  if (a === 'x64' || a === 'amd64' || a === 'x86_64') return 'x64'
  if (a === 'arm64' || a === 'aarch64') return 'arm64'
  if (a === 'ia32' || a === 'x86' || a === 'i386' || a === 'i686') return 'ia32'
  return null
}

/**
 * Read the COFF Machine of a PE file and return `'x64' | 'ia32' | 'arm64'`,
 * or `null` when the file is missing / not a PE / uses an unknown Machine.
 */
export function readPeArch(filePath) {
  let fd
  try {
    fd = fs.openSync(filePath, 'r')
  } catch {
    return null
  }
  try {
    const header = Buffer.alloc(0x40)
    if (fs.readSync(fd, header, 0, 0x40, 0) < 0x40) return null
    if (header[0] !== 0x4d || header[1] !== 0x5a) return null

    const peOffset = header.readUInt32LE(0x3c)
    // Guard absurd e_lfanew values (corrupt / truncated files).
    if (peOffset < 0x40 || peOffset > 1024 * 1024) return null

    const pe = Buffer.alloc(6)
    if (fs.readSync(fd, pe, 0, 6, peOffset) < 6) return null
    if (pe[0] !== 0x50 || pe[1] !== 0x45 || pe[2] !== 0x00 || pe[3] !== 0x00) {
      return null
    }
    const machine = pe.readUInt16LE(4)
    return MACHINE_TO_ARCH[machine] ?? null
  } finally {
    try {
      fs.closeSync(fd)
    } catch {
      /* ignore */
    }
  }
}

/**
 * True when `filePath` is a PE whose Machine matches `expectedArch`.
 * Returns false when the PE is missing, unreadable, or mismatched.
 * Non-Windows callers should not use this as a hard gate.
 */
export function peArchMatches(filePath, expectedArch) {
  const want = normalizeCpuArch(expectedArch)
  if (!want) return false
  const got = readPeArch(filePath)
  return got === want
}

/**
 * Build a minimal (invalid-as-image, valid-as-header) PE buffer for tests.
 */
export function makeMinimalPeBuffer(arch) {
  const normalized = normalizeCpuArch(arch)
  const machine = normalized ? PE_MACHINE[normalized] : PE_MACHINE.x64
  const peOffset = 0x40
  const buf = Buffer.alloc(peOffset + 6, 0)
  buf[0] = 0x4d
  buf[1] = 0x5a
  buf.writeUInt32LE(peOffset, 0x3c)
  buf[peOffset] = 0x50
  buf[peOffset + 1] = 0x45
  buf[peOffset + 2] = 0x00
  buf[peOffset + 3] = 0x00
  buf.writeUInt16LE(machine, peOffset + 4)
  return buf
}
