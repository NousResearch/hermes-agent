import { closeSync, openSync, readSync } from 'node:fs'

const CPU_TYPE_X86_64 = 0x01000007
const CPU_TYPE_ARM64 = 0x0100000c
const FAT_MAGIC = 0xcafebabe
const FAT_CIGAM = 0xbebafeca
const FAT_MAGIC_64 = 0xcafebabf
const FAT_CIGAM_64 = 0xbfbafeca
const MH_MAGIC_64 = 0xfeedfacf
const MH_CIGAM_64 = 0xcffaedfe

function cpuTypeToArch(cpuType) {
  if (cpuType === CPU_TYPE_X86_64) return 'x64'
  if (cpuType === CPU_TYPE_ARM64) return 'arm64'
  return null
}

export function classifyMachOArchitectures(filePath) {
  let descriptor
  let buffer
  try {
    descriptor = openSync(filePath, 'r')
    buffer = Buffer.alloc(4096)
    const bytesRead = readSync(descriptor, buffer, 0, buffer.length, 0)
    buffer = buffer.subarray(0, bytesRead)
  } catch {
    return null
  } finally {
    if (descriptor !== undefined) closeSync(descriptor)
  }
  if (buffer.length < 8) return null

  const magic = buffer.readUInt32BE(0)
  if (magic === MH_MAGIC_64 || magic === MH_CIGAM_64) {
    const littleEndian = magic === MH_CIGAM_64
    const cpuType = littleEndian ? buffer.readUInt32LE(4) : buffer.readUInt32BE(4)
    const arch = cpuTypeToArch(cpuType)
    return arch ? new Set([arch]) : null
  }

  const fat = magic === FAT_MAGIC || magic === FAT_CIGAM || magic === FAT_MAGIC_64 || magic === FAT_CIGAM_64
  if (!fat) return null

  const littleEndian = magic === FAT_CIGAM || magic === FAT_CIGAM_64
  const is64 = magic === FAT_MAGIC_64 || magic === FAT_CIGAM_64
  const readUInt32 = littleEndian ? buffer.readUInt32LE.bind(buffer) : buffer.readUInt32BE.bind(buffer)
  const count = readUInt32(4)
  const entrySize = is64 ? 32 : 20
  const arches = new Set()
  for (let index = 0; index < count; index += 1) {
    const offset = 8 + index * entrySize
    if (offset + entrySize > buffer.length) return null
    const arch = cpuTypeToArch(readUInt32(offset))
    if (arch) arches.add(arch)
  }
  return arches.size > 0 ? arches : null
}

export function electronBuilderArchFlag(architectures) {
  if (!architectures || architectures.size === 0) return null
  if (architectures.size === 1) return `--${[...architectures][0]}`
  if (architectures.has('arm64') && architectures.has('x64') && architectures.size === 2) return '--universal'
  return null
}

export function hasExplicitArchFlag(args) {
  return args.some((arg) =>
    ['--ia32', '--x64', '--armv7l', '--arm64', '--universal'].includes(arg) ||
    arg === '--arch' ||
    arg.startsWith('--arch=')
  )
}

export function archNameFromBuilderValue(value) {
  return ['ia32', 'x64', 'armv7l', 'arm64', 'universal'][value] ?? null
}

export function assertMachOArchitecture(filePath, expectedArch) {
  const architectures = classifyMachOArchitectures(filePath)
  if (!architectures) {
    throw new Error(`cannot classify packed Electron binary architecture: ${filePath}`)
  }
  if (expectedArch === 'universal') {
    if (!(architectures.has('arm64') && architectures.has('x64'))) {
      throw new Error(`packed Electron architecture mismatch: expected universal, got ${[...architectures].join('+')}`)
    }
    return
  }
  if (!architectures.has(expectedArch)) {
    throw new Error(`packed Electron architecture mismatch: expected ${expectedArch}, got ${[...architectures].join('+')}`)
  }
}
