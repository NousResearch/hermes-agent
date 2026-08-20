#!/usr/bin/env node

import { closeSync, existsSync, openSync, readFileSync, readSync } from 'node:fs'
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path'
import { spawnSync } from 'node:child_process'

function usage(message = '') {
  if (message) console.error(`run-node-package-bin: ${message}`)
  console.error('usage: node scripts/run-node-package-bin.mjs <package> [--bin <name>] [--cwd <dir>] -- [args...]')
  process.exit(2)
}

const input = process.argv.slice(2)
if (input.length === 0 || input[0].startsWith('-')) usage('package name is required')

const packageName = input.shift()
let binName = null
let cwd = process.cwd()
const childArgs = []
let afterSeparator = false
while (input.length) {
  const token = input.shift()
  if (afterSeparator) {
    childArgs.push(token)
    continue
  }
  if (token === '--') {
    afterSeparator = true
    continue
  }
  if (token === '--bin') {
    if (!input.length) usage('--bin requires a value')
    binName = input.shift()
    continue
  }
  if (token === '--cwd') {
    if (!input.length) usage('--cwd requires a value')
    const value = input.shift()
    cwd = isAbsolute(value) ? value : resolve(process.cwd(), value)
    continue
  }
  usage(`unknown option before --: ${token}`)
}

function packageJsonFrom(start) {
  let current = resolve(start)
  const packageParts = packageName.split('/').filter(Boolean)
  if (packageParts.length === 0 || packageParts.some(part => part === '.' || part === '..')) {
    usage(`invalid package name: ${packageName}`)
  }

  while (true) {
    const candidate = join(current, 'node_modules', ...packageParts, 'package.json')
    if (existsSync(candidate)) return candidate
    const parent = dirname(current)
    if (parent === current) break
    current = parent
  }
  return null
}

const packageJson = packageJsonFrom(cwd)
if (!packageJson) {
  console.error(`run-node-package-bin: ${packageName} is not installed from ${cwd} or any ancestor workspace`)
  process.exit(127)
}

let metadata
try {
  metadata = JSON.parse(readFileSync(packageJson, 'utf8'))
} catch (error) {
  console.error(`run-node-package-bin: cannot read ${packageJson}: ${error.message}`)
  process.exit(1)
}

let binRelative = null
if (typeof metadata.bin === 'string') {
  binRelative = metadata.bin
} else if (metadata.bin && typeof metadata.bin === 'object') {
  const defaultName = packageName.split('/').pop()
  const selected = binName || (Object.hasOwn(metadata.bin, defaultName) ? defaultName : null)
  if (selected && typeof metadata.bin[selected] === 'string') {
    binRelative = metadata.bin[selected]
  } else if (!binName && Object.keys(metadata.bin).length === 1) {
    binRelative = Object.values(metadata.bin)[0]
  }
}

if (!binRelative) {
  console.error(
    `run-node-package-bin: ${packageName} has no ${binName ? JSON.stringify(binName) : 'unambiguous'} executable in package.json`
  )
  process.exit(127)
}

const packageDir = dirname(packageJson)
const executable = resolve(packageDir, binRelative)
const rel = relative(packageDir, executable)
if (rel.startsWith(`..${sep}`) || rel === '..' || isAbsolute(rel) || !existsSync(executable)) {
  console.error(`run-node-package-bin: invalid or missing executable ${binRelative} for ${packageName}`)
  process.exit(127)
}

function isNativeExecutable(path) {
  const header = Buffer.alloc(4)
  let fd
  try {
    fd = openSync(path, 'r')
    const bytes = readSync(fd, header, 0, header.length, 0)
    if (bytes < 2) return false
  } catch {
    return false
  } finally {
    if (fd !== undefined) closeSync(fd)
  }

  if (header[0] === 0x7f && header.subarray(1, 4).toString('ascii') === 'ELF') return true
  if (header[0] === 0x4d && header[1] === 0x5a) return true // PE/COFF (MZ)

  const magic = header.toString('hex')
  return new Set([
    'feedface', // Mach-O 32-bit
    'cefaedfe', // Mach-O 32-bit reversed
    'feedfacf', // Mach-O 64-bit
    'cffaedfe', // Mach-O 64-bit reversed
    'cafebabe', // universal/fat Mach-O
    'bebafeca' // universal/fat Mach-O reversed
  ]).has(magic)
}

// Most npm bins are JavaScript files and need the current Node runtime. A few
// packages (notably esbuild after its postinstall) replace that bin with a
// platform-native executable. Feeding an ELF/PE/Mach-O binary to Node fails
// with SyntaxError, so execute native bins directly while keeping JS bins on
// the exact Node process that invoked this resolver.
const native = isNativeExecutable(executable)
const result = spawnSync(native ? executable : process.execPath, native ? childArgs : [executable, ...childArgs], {
  cwd,
  env: process.env,
  stdio: 'inherit'
})
if (result.error) {
  console.error(`run-node-package-bin: failed to launch ${packageName}: ${result.error.message}`)
  process.exit(1)
}
process.exit(result.status ?? 1)
