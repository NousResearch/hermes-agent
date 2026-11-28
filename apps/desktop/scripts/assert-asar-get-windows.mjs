// Packaged-bundle file-tree assertion (mega-PR Target 8).
//
// Reads a packaged Electron app.asar header (no unpacking) and asserts the
// get-windows tree the stager demands is present, helper binary included.
// Reader design verified by hand against two production macOS bundles (both
// parse to the identical six-entry tree with the 228,664-byte `main`):
// the header is a Chromium pickle — UInt32LE size candidates at offsets
// 4/8/12, JSON document at offset 16, nested {"files": {...}} objects.
import { readFileSync } from 'node:fs'

const GET_WINDOWS_REQUIRED = [
  'index.js',
  'package.json',
  'lib/linux.js',
  'lib/macos.js',
  'lib/windows.js',
  'main',
]

export function listAsarFiles(asarPath) {
  const raw = readFileSync(asarPath)
  const candidates = [raw.readUInt32LE(12), raw.readUInt32LE(8), raw.readUInt32LE(4)]
  let header = null
  for (const size of candidates) {
    try {
      header = JSON.parse(raw.subarray(16, 16 + size).toString('utf8'))
      break
    } catch {
      // Wrong size candidate for this archive; try the next.
    }
  }
  if (!header || typeof header !== 'object') throw new Error(`unreadable asar header: ${asarPath}`)
  const out = []
  const walk = (node, prefix) => {
    for (const [name, entry] of Object.entries(node.files ?? {})) {
      if (entry && typeof entry === 'object' && entry.files) walk(entry, `${prefix}${name}/`)
      else out.push(prefix + name)
    }
  }
  walk(header, '')
  return out
}

export function getWindowsFilesInAsar(asarPath) {
  return listAsarFiles(asarPath).filter(file => file.split('/').includes('get-windows'))
}

// Darwin-bundle check (not cross-platform): the stager demands the macOS
// helper binary `main` on darwin only — win32 ships a .node binding instead
// and Linux stages neither native payload (its loader shells out to xprop).
export function assertDarwinGetWindowsTree(asarPath) {
  const staged = getWindowsFilesInAsar(asarPath)
  if (staged.length === 0) {
    throw new Error(
      `packaged bundle has no get-windows tree: ${asarPath} — ` +
        'the stager demands it; ship no bundle without it'
    )
  }
  const leafNames = staged.map(file => file.slice(file.indexOf('get-windows/') + 'get-windows/'.length))
  const missing = GET_WINDOWS_REQUIRED.filter(name => !leafNames.includes(name))
  if (missing.length > 0) {
    throw new Error(`packaged get-windows tree is missing ${missing.join(', ')}: ${asarPath}`)
  }
  return staged
}
