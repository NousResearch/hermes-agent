// Target 8: packaged-bundle file-tree test.
//
// Builds minimal synthetic app.asar buffers (16-byte pickle prelude + JSON
// header, the layout verified against production bundles) and asserts the
// file-tree check passes a bundle carrying the demanded tree and fails a
// bundle without it — the guard that would have held the November 6
// failure to a named, pre-packaging error instead of a shipped gap.
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, expect, test } from 'vitest'
import { assertGetWindowsTree, getWindowsFilesInAsar, listAsarFiles } from './assert-asar-get-windows.mjs'

const roots = []
afterEach(() => { for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true }) })

function fileNode(size = 10) {
  return { size, executable: false }
}

// Minimal asar: 16-byte prelude (width marker, then three size candidates
// with the true JSON length at offset 12, matching production archives),
// then the header JSON document.
function asarBuffer(header) {
  const json = Buffer.from(JSON.stringify(header), 'utf8')
  const prelude = Buffer.alloc(16)
  prelude.writeUInt32LE(4, 0)
  prelude.writeUInt32LE(json.length + 8, 4)
  prelude.writeUInt32LE(json.length + 4, 8)
  prelude.writeUInt32LE(json.length, 12)
  return Buffer.concat([prelude, json])
}

function treeFiles() {
  return {
    'index.js': fileNode(),
    'package.json': fileNode(),
    lib: { files: { 'linux.js': fileNode(), 'macos.js': fileNode(), 'windows.js': fileNode() } },
    main: { size: 228664, executable: true },
  }
}

function asarFile(withTree) {
  const dir = mkdtempSync(join(tmpdir(), 'asar-tree-'))
  roots.push(dir)
  const tree = withTree
    ? { dist: { files: { node_modules: { files: { 'get-windows': { files: treeFiles() } } } } } }
    : { dist: { files: { node_modules: { files: {} } } } }
  const path = join(dir, 'app.asar')
  writeFileSync(path, asarBuffer({ files: tree }))
  return path
}

test('bundled tree with helper passes', () => {
  const staged = assertGetWindowsTree(asarFile(true))
  expect(staged).toHaveLength(6)
  expect(staged.some(file => file.endsWith('/main'))).toBe(true)
})

test('bundle without the tree fails with a shippable message', () => {
  expect(() => assertGetWindowsTree(asarFile(false))).toThrow(/no get-windows tree/)
})

test('partial tree names the missing entries', () => {
  const dir = mkdtempSync(join(tmpdir(), 'asar-tree-'))
  roots.push(dir)
  const partial = treeFiles()
  delete partial.main
  const header = { files: { dist: { files: { node_modules: { files: { 'get-windows': { files: partial } } } } } } }
  const path = join(dir, 'app.asar')
  writeFileSync(path, asarBuffer(header))
  expect(() => assertGetWindowsTree(path)).toThrow(/missing main/)
})

test('reader lists nested files in order', () => {
  expect(getWindowsFilesInAsar(asarFile(true)).sort()).toEqual([
    'dist/node_modules/get-windows/index.js',
    'dist/node_modules/get-windows/lib/linux.js',
    'dist/node_modules/get-windows/lib/macos.js',
    'dist/node_modules/get-windows/lib/windows.js',
    'dist/node_modules/get-windows/main',
    'dist/node_modules/get-windows/package.json',
  ])
  expect(listAsarFiles(asarFile(false))).toEqual([])
})
