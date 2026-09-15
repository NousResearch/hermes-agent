import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { spawnSync } from 'node:child_process'
import { test } from 'vitest'

import { auditTree, archMatches, classifyHeader, findUnpackedDirs, isExemptPath, peArch } from '../scripts/audit-bundle-arch.mjs'

// ─── header builders: the smallest buffers each format needs ───────────────

function elfHeader(machine) {
  const buf = Buffer.alloc(64)
  buf.write('\x7fELF', 0, 'latin1')
  buf.writeUInt16LE(machine, 18)
  return buf
}

function machoThin(cputype, { swapped = false } = {}) {
  const buf = Buffer.alloc(64)
  if (swapped) {
    buf.writeUInt32BE(0xcffaedfe, 0) // magic as stored little-endian on disk
    buf.writeUInt32LE(cputype, 4)
  } else {
    buf.writeUInt32BE(0xfeedfacf, 0)
    buf.writeUInt32BE(cputype, 4)
  }
  return buf
}

function machoFat(cputypes) {
  const buf = Buffer.alloc(8 + cputypes.length * 20)
  buf.writeUInt32BE(0xcafebabe, 0)
  buf.writeUInt32BE(cputypes.length, 4)
  cputypes.forEach((t, i) => buf.writeUInt32BE(t, 8 + i * 20))
  return buf
}

function mzStub(peOffset) {
  const buf = Buffer.alloc(0x40)
  buf.write('MZ', 0, 'latin1')
  buf.writeUInt32LE(peOffset, 0x3c)
  return buf
}

// ─── classifyHeader ─────────────────────────────────────────────────

test('classifyHeader names the arch for each executable format', () => {
  assert.deepEqual(classifyHeader(elfHeader(0x3e)).arches, ['x64'])
  assert.deepEqual(classifyHeader(elfHeader(0xb7)).arches, ['arm64'])
  assert.deepEqual(classifyHeader(machoThin(0x0100000c)).arches, ['arm64'])
  assert.deepEqual(classifyHeader(machoThin(0x01000007, { swapped: true })).arches, ['x64'])
  assert.deepEqual(classifyHeader(machoFat([0x01000007, 0x0100000c])).arches, ['x64', 'arm64'])
})

test('classifyHeader defers PE to the offset named in the MZ stub', () => {
  const sniffed = classifyHeader(mzStub(0x180))
  assert.equal(sniffed.format, 'pe')
  assert.equal(sniffed.peHeaderOffset, 0x180)
  // The machine code itself resolves through peArch.
  assert.equal(peArch(0x8664), 'x64')
  assert.equal(peArch(0xaa64), 'arm64')
  assert.equal(peArch(0xa641), 'arm64ec')
  assert.match(peArch(0xbeef), /unknown/)
})

test('classifyHeader skips non-binaries, tiny files, and Java class files', () => {
  assert.equal(classifyHeader(Buffer.from('#!/bin/sh\necho hi\n')), null)
  assert.equal(classifyHeader(Buffer.from('MZ')), null) // too short to carry a PE offset
  assert.equal(classifyHeader(Buffer.alloc(0)), null)
  // Java .class: same magic as a fat Mach-O, giant "slice count" (version).
  const javaClass = Buffer.alloc(16)
  javaClass.writeUInt32BE(0xcafebabe, 0)
  javaClass.writeUInt32BE(65, 4)
  assert.equal(classifyHeader(javaClass), null)
})

// ─── archMatches ─────────────────────────────────────────────────

test('archMatches: exact match, universal slices, arm64ec, and rejections', () => {
  assert.ok(archMatches(['arm64'], 'arm64'))
  assert.ok(archMatches(['x64', 'arm64'], 'arm64')) // universal binary covers the target
  assert.ok(archMatches(['arm64ec'], 'arm64')) // arm64-ABI by definition
  assert.ok(!archMatches(['x64'], 'arm64')) // the shipped-x64-shim bug this audit exists for
  assert.ok(!archMatches(['arm64ec'], 'x64')) // arm64ec does not run on x64 hosts
  assert.ok(!archMatches(['unknown(0xbeef)'], 'x64')) // unclassifiable ships nowhere
})

// ─── findUnpackedDirs ─────────────────────────────────────────────────

test('findUnpackedDirs matches electron-builder output shapes only', () => {
  const dirs = findUnpackedDirs([
    'win-unpacked', 'win-arm64-unpacked', 'linux-unpacked', 'linux-arm64-unpacked',
    'mac', 'mac-arm64',
    'builder-debug.yml', 'Hermes-0.20.0.exe', 'latest.yml', '.icon-ico'
  ])
  assert.deepEqual(dirs, [
    'win-unpacked', 'win-arm64-unpacked', 'linux-unpacked', 'linux-arm64-unpacked',
    'mac', 'mac-arm64'
  ])
})

test.each([
  ['resources/agent-payload/tools/git-2.53.0-win32-x64/mingw64/bin/Avalonia.dll', true],
  ['resources\\agent-payload\\tools\\git-2.53.0-win32-x64\\mingw64\\libexec\\git-core\\GitHub.dll', true],
  ['resources\\agent-payload\\tools\\git-2.53.0-win32-x64\\usr\\libexec\\getprocaddr32.exe', true],
  ['resources/agent-payload/tools/git-2.54.0-win32-arm64/clangarm64/libexec/git-core/msalruntime.dll', true],
  ['resources/agent-payload/tools/git-2.53.0-linux-x64/bin/git', false],
  ['resources/agent-payload/tools/git-2.53.0-darwin-arm64/libexec/git-core/git-remote-https', false],
  ['resources/agent-payload/tools/python-3.11.16+20260814-linux-x64/lib/python3.11/site-packages/pip/_vendor/distlib/t32.exe', true],
  ['resources/agent-payload/tools/python-3.11.16+20260814-linux-x64/lib/python3.11/site-packages/setuptools/cli.exe', true],
  ['resources/agent-payload/venv/lib/python3.11/site-packages/setuptools/cli-32.exe', true],
  ['resources/agent-payload/venv/Lib/site-packages/setuptools/cli.exe', true],
  ['resources/agent-payload/venv/lib/python3.11/site-packages/discord/bin/libopus-0.x86.dll', true],
  ['resources/agent-payload/venv/lib/python3.11/site-packages/discord/bin/libopus-0.x64.dll', true],
  ['resources/agent-payload/venv/lib/python3.11/site-packages/pvporcupine/lib/mac/arm64/libpv_porcupine.dylib', true],
  ['resources/agent-payload/venv/lib/python3.11/site-packages/pvporcupine/lib/raspberry-pi/arm11/libpv_porcupine.so', true],
  ['resources/agent-payload/venv/lib/python3.11/site-packages/debugpy/_vendored/pydevd/pydevd_attach_to_process/attach_linux_amd64.so', true],
  ['resources/agent-payload/venv/lib/python3.11/site-packages/debugpy/_vendored/pydevd/pydevd_attach_to_process/inject_dll_x86.exe', true],
  ['resources/agent-payload/tools/agent-browser-0.35.1-win32-arm64/bin/agent-browser-win32-x64.exe', true],
  ['resources\\agent-payload\\tools\\agent-browser-0.26.0-win32-arm64\\bin\\agent-browser-win32-x64.exe', true],
  ['resources/agent-payload/tools/chromium-1208/chrome-win64/chrome.exe', true],
  ['resources\\agent-payload\\tools\\chromium-1208\\chrome-win64\\chrome.dll', true],
  ['resources/agent-payload/tools/chromium-1208/chrome-linux/chrome', false],
  ['resources/agent-payload/tools/chromium-1208/chrome-mac-arm64/chrome', false],
  ['resources/agent-payload/tools/chromium_headless_shell-1208/chrome-headless-shell-win64/chrome-headless-shell.exe', false],
  ['resources/agent-payload/tools/chromium-1208/chrome-headless-shell-win64/chrome-headless-shell.exe', false],
  ['resources/agent-payload/uv-cache/archive-v0/abc/setuptools/cli.exe', true],
  ['resources\\agent-payload\\uv-cache\\archive-v0\\abc\\discord\\bin\\libopus-0.x64.dll', true],
  ['resources/agent-payload/uv-cache/builds-v0/whatever/build.exe', true],
  ['resources/agent-payload/tools/something-1.0-win32-arm64/bin/thing.exe', false],
  ['resources/agent-payload/tools/uv-cache-1.0/tool.exe', false],
  ['resources/agent-payload/hermes-agent/something.exe', false],
])('foreign architecture exemption %s → %s', (file, exempt) => {
  assert.equal(isExemptPath(file), exempt)
})

test('tree and CLI audit real second-read PE bytes and fail closed on absent artifacts', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arch-tree-'))
  const cli = () => spawnSync(process.execPath, [path.resolve(import.meta.dirname, 'audit-bundle-arch.mjs'), '--arch=arm64', `--root=${root}`], { encoding: 'utf8' })
  try {
    assert.match(cli().stderr, /no unpacked app directory/)
    const app = path.join(root, 'win-arm64-unpacked')
    fs.mkdirSync(app)
    assert.match(cli().stderr, /no native binaries/)
    fs.writeFileSync(path.join(app, 'elf'), elfHeader(0xb7))
    fs.writeFileSync(path.join(app, 'macho'), machoThin(0x0100000c))
    const pe = Buffer.alloc(8198)
    mzStub(8192).copy(pe)
    pe.writeUInt32LE(0x4550, 8192)
    pe.writeUInt16LE(0xa641, 8196)
    fs.writeFileSync(path.join(app, 'arm64ec.exe'), pe)
    assert.deepEqual(auditTree(app, 'arm64'), { binaries: 3, exempted: 0, mismatches: [] })
    assert.equal(cli().status, 0)
    pe.writeUInt16LE(0x8664, 8196)
    const cache = path.join(app, 'resources/agent-payload/uv-cache/builds-v0')
    fs.mkdirSync(cache, { recursive: true })
    fs.writeFileSync(path.join(cache, 'cached.exe'), pe)
    fs.writeFileSync(path.join(app, 'wrong-arch.dat'), pe)
    assert.deepEqual(auditTree(app, 'arm64'), {
      binaries: 5, exempted: 1, mismatches: [{ file: 'wrong-arch.dat', format: 'pe', arches: ['x64'] }]
    })
    const failed = cli()
    assert.equal(failed.status, 1)
    assert.match(failed.stderr, /wrong-arch.dat/)
  } finally { fs.rmSync(root, { recursive: true, force: true }) }
})
