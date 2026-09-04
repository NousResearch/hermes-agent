import assert from 'node:assert/strict'
import { test } from 'vitest'

import { archMatches, classifyHeader, findUnpackedDirs, isExemptPath, peArch } from '../scripts/audit-bundle-arch.mjs'

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

// ─── the git exemption is PortableGit-shaped, not git-shaped ───────────────

test('PortableGit internals stay exempt under the payload store', () => {
  // .NET assemblies report ia32 because they are format-neutral.
  for (const relPath of [
    'resources/agent-payload/tools/git-2.53.0-win32-x64/mingw64/bin/Avalonia.dll',
    'resources\\agent-payload\\tools\\git-2.53.0-win32-x64\\mingw64\\libexec\\git-core\\GitHub.dll',
    'resources\\agent-payload\\tools\\git-2.53.0-win32-x64\\usr\\libexec\\getprocaddr32.exe',
    'resources/agent-payload/tools/git-2.54.0-win32-arm64/clangarm64/libexec/git-core/msalruntime.dll'
  ]) {
    assert.equal(isExemptPath(relPath), true, relPath)
  }
})

test('a POSIX git store entry is audited', () => {
  // posix uses system git; a linux/darwin git store has no mingw64/usr/cmd
  // segment, so a wrong-arch binary there must fail the audit.
  for (const relPath of [
    'resources/agent-payload/tools/git-2.53.0-linux-x64/bin/git',
    'resources/agent-payload/tools/git-2.53.0-darwin-arm64/libexec/git-core/git-remote-https'
  ]) {
    assert.equal(isExemptPath(relPath), false, relPath)
  }
})

test('pip and setuptools launcher stubs are exempt in the store and the venv', () => {
  for (const relPath of [
    'resources/agent-payload/tools/python-3.11.16+20260814-linux-x64/lib/python3.11/site-packages/pip/_vendor/distlib/t32.exe',
    'resources/agent-payload/tools/python-3.11.16+20260814-linux-x64/lib/python3.11/site-packages/setuptools/cli.exe',
    'resources/agent-payload/venv/lib/python3.11/site-packages/setuptools/cli-32.exe',
    'resources/agent-payload/venv/Lib/site-packages/setuptools/cli.exe'
  ]) {
    assert.equal(isExemptPath(relPath), true, relPath)
  }
})

test('discord opus and pvporcupine multi-arch siblings are exempt', () => {
  for (const relPath of [
    'resources/agent-payload/venv/lib/python3.11/site-packages/discord/bin/libopus-0.x86.dll',
    'resources/agent-payload/venv/lib/python3.11/site-packages/discord/bin/libopus-0.x64.dll',
    'resources/agent-payload/venv/lib/python3.11/site-packages/pvporcupine/lib/mac/arm64/libpv_porcupine.dylib',
    'resources/agent-payload/venv/lib/python3.11/site-packages/pvporcupine/lib/raspberry-pi/arm11/libpv_porcupine.so',
    'resources/agent-payload/venv/lib/python3.11/site-packages/debugpy/_vendored/pydevd/pydevd_attach_to_process/attach_linux_amd64.so',
    'resources/agent-payload/venv/lib/python3.11/site-packages/debugpy/_vendored/pydevd/pydevd_attach_to_process/inject_dll_x86.exe'
  ]) {
    assert.equal(isExemptPath(relPath), true, relPath)
  }
})

test('the emulated x64 agent-browser exe is exempt in a win32-arm64 payload', () => {
  for (const relPath of [
    'resources/agent-payload/tools/agent-browser-0.35.1-win32-arm64/bin/agent-browser-win32-x64.exe',
    'resources\\agent-payload\\tools\\agent-browser-0.26.0-win32-arm64\\bin\\agent-browser-win32-x64.exe'
  ]) {
    assert.equal(isExemptPath(relPath), true, relPath)
  }
  // A non-chromium x64 binary in the payload store still fails: the
  // chromium exemption is scoped to the store entries' -win64 trees, not
  // to "any x64 exe under tools/".
  assert.equal(isExemptPath('resources/agent-payload/tools/something-1.0-win32-arm64/bin/thing.exe'), false)
})

test('the emulated x64 chromium trees are exempt in a win32-arm64 payload', () => {
  // PlaywrightBrowser._CFT maps win32-arm64 to the CfT win64 (x64) build;
  // store entries are named by playwright revision (dashes→underscores)
  // and the zip extracts in place, so the x64 trees sit under
  // chrome-win64 / chrome-headless-shell-win64 inside the entry.
  for (const relPath of [
    'resources/agent-payload/tools/chromium-1208/chrome-win64/chrome.exe',
    'resources\\agent-payload\\tools\\chromium-1208\\chrome-win64\\chrome.dll',
    'resources/agent-payload/tools/chromium_headless_shell-1208/chrome-headless-shell-win64/chrome-headless-shell.exe'
  ]) {
    assert.equal(isExemptPath(relPath), true, relPath)
  }
  // The linux/darwin chromium trees stay audited: no -win64 segment.
  assert.equal(isExemptPath('resources/agent-payload/tools/chromium-1208/chrome-linux/chrome'), false)
  assert.equal(isExemptPath('resources/agent-payload/tools/chromium-1208/chrome-mac-arm64/chrome'), false)
})

test('the uv wheel/build cache is exempt in the payload', () => {
  // pm bundle deliberately ships uv-cache/ for warm venv rebuilds; it
  // holds cached sdists/archives uv may have built for ANY arch (x64/ia32
  // PEs on an arm64 payload) — inert cache bytes, never loaded at runtime.
  for (const relPath of [
    'resources/agent-payload/uv-cache/archive-v0/abc/setuptools/cli.exe',
    'resources\\agent-payload\\uv-cache\\archive-v0\\abc\\discord\\bin\\libopus-0.x64.dll',
    'resources/agent-payload/uv-cache/builds-v0/whatever/build.exe'
  ]) {
    assert.equal(isExemptPath(relPath), true, relPath)
  }
  // Anything outside the cache stays audited.
  assert.equal(isExemptPath('resources/agent-payload/tools/uv-cache-1.0/tool.exe'), false)
})

test('a random payload binary is not exempt', () => {
  assert.equal(isExemptPath('resources/agent-payload/hermes-agent/something.exe'), false)
})
