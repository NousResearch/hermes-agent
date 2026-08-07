import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import {
  buildDesktopUpdateArgs,
  buildMacPublishScript,
  resolveStagedMacApp,
  validateMacUpdatePaths
} from './update-macos-publish'

function makeExecutable(file: string, body: string) {
  fs.writeFileSync(file, `#!/bin/bash\nset -eu\n${body}\n`, { mode: 0o755 })
}

function runPublisher(root: string, script: string) {
  const file = path.join(root, 'publisher.sh')
  fs.writeFileSync(file, script, { mode: 0o755 })
  return spawnSync('/bin/bash', [file])
}

function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-mac-publish-test-'))
  const stage = path.join(root, 'stage')
  const candidate = path.join(stage, 'mac-arm64', 'Hermes.app')
  const target = path.join(root, 'Applications', 'Hermes.app')
  for (const app of [candidate, target]) {
    const executable = path.join(app, 'Contents', 'MacOS', 'Hermes')
    fs.mkdirSync(path.dirname(executable), { recursive: true })
    fs.writeFileSync(executable, app === candidate ? 'new' : 'old', { mode: 0o755 })
  }
  return { root, stage, candidate, target }
}

test('Desktop update argv suppresses autonomous build and stages only macOS packaging', () => {
  assert.deepEqual(buildDesktopUpdateArgs(['--branch', 'main']), [
    'update',
    '--yes',
    '--skip-desktop-build',
    '--branch',
    'main'
  ])
  assert.deepEqual(buildDesktopUpdateArgs([], { updateOnly: true }), ['update', '--yes', '--skip-desktop-build'])
  assert.deepEqual(buildDesktopUpdateArgs([], { stageRoot: '/tmp/txn' }), [
    'desktop',
    '--build-only',
    '--output-dir',
    '/tmp/txn'
  ])
  assert.deepEqual(buildDesktopUpdateArgs([]), ['update', '--yes', '--skip-desktop-build'])
})

test('staged candidate resolution searches only the explicit output root', () => {
  const { root, stage, candidate } = fixture()
  try {
    assert.equal(resolveStagedMacApp(stage, fs.existsSync), candidate)
    assert.equal(resolveStagedMacApp(path.join(root, 'other'), fs.existsSync), null)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('stage, candidate, target, and rollback paths must be non-aliasing', () => {
  const { root, stage, candidate, target } = fixture()
  try {
    const safe = validateMacUpdatePaths({ stageRoot: stage, candidateApp: candidate, targetApp: target })
    assert.equal(safe.candidateApp, fs.realpathSync(candidate))
    assert.throws(
      () => validateMacUpdatePaths({ stageRoot: stage, candidateApp: candidate, targetApp: candidate }),
      /distinct/
    )
    assert.throws(
      () => validateMacUpdatePaths({ stageRoot: stage, candidateApp: target, targetApp: target }),
      /inside staging root/
    )
    assert.throws(
      () => validateMacUpdatePaths({ stageRoot: root, candidateApp: candidate, targetApp: target }),
      /must not contain target/
    )
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('publisher makes no target or sibling write while old app pid is alive', () => {
  const { root, stage, candidate, target } = fixture()
  const oldContents = fs.readFileSync(path.join(target, 'Contents', 'MacOS', 'Hermes'), 'utf8')
  try {
    const script = buildMacPublishScript({
      pid: process.pid,
      stageRoot: stage,
      candidateApp: candidate,
      targetApp: target,
      waitIterations: 1,
      waitSeconds: 0
    })
    const result = runPublisher(root, script)
    assert.notEqual(result.status, 0)
    assert.equal(fs.readFileSync(path.join(target, 'Contents', 'MacOS', 'Hermes'), 'utf8'), oldContents)
    assert.equal(fs.existsSync(`${target}.hermes-update-new`), false)
    assert.equal(fs.existsSync(`${target}.hermes-update-old`), false)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('publisher swaps staged app after exit, relaunches exact target, and safely cleans stage', () => {
  const { root, stage, candidate, target } = fixture()
  const copy = path.join(root, 'copy.sh')
  const open = path.join(root, 'open.sh')
  const xattr = path.join(root, 'xattr.sh')
  const startupProbe = path.join(root, 'startup-probe.sh')
  const opened = path.join(root, 'opened.txt')
  makeExecutable(copy, '/bin/cp -R "$1" "$2"')
  makeExecutable(open, `printf '%s' "$1" > ${JSON.stringify(opened)}`)
  makeExecutable(xattr, 'exit 0')
  makeExecutable(startupProbe, 'exit 0')
  try {
    const script = buildMacPublishScript({
      pid: 99999999,
      stageRoot: stage,
      candidateApp: candidate,
      targetApp: target,
      tools: { copy, open, xattr, startupProbe }
    })
    const result = runPublisher(root, script)
    assert.equal(result.status, 0, result.stderr.toString())
    assert.equal(fs.readFileSync(path.join(target, 'Contents', 'MacOS', 'Hermes'), 'utf8'), 'new')
    assert.equal(fs.readFileSync(opened, 'utf8'), fs.realpathSync(target))
    assert.equal(fs.existsSync(`${target}.hermes-update-old`), false)
    assert.equal(fs.existsSync(stage), false)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('publisher copy failure is fail-closed and preserves old target', () => {
  const { root, stage, candidate, target } = fixture()
  const copy = path.join(root, 'copy-fail.sh')
  makeExecutable(copy, 'exit 7')
  try {
    const script = buildMacPublishScript({
      pid: 99999999,
      stageRoot: stage,
      candidateApp: candidate,
      targetApp: target,
      tools: { copy }
    })
    const result = runPublisher(root, script)
    assert.notEqual(result.status, 0)
    assert.equal(fs.readFileSync(path.join(target, 'Contents', 'MacOS', 'Hermes'), 'utf8'), 'old')
    assert.equal(fs.existsSync(`${target}.hermes-update-old`), false)
    assert.equal(fs.existsSync(stage), true)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('publisher rolls back when final target validation fails', () => {
  const { root, stage, candidate, target } = fixture()
  const copy = path.join(root, 'copy.sh')
  const validate = path.join(root, 'validate.sh')
  const counter = path.join(root, 'validate-count')
  makeExecutable(copy, '/bin/cp -R "$1" "$2"')
  makeExecutable(
    validate,
    `n=0; [ -f ${JSON.stringify(counter)} ] && n=$(/bin/cat ${JSON.stringify(counter)}); n=$((n+1)); printf '%s' "$n" > ${JSON.stringify(counter)}; [ "$n" -eq 1 ]`
  )
  try {
    const script = buildMacPublishScript({
      pid: 99999999,
      stageRoot: stage,
      candidateApp: candidate,
      targetApp: target,
      tools: { copy, validate }
    })
    const result = runPublisher(root, script)
    assert.notEqual(result.status, 0)
    assert.equal(fs.readFileSync(path.join(target, 'Contents', 'MacOS', 'Hermes'), 'utf8'), 'old')
    assert.equal(fs.existsSync(`${target}.hermes-update-old`), false)
    assert.equal(fs.existsSync(stage), true)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('publisher prefers direct executable launch when open does not yield a process', () => {
  const { root, stage, candidate, target } = fixture()
  const copy = path.join(root, 'copy.sh')
  const open = path.join(root, 'open.sh')
  const xattr = path.join(root, 'xattr.sh')
  const startupProbe = path.join(root, 'startup-probe.sh')
  const opened = path.join(root, 'opened.txt')
  const launched = path.join(root, 'launched.txt')
  makeExecutable(copy, '/bin/cp -R "$1" "$2"')
  makeExecutable(open, `printf '%s' "$1" > ${JSON.stringify(opened)}; exit 0`)
  makeExecutable(xattr, 'exit 0')
  // Become healthy only after the target EXE has been executed (direct launch).
  makeExecutable(
    startupProbe,
    `if [ -f ${JSON.stringify(launched)} ]; then exit 0; fi; exit 1`
  )
  const exe = path.join(candidate, 'Contents', 'MacOS', 'Hermes')
  // Must remain a long-lived probe marker writer; do not use set -e body failures.
  fs.writeFileSync(
    exe,
    `#!/bin/bash\nprintf 'launched' > ${JSON.stringify(launched)}\n# keep a background heart so pgrep-like probes that exec this binary are ok\nexit 0\n`,
    { mode: 0o755 }
  )
  try {
    const script = buildMacPublishScript({
      pid: 99999999,
      stageRoot: stage,
      candidateApp: candidate,
      targetApp: target,
      tools: { copy, open, xattr, startupProbe }
    })
    const result = runPublisher(root, script)
    assert.equal(
      result.status,
      0,
      `status=${result.status} out=${result.stdout?.toString?.() || ''} err=${result.stderr?.toString?.() || ''}`
    )
    assert.equal(fs.existsSync(launched), true)
    assert.equal(fs.existsSync(`${target}.hermes-update-old`), false)
    assert.equal(fs.existsSync(stage), false)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
}, 20_000)
