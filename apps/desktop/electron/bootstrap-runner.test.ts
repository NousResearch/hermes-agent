import assert from 'node:assert/strict'
import fs from 'node:fs'
import http from 'node:http'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import {
  buildPinArgs,
  buildPosixPinArgs,
  cachedScriptPath,
  cleanInstallerLogLine,
  downloadInstallScript,
  hasExistingGitCheckout,
  installRefForStamp,
  isPinnedCommit,
  resolveInstallScript,
  resolveMarkerPinnedCommit,
  runBootstrap,
  terminateInstallerTree
} from './bootstrap-runner'

const SCRIPT_NAME = process.platform === 'win32' ? 'install.ps1' : 'install.sh'
const ZERO_COMMIT = '0000000000000000000000000000000000000000'

function mkTmpHome() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-bootstrap-test-'))
}

test('runBootstrap bails immediately when the signal is already aborted', async () => {
  const controller = new AbortController()
  controller.abort()

  const events = []

  const result = await runBootstrap({
    installStamp: null,
    activeRoot: '/tmp/hermes-runner-test',
    sourceRepoRoot: null,
    hermesHome: '/tmp/hermes-runner-test',
    logRoot: '/tmp/hermes-runner-test',
    onEvent: ev => events.push(ev),
    abortSignal: controller.signal
  })

  // Cancelled before any install script is spawned.
  assert.deepEqual(result, { ok: false, cancelled: true })
  assert.ok(
    events.some(ev => ev.type === 'failed' && /cancelled/i.test(ev.error)),
    'should emit a cancelled failure event'
  )
})

test('cancellation at the final stage success prevents the marker', async () => {
  const home = mkTmpHome()
  const controller = new AbortController()
  let markerWrites = 0
  fs.mkdirSync(path.join(home, 'scripts'))

  const script = process.platform === 'win32'
    ? `param([switch]$Manifest, [string]$Stage, [switch]$NonInteractive, [switch]$Json)\nif ($Manifest) { '{"stages":[{"name":"last"}]}' } else { '{"ok":true,"stage":"last"}' }\n`
    : `#!/usr/bin/env bash\nif [ "$1" = "--manifest" ]; then echo '{"stages":[{"name":"last"}]}'; else echo '{"ok":true,"stage":"last"}'; fi\n`

  fs.writeFileSync(path.join(home, 'scripts', SCRIPT_NAME), script)

  try {
    const result = await runBootstrap({
      installStamp: null, activeRoot: home, sourceRepoRoot: home, hermesHome: home,
      abortSignal: controller.signal,
      writeMarker: () => { markerWrites++;

 return {} },
      onEvent: event => { if (event.type === 'stage' && event.state === 'succeeded') {controller.abort()} }
    })

    assert.equal(result.cancelled, true)
    assert.equal(markerWrites, 0)
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test.each([false, true])('cancelled complete stage preserves existing marker=%s', async existing => {
  const home = mkTmpHome()
  const controller = new AbortController()
  const marker = path.join(home, '.hermes-bootstrap-complete')
  const original = Buffer.from('prior valid marker\n')

  if (existing) {fs.writeFileSync(marker, original)}
  fs.mkdirSync(path.join(home, 'scripts'))

  const script = process.platform === 'win32'
    ? `param([switch]$Manifest, [string]$Stage, [switch]$NonInteractive, [switch]$Json)\n$InstallDir = $env:HERMES_HOME\nif ($Manifest) { '{"stages":[{"name":"complete"}]}' } else { 'new marker' | Set-Content (Join-Path $InstallDir ".hermes-bootstrap-complete") -Encoding UTF8; '{"ok":true,"stage":"complete"}' }\n`
    : `#!/usr/bin/env bash\nINSTALL_DIR="$HERMES_HOME"\nif [ "$1" = "--manifest" ]; then echo '{"stages":[{"name":"complete"}]}'; else printf 'new marker' > "$INSTALL_DIR/.hermes-bootstrap-complete.tmp"; mv -f "$INSTALL_DIR/.hermes-bootstrap-complete.tmp" "$INSTALL_DIR/.hermes-bootstrap-complete"; echo '{"ok":true,"stage":"complete"}'; fi\n`

  fs.writeFileSync(path.join(home, 'scripts', SCRIPT_NAME), script)

  try {
    const result = await runBootstrap({
      installStamp: null, activeRoot: home, sourceRepoRoot: home, hermesHome: home,
      abortSignal: controller.signal,
      writeMarker: () => { throw new Error('unexpected marker write') },
      onEvent: event => { if (event.type === 'stage' && event.state === 'succeeded') {controller.abort()} }
    })

    assert.equal(result.cancelled, true)

    if (existing) {assert.deepEqual(fs.readFileSync(marker), original)}
    else {assert.equal(fs.existsSync(marker), false)}
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('completion marker pin uses staged value before an older marker', () => {
  const home = mkTmpHome()
  const oldPin = 'a'.repeat(40)
  const stagedCommit = 'b'.repeat(40)

  try {
    fs.writeFileSync(path.join(home, '.hermes-bootstrap-complete'), JSON.stringify({ pinnedCommit: oldPin }))
    assert.equal(resolveMarkerPinnedCommit(null, home, { resolveHead: () => null, stagedCommit }), stagedCommit)
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('unrecognized complete stage is refused before modifying its marker', async () => {
  const home = mkTmpHome()
  const marker = path.join(home, '.hermes-bootstrap-complete')
  fs.writeFileSync(marker, 'original')
  fs.mkdirSync(path.join(home, 'scripts'))

  const script = process.platform === 'win32'
    ? `param([switch]$Manifest, [string]$Stage, [switch]$NonInteractive, [switch]$Json)\nif ($Manifest) { '{"stages":[{"name":"complete"}]}' } else { Set-Content (Join-Path $env:HERMES_HOME '.hermes-bootstrap-complete') 'changed'; '{"ok":true,"stage":"complete"}' }\n`
    : `#!/usr/bin/env bash\nif [ "$1" = "--manifest" ]; then echo '{"stages":[{"name":"complete"}]}'; else echo changed > "$HERMES_HOME/.hermes-bootstrap-complete"; echo '{"ok":true,"stage":"complete"}'; fi\n`

  fs.writeFileSync(path.join(home, 'scripts', SCRIPT_NAME), script)

  try {
    const result = await runBootstrap({ installStamp: null, activeRoot: home, sourceRepoRoot: home, hermesHome: home })
    assert.equal(result.ok, false)
    assert.match(result.error, /Unrecognized installer completion marker/)
    assert.equal(fs.readFileSync(marker, 'utf8'), 'original')
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('an exited installer root reports unverified tree termination without signaling its PID', async () => {
  await assert.rejects(terminateInstallerTree({ pid: 1, exitCode: 0, signalCode: null }, true),
    error => (error as Error & { installTerminationUnverified?: boolean }).installTerminationUnverified === true)
})

test.skipIf(process.platform !== 'win32')('cancelling a Windows installer kills its owned child tree', async () => {
  const home = mkTmpHome()
  const controller = new AbortController()
  let childPid = 0
  fs.mkdirSync(path.join(home, 'scripts'))
  fs.writeFileSync(path.join(home, 'scripts', 'install.ps1'), `param([switch]$Manifest, [string]$Stage, [switch]$NonInteractive, [switch]$Json)
if ($Manifest) { '{"stages":[{"name":"last"}]}' ; exit }
$child = Start-Process -FilePath "$PSHOME\\powershell.exe" -ArgumentList '-NoProfile','-Command','Start-Sleep -Seconds 5' -WindowStyle Hidden -PassThru
Write-Output "child-pid=$($child.Id)"
Start-Sleep -Seconds 5
`)

  try {
    const result = await runBootstrap({
      installStamp: null, activeRoot: home, sourceRepoRoot: home, hermesHome: home,
      abortSignal: controller.signal,
      onEvent: event => {
        const match = String(event.line || '').match(/^child-pid=(\d+)$/)

        if (match) { childPid = Number(match[1]); controller.abort() }
      }
    })

    assert.ok(childPid > 0)
    assert.equal(result.cancelled, true)
    assert.throws(() => process.kill(childPid, 0))
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
}, 20_000)

test.each(['abort', 'timeout'])('installer download %s preserves the cached file', async mode => {
  const home = mkTmpHome()
  const dest = path.join(home, 'install.ps1')
  fs.writeFileSync(dest, 'existing')
  const controller = new AbortController()
  let received!: () => void
  const started = new Promise<void>(resolve => { received = resolve })

  const server = http.createServer((_req, response) => {
    response.writeHead(200)
    response.write('partial')
    received()
  })

  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address()
  assert.ok(address && typeof address !== 'string')

  try {
    const downloading = downloadInstallScript('fixture', dest, {
      url: `http://127.0.0.1:${address.port}/`, get: http.get,
      signal: controller.signal, timeoutMs: mode === 'timeout' ? 100 : 5000
    })

    const rejected = assert.rejects(downloading)
    await started

    if (mode === 'abort') {controller.abort()}
    await rejected
    assert.equal(fs.readFileSync(dest, 'utf8'), 'existing')
    assert.deepEqual(fs.readdirSync(home), ['install.ps1'])
  } finally {
    server.closeAllConnections()
    await new Promise<void>(resolve => server.close(() => resolve()))
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('existing checkout detection requires git metadata', () => {
  const home = mkTmpHome()

  try {
    const activeRoot = path.join(home, 'hermes-agent')
    assert.equal(hasExistingGitCheckout(activeRoot), false)

    fs.mkdirSync(path.join(activeRoot, '.git'), { recursive: true })
    assert.equal(hasExistingGitCheckout(activeRoot), true)
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('fresh bootstrap args include the packaged commit pin', () => {
  const installStamp = { commit: 'a'.repeat(40), branch: 'main' }

  assert.deepEqual(buildPinArgs(installStamp), ['-Commit', installStamp.commit, '-Branch', 'main'])
  assert.deepEqual(
    buildPosixPinArgs({
      installStamp,
      activeRoot: '/tmp/hermes-agent',
      hermesHome: '/tmp/hermes'
    }),
    ['--dir', '/tmp/hermes-agent', '--hermes-home', '/tmp/hermes', '--branch', 'main', '--commit', installStamp.commit]
  )
})

test('existing-checkout bootstrap args keep branch but skip the packaged commit pin', () => {
  const installStamp = { commit: 'a'.repeat(40), branch: 'main' }

  assert.deepEqual(buildPinArgs(installStamp, { pinCommit: false }), ['-Branch', 'main'])
  assert.deepEqual(
    buildPosixPinArgs({
      installStamp,
      activeRoot: '/tmp/hermes-agent',
      hermesHome: '/tmp/hermes',
      pinCommit: false
    }),
    ['--dir', '/tmp/hermes-agent', '--hermes-home', '/tmp/hermes', '--branch', 'main']
  )
})

test('fallback install stamps use an unpinned branch ref', () => {
  const stamp = { commit: ZERO_COMMIT, branch: 'main' }

  assert.equal(isPinnedCommit(ZERO_COMMIT), false)
  assert.deepEqual(installRefForStamp(stamp), {
    ref: 'main',
    cacheKey: 'branch-main',
    pinned: false
  })
  // Must NOT pass -Commit / --commit for the all-zero placeholder.
  assert.deepEqual(buildPinArgs(stamp), ['-Branch', 'main'])
  assert.deepEqual(
    buildPosixPinArgs({
      installStamp: stamp,
      activeRoot: '/tmp/hermes',
      hermesHome: '/tmp/home'
    }),
    ['--dir', '/tmp/hermes', '--hermes-home', '/tmp/home', '--branch', 'main']
  )
})

test('existing-checkout installer ref follows the branch instead of the packaged commit', () => {
  const stamp = { commit: 'a'.repeat(40), branch: 'main' }

  assert.deepEqual(installRefForStamp(stamp, { pinCommit: false }), {
    ref: 'main',
    cacheKey: 'branch-main',
    pinned: false
  })
  assert.deepEqual(installRefForStamp(stamp), {
    ref: stamp.commit,
    cacheKey: stamp.commit,
    pinned: true
  })
})

test('resolveMarkerPinnedCommit prefers installed checkout HEAD over the packaged artifact', () => {
  const realHead = 'c'.repeat(40)
  assert.equal(
    resolveMarkerPinnedCommit({ commit: ZERO_COMMIT, branch: 'main' }, '/tmp/checkout', {
      resolveHead: () => realHead
    }),
    realHead
  )
  assert.equal(
    resolveMarkerPinnedCommit({ commit: 'd'.repeat(40), branch: 'main' }, '/tmp/checkout', {
      resolveHead: () => realHead
    }),
    realHead,
    'the installed checkout owns source runtime identity'
  )
  assert.equal(
    resolveMarkerPinnedCommit({ commit: ZERO_COMMIT, branch: 'main' }, '/tmp/missing', {
      resolveHead: () => null
    }),
    null
  )
})

test('resolveInstallScript downloads fallback stamps by branch instead of zero commit', async () => {
  const home = mkTmpHome()

  try {
    const cached = cachedScriptPath(home, 'branch-main')
    fs.mkdirSync(path.dirname(cached), { recursive: true })
    fs.writeFileSync(cached, 'stale branch installer\n')

    const refs = []

    const result = await resolveInstallScript({
      installStamp: { commit: ZERO_COMMIT, branch: 'main' },
      sourceRepoRoot: null,
      hermesHome: home,
      emit: () => {},
      _download: async (ref, destPath) => {
        refs.push(ref)
        fs.mkdirSync(path.dirname(destPath), { recursive: true })
        fs.writeFileSync(destPath, '#!/bin/sh\necho fallback branch\n')

        return destPath
      }
    })

    assert.deepEqual(refs, ['main'])
    assert.equal(result.source, 'download')
    assert.equal(result.commit, null)
    assert.equal(result.path, cached)
    assert.match(fs.readFileSync(cached, 'utf8'), /fallback branch/)
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('resolveInstallScript refreshes the live branch for an existing checkout', async () => {
  const home = mkTmpHome()

  try {
    const commit = 'a'.repeat(40)
    const cached = cachedScriptPath(home, 'branch-main')
    fs.mkdirSync(path.dirname(cached), { recursive: true })
    fs.writeFileSync(cached, 'stale installer\n')

    const refs = []

    const result = await resolveInstallScript({
      installStamp: { commit, branch: 'main' },
      sourceRepoRoot: null,
      hermesHome: home,
      emit: () => {},
      pinCommit: false,
      _download: async (ref, destPath) => {
        refs.push(ref)
        fs.writeFileSync(destPath, 'fresh branch installer\n')

        return destPath
      }
    })

    assert.deepEqual(refs, ['main'])
    assert.equal(result.source, 'download')
    assert.equal(result.commit, null)
    assert.equal(result.path, cached)
    assert.equal(fs.readFileSync(cached, 'utf8'), 'fresh branch installer\n')
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('resolveInstallScript refreshes an immutable-pin cache on a fresh install', async () => {
  const home = mkTmpHome()

  try {
    const commit = 'a'.repeat(40)
    const cached = cachedScriptPath(home, commit)
    fs.mkdirSync(path.dirname(cached), { recursive: true })
    fs.writeFileSync(cached, 'stale installer\n')

    const refs = []

    const result = await resolveInstallScript({
      installStamp: { commit, branch: 'main' },
      sourceRepoRoot: null,
      hermesHome: home,
      emit: () => {},
      _download: async (ref, destPath) => {
        refs.push(ref)
        fs.writeFileSync(destPath, 'fresh pinned installer\n')

        return destPath
      }
    })

    assert.deepEqual(refs, [commit])
    assert.equal(result.source, 'download')
    assert.equal(result.commit, commit)
    assert.equal(result.path, cached)
    assert.equal(fs.readFileSync(cached, 'utf8'), 'fresh pinned installer\n')
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('resolveInstallScript fails closed instead of executing an installed stale script', async () => {
  const home = mkTmpHome()

  try {
    const commit = 'a'.repeat(40)
    const scriptsDir = path.join(home, 'hermes-agent', 'scripts')
    fs.mkdirSync(scriptsDir, { recursive: true })
    fs.writeFileSync(path.join(scriptsDir, SCRIPT_NAME), 'stale installed script\n')

    await assert.rejects(
      resolveInstallScript({
        installStamp: { commit, branch: 'main' },
        sourceRepoRoot: null,
        hermesHome: home,
        emit: () => {},
        _download: async () => {
          throw new Error('Failed to download install script: HTTP 404')
        }
      }),
      /HTTP 404/
    )
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

// #112675: install.sh colours its banners and curl/uv redraw progress with \r
// even into a pipe; the overlay renders lines as plain text, so the emitter
// must hand every consumer (log ring, Details panel, Copy output) the text a
// terminal would be left showing.
test('installer log lines reach the emitter without escape sequences; \\r redraws keep the last frame', () => {
  assert.equal(cleanInstallerLogLine('\u001b[0;32m✓\u001b[0m Detected: macos (macos)'), '✓ Detected: macos (macos)')
  assert.equal(cleanInstallerLogLine('\u001b[2K\u001b[1GCloning repository…\u001b[K'), 'Cloning repository…')
  assert.equal(cleanInstallerLogLine('\u001b]0;hermes\u0007Installing Hermes'), 'Installing Hermes')
  assert.equal(cleanInstallerLogLine('\r 12%\r 67%\r100%\u001b[K'), '100%')
  assert.equal(cleanInstallerLogLine('Resolving dependencies…\r'), 'Resolving dependencies…')
  // Only-escape frames drop entirely, so the caller emits nothing for them.
  assert.equal(cleanInstallerLogLine('\u001b[0m\r'), '')
  // Plain multi-byte text is untouched.
  assert.equal(cleanInstallerLogLine('Ready — café ✓ 中文'), 'Ready — café ✓ 中文')
})

test.skipIf(process.platform === 'win32')(
  'a manifest-step failure surfaces the installer tail without escape sequences',
  async () => {
    const home = mkTmpHome()
    fs.mkdirSync(path.join(home, 'scripts'))
    fs.writeFileSync(
      path.join(home, 'scripts', 'install.sh'),
      '#!/usr/bin/env bash\nprintf "\\033[0;31m\\xe2\\x9c\\x97\\033[0m manifest broke\\n" >&2\nexit 3\n'
    )

    const result = await runBootstrap({
      installStamp: null,
      activeRoot: home,
      sourceRepoRoot: home,
      hermesHome: home,
      logRoot: home,
      onEvent: () => {}
    })

    assert.equal(result.ok, false)
    assert.equal(result.error, 'install.sh --manifest failed: exit 3\n✗ manifest broke')
  }
)
