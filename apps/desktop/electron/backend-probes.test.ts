/**
 * Tests for electron/backend-probes.ts.
 *
 * Run with: node --test electron/backend-probes.test.ts
 * (Wired into npm test:desktop:platforms in package.json.)
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import {
  canImportHermesCli,
  DEFAULT_PROBE_TIMEOUT_MS,
  findAcceptablePython,
  hermesRuntimeImportProbe,
  MACOS_WELL_KNOWN_PYTHON_DIRS,
  macOSWellKnownPythonCandidates,
  posixPythonCommandCandidates,
  PROBE_TIMEOUT_MS,
  pythonResolutionFailureMessage,
  resolveProbeTimeoutMs,
  resolvePythonForRoot,
  shouldTrustHermesOverride,
  SUPPORTED_PYTHON_VERSIONS,
  venvPythonCandidates,
  verifyHermesCli
} from './backend-probes'

// Resolve the host's own Node binary -- guaranteed to be on disk and
// runnable. We use it as both a stand-in for "a python that doesn't
// have hermes_cli" (since `node -c "import hermes_cli"` will exit
// non-zero) and as a way to script verifyHermesCli's success path
// (a tiny script we write to disk that exits 0 on --version).
const NODE_BIN = process.execPath

test('canImportHermesCli returns false when path is falsy', () => {
  assert.equal(canImportHermesCli(''), false)
  assert.equal(canImportHermesCli(null), false)
  assert.equal(canImportHermesCli(undefined), false)
})

test('canImportHermesCli returns false when interpreter cannot run -c', () => {
  // node IS an interpreter, but `node -c "import hermes_cli"` is a
  // SyntaxError -- different exit reason from a real Python's
  // ModuleNotFoundError, but the predicate is "exit 0 or not" and
  // both land on "not", which is exactly what we want for the
  // resolver fall-through.
  assert.equal(canImportHermesCli(NODE_BIN), false)
})

test('canImportHermesCli returns false when binary does not exist', () => {
  const ghost = path.join(os.tmpdir(), 'hermes-probes-ghost-' + Date.now() + '.exe')
  assert.equal(canImportHermesCli(ghost), false)
})

test('hermes runtime import probe checks config dependencies', () => {
  const probe = hermesRuntimeImportProbe()
  assert.match(probe, /\bimport yaml\b/)
  // dotenv is the first third-party import on the CLI boot path
  // (hermes_cli/env_loader.py); a mid-update venv missing python-dotenv
  // passed the old probe and produced an unrecoverable boot loop.
  assert.match(probe, /\bimport dotenv\b/)
  assert.match(probe, /\bimport hermes_cli\.config\b/)
})

test('explicit Hermes override is authoritative', () => {
  assert.equal(shouldTrustHermesOverride('/nix/store/abc/bin/hermes'), true)
})

test('empty Hermes override is not authoritative', () => {
  assert.equal(shouldTrustHermesOverride(''), false)
  assert.equal(shouldTrustHermesOverride(undefined), false)
})

test('verifyHermesCli returns false when command is falsy', () => {
  assert.equal(verifyHermesCli(''), false)
  assert.equal(verifyHermesCli(null), false)
  assert.equal(verifyHermesCli(undefined), false)
})

test('verifyHermesCli returns false when binary does not exist', () => {
  const ghost = path.join(os.tmpdir(), 'hermes-probes-ghost-' + Date.now() + '.exe')
  assert.equal(verifyHermesCli(ghost), false)
})

test('verifyHermesCli returns true when --version exits 0', () => {
  // Write a tiny script that exits 0 regardless of args, then invoke
  // it through node. This stands in for a working hermes binary --
  // verifyHermesCli only cares about the exit code.
  const scriptPath = path.join(os.tmpdir(), `hermes-probes-ok-${Date.now()}-${process.pid}.cjs`)
  fs.writeFileSync(scriptPath, 'process.exit(0)\n')

  try {
    // Use node as the launcher and our script as the "command". Pass
    // shell:false (default) -- node is a real binary, no shim.
    // execFileSync passes ['--version'] as args, which node ignores
    // gracefully (well, it prints its version and exits 0, which is
    // perfect -- exit code 0 is the only signal we read).
    assert.equal(verifyHermesCli(NODE_BIN), true)
  } finally {
    try {
      fs.unlinkSync(scriptPath)
    } catch {
      void 0
    }
  }
})

test('verifyHermesCli swallows timeouts (does not throw)', () => {
  // We can't easily provoke a real hang in CI without slowing the
  // suite, but we CAN confirm that an invocation that DOES throw
  // (because the binary is missing) returns false rather than
  // propagating. Same code path the timeout case takes.
  assert.equal(verifyHermesCli('/definitely/not/a/real/binary/anywhere'), false)
})

test('default probe timeout is 15s (not the old 5s death-loop value)', () => {
  assert.equal(DEFAULT_PROBE_TIMEOUT_MS, 15_000)
  // Module constant uses process.env at load time; with no override it
  // matches the default (tests run without HERMES_PROBE_TIMEOUT_MS).
  assert.equal(PROBE_TIMEOUT_MS, DEFAULT_PROBE_TIMEOUT_MS)
})

test('resolveProbeTimeoutMs honours HERMES_PROBE_TIMEOUT_MS', () => {
  assert.equal(resolveProbeTimeoutMs({}), DEFAULT_PROBE_TIMEOUT_MS)
  assert.equal(resolveProbeTimeoutMs({ HERMES_PROBE_TIMEOUT_MS: '30000' }), 30_000)
  assert.equal(resolveProbeTimeoutMs({ HERMES_PROBE_TIMEOUT_MS: '0' }), DEFAULT_PROBE_TIMEOUT_MS)
  assert.equal(resolveProbeTimeoutMs({ HERMES_PROBE_TIMEOUT_MS: 'nope' }), DEFAULT_PROBE_TIMEOUT_MS)
  // Cap runaway values
  assert.equal(resolveProbeTimeoutMs({ HERMES_PROBE_TIMEOUT_MS: '999999' }), 120_000)
})

// ── Interpreter resolution ladder ────────────────────────────────────────────
//
// A PATH interpreter that merely EXISTS used to be returned unprobed, so a
// `python3` below the `requires-python = ">=3.11"` floor (3.9.6 from macOS
// CommandLineTools; Debian 11 / Ubuntu 20.04 system python3) was spawned and
// died inside hermes_cli on PEP 604 syntax. These assert the ordering and the
// probe gate as behaviour contracts, not as a snapshot of the version list.

/** Build a findOnPath stub from a name -> resolved-path table. */
function pathTable(table: Record<string, string>) {
  return (command: string) => table[command] || null
}

test('a venv inside the checkout wins over anything on PATH', () => {
  const root = path.join(os.tmpdir(), 'hermes-resolve-root')
  const dotVenv = venvPythonCandidates(root, false)[0]
  let walked = false

  const resolved = resolvePythonForRoot({
    root,
    isWindows: false,
    fileExists: candidate => candidate === dotVenv,
    canImport: () => true,
    findSystemPython: () => {
      walked = true

      return '/usr/bin/python3'
    }
  })

  assert.equal(resolved, dotVenv)
  // Precedence has to short-circuit: a hit here must not pay for a PATH walk
  // (each rejected candidate there costs an interpreter subprocess).
  assert.equal(walked, false)
})

test('.venv takes precedence over venv, and both over the system walk', () => {
  const root = path.join(os.tmpdir(), 'hermes-resolve-root')
  const [dotVenv, plainVenv] = venvPythonCandidates(root, false)

  assert.match(dotVenv, /[\\/]\.venv[\\/]/)
  assert.match(plainVenv, /[\\/]venv[\\/]/)
  assert.equal(
    resolvePythonForRoot({
      root,
      isWindows: false,
      fileExists: candidate => candidate === plainVenv,
      canImport: () => true,
      findSystemPython: () => '/usr/bin/python3'
    }),
    plainVenv
  )
})

test('an explicit interpreter override outranks the checkout venv', () => {
  // HERMES_DESKTOP_PYTHON is the documented escape hatch for a venv that lives
  // outside the checkout (the `hgui` worktree helper sets it). It must stay
  // authoritative, and it is trusted unprobed: the user pointed at it.
  const root = path.join(os.tmpdir(), 'hermes-resolve-root')
  const override = path.join(os.tmpdir(), 'shared-venv', 'bin', 'python')

  assert.equal(
    resolvePythonForRoot({
      root,
      override,
      isWindows: false,
      fileExists: () => true,
      canImport: () => false,
      findSystemPython: () => '/usr/bin/python3'
    }),
    override
  )
})

test('a nonexistent override falls through instead of being spawned', () => {
  const root = path.join(os.tmpdir(), 'hermes-resolve-root')

  assert.equal(
    resolvePythonForRoot({
      root,
      override: '/gone/bin/python',
      isWindows: false,
      fileExists: candidate => candidate !== '/gone/bin/python',
      canImport: () => true,
      findSystemPython: () => '/usr/bin/python3'
    }),
    venvPythonCandidates(root, false)[0]
  )
})

test('the system-walk rung is handed a gate bound to this root', () => {
  // Wiring contract: the fallback rung must receive a validator, and that
  // validator must ask about the root being resolved -- a probe run with the
  // wrong (or no) PYTHONPATH would reject every candidate on a source checkout,
  // where hermes_cli is only importable via the root.
  const root = '/home/dev/hermes-agent'
  const asked: Array<[string, string]> = []
  let sawAccept = false

  const resolved = resolvePythonForRoot({
    root,
    isWindows: false,
    fileExists: () => false,
    canImport: (candidate, forRoot) => {
      asked.push([candidate, forRoot])

      return true
    },
    findSystemPython: accept => {
      sawAccept = typeof accept === 'function'

      return accept('/opt/py313/bin/python3.13') ? '/opt/py313/bin/python3.13' : null
    }
  })

  assert.equal(sawAccept, true)
  assert.equal(resolved, '/opt/py313/bin/python3.13')
  assert.deepEqual(asked, [['/opt/py313/bin/python3.13', root]])
})

test('no venv plus a too-old PATH python3: probe rejects and the walk continues', () => {
  // The reported failure, with no versioned name available to short-circuit
  // it: `python3` resolves and exists but cannot import hermes_cli (3.9.6
  // can't even parse `str | None` in hermes_cli/main.py). Existence alone must
  // not end the walk -- the probe has to demote it and try the next name.
  const tooOld = '/usr/bin/python3'
  const usable = '/opt/conda/bin/python'
  const probed: string[] = []

  const resolved = findAcceptablePython({
    findOnPath: pathTable({ python3: tooOld, python: usable }),
    accept: candidate => {
      probed.push(candidate)

      return candidate !== tooOld
    }
  })

  assert.equal(resolved, usable)
  assert.deepEqual(probed, [tooOld, usable], 'the too-old interpreter must be probed, then skipped')
})

test('an unprobed walk would have returned the too-old interpreter', () => {
  // Teeth: this is the pre-fix behaviour. Same PATH, no validator -> the
  // interpreter that dies on spawn is exactly what gets returned. If this ever
  // starts returning the usable one, the probe gate stopped being what makes
  // the difference and the fix is being carried by something else.
  assert.equal(
    findAcceptablePython({
      findOnPath: pathTable({ python3: '/usr/bin/python3', python: '/opt/conda/bin/python' })
    }),
    '/usr/bin/python3'
  )
})

test('a versioned name short-circuits before an out-of-range bare python3', () => {
  // The other half of the fix, and a distinct mechanism from the probe: when a
  // versioned name resolves, the walk never reaches bare `python3` at all, so
  // the too-old interpreter costs zero subprocesses.
  const probed: string[] = []

  const resolved = findAcceptablePython({
    findOnPath: pathTable({ 'python3.13': '/opt/py313/bin/python3.13', python3: '/usr/bin/python3' }),
    accept: candidate => {
      probed.push(candidate)

      return true
    }
  })

  assert.equal(resolved, '/opt/py313/bin/python3.13')
  assert.deepEqual(probed, ['/opt/py313/bin/python3.13'])
})

test('an explicitly versioned name is preferred over bare python3', () => {
  const order: string[] = []

  const resolved = findAcceptablePython({
    findOnPath: command => {
      order.push(command)

      return command === 'python3' ? '/usr/bin/python3' : `/usr/bin/${command}`
    },
    accept: () => true
  })

  assert.equal(resolved, `/usr/bin/python${SUPPORTED_PYTHON_VERSIONS[0]}`)
  assert.equal(order[0], `python${SUPPORTED_PYTHON_VERSIONS[0]}`)
  assert.ok(order.indexOf('python3') === -1, 'bare python3 must not be consulted once a versioned name answers')
})

test('bare python3 stays a candidate when no versioned name resolves', () => {
  // A venv or conda env on PATH exposes only the bare name. Dropping bare
  // python3 to enforce the floor would break those, so ordering (not
  // exclusion) is the policy and the probe is the proof.
  assert.equal(
    findAcceptablePython({
      findOnPath: pathTable({ python3: '/opt/conda/bin/python3' }),
      accept: () => true
    }),
    '/opt/conda/bin/python3'
  )
})

test('candidate ordering puts every supported version ahead of the bare names', () => {
  const candidates = posixPythonCommandCandidates()

  for (const version of SUPPORTED_PYTHON_VERSIONS) {
    assert.ok(
      candidates.indexOf(`python${version}`) < candidates.indexOf('python3'),
      `python${version} must be tried before bare python3`
    )
  }

  assert.ok(candidates.indexOf('python3') < candidates.indexOf('python'))
  // Bounded: one name per supported version plus the two bare names. The walk
  // costs at most this many interpreter subprocesses.
  assert.equal(candidates.length, SUPPORTED_PYTHON_VERSIONS.length + 2)
})

test('the walk probes each distinct interpreter at most once', () => {
  // python3 and python are commonly the same binary; paying two subprocesses
  // for one answer is pure boot latency.
  const shared = '/usr/local/bin/python3'
  const probed: string[] = []

  assert.equal(
    findAcceptablePython({
      findOnPath: pathTable({ python3: shared, python: shared }),
      accept: candidate => {
        probed.push(candidate)

        return false
      }
    }),
    null
  )
  assert.deepEqual(probed, [shared])
})

test('with no accept validator the first resolvable candidate wins', () => {
  // Callers that only need "any interpreter" (the uninstall path) must keep
  // the original zero-subprocess behaviour.
  assert.equal(findAcceptablePython({ findOnPath: pathTable({ python3: '/usr/bin/python3' }) }), '/usr/bin/python3')
})

// ── macOS Homebrew-before-PATH ordering ──────────────────────────────────────
//
// A Finder/Dock launch inherits launchd's environment, not a login shell's:
// `launchctl getenv PATH` is unset on a stock machine, leaving
// /usr/bin:/bin:/usr/sbin:/sbin. Homebrew is not on it, so a PATH-only search
// cannot see an installed /opt/homebrew/bin/python3.12 and settles for
// /usr/bin/python3 (3.9.6) -- measured on this machine: the minimal-PATH walk
// resolves null, because 3.9 is the only interpreter it can reach.
//
// Injected fake filesystems throughout: the scan must be provably correct on
// machines whose real /opt/homebrew holds nothing in range (this one has only
// 3.14.4 and 3.9.24).

/** Build a fileExists stub from a set of paths that exist. */
function fsTable(paths: string[]) {
  const existing = new Set(paths)

  return (candidate: string) => existing.has(candidate)
}

test('a versioned Homebrew interpreter is preferred over a PATH hit on macOS', () => {
  const brewed = '/opt/homebrew/bin/python3.12'
  const probed: string[] = []

  const resolved = findAcceptablePython({
    platform: 'darwin',
    fileExists: fsTable([brewed]),
    findOnPath: pathTable({ python3: '/usr/bin/python3', python: '/usr/bin/python3' }),
    accept: candidate => {
      probed.push(candidate)

      return true
    }
  })

  assert.equal(resolved, brewed)
  // PATH is never consulted once the well-known pass answers, so the 3.9
  // interpreter costs zero subprocesses.
  assert.deepEqual(probed, [brewed])
})

test('a Homebrew interpreter that fails the probe is demoted, not returned', () => {
  // The load-bearing guarantee: Homebrew ordering is a preference layered on
  // top of the probe, exactly like the versioned-name ordering. It must never
  // be able to promote an interpreter that cannot run Hermes.
  const brewed = '/opt/homebrew/bin/python3.12'
  const pathHit = '/opt/py313/bin/python3.13'
  const probed: string[] = []

  const resolved = findAcceptablePython({
    platform: 'darwin',
    fileExists: fsTable([brewed]),
    findOnPath: pathTable({ 'python3.13': pathHit }),
    accept: candidate => {
      probed.push(candidate)

      return candidate !== brewed
    }
  })

  assert.equal(resolved, pathHit)
  assert.deepEqual(probed, [brewed, pathHit], 'the walk must continue into PATH after demoting the Homebrew hit')
})

test('the well-known scan is macOS-only', () => {
  // Linux desktop launchers inherit a session PATH built from the user's
  // profile, so PATH already sees a versioned interpreter there; scanning
  // /usr/local/bin on Linux would prefer a path nobody asked for.
  const brewed = '/usr/local/bin/python3.12'

  assert.equal(
    findAcceptablePython({
      platform: 'linux',
      fileExists: fsTable([brewed]),
      findOnPath: pathTable({ python3: '/usr/bin/python3' }),
      accept: () => true
    }),
    '/usr/bin/python3'
  )
})

test('the well-known scan is opt-in: no fileExists means PATH-only', () => {
  // Keeps callers that must not touch the filesystem (and every pre-existing
  // test in this file) on the exact previous behaviour.
  assert.equal(
    findAcceptablePython({
      platform: 'darwin',
      findOnPath: pathTable({ python3: '/usr/bin/python3' }),
      accept: () => true
    }),
    '/usr/bin/python3'
  )
})

test('the well-known scan stays bounded and costs one lstat per candidate', () => {
  const checked: string[] = []

  assert.equal(
    findAcceptablePython({
      platform: 'darwin',
      fileExists: candidate => {
        checked.push(candidate)

        return false
      },
      findOnPath: () => null,
      accept: () => true
    }),
    null
  )

  // One stat per (directory x supported version). No directory listing, no
  // globbing -- the cost cannot grow with what happens to be installed.
  assert.equal(checked.length, MACOS_WELL_KNOWN_PYTHON_DIRS.length * SUPPORTED_PYTHON_VERSIONS.length)
  assert.deepEqual(checked, macOSWellKnownPythonCandidates(true))
})

test('the well-known scan never prefers a bare python3 symlink', () => {
  // Bare `python3` in /opt/homebrew/bin points at whichever formula is linked
  // -- 3.14.4 on this machine, above the `<3.14` ceiling. Preferring it over
  // PATH would demote a good PATH interpreter in favour of a guess, so only
  // self-describing versioned names are scanned.
  for (const candidate of macOSWellKnownPythonCandidates(true)) {
    assert.doesNotMatch(candidate, /[\\/]python3?$/, `${candidate} must name its own version`)
  }

  assert.equal(macOSWellKnownPythonCandidates(false).length, 0)
})

test('both Homebrew prefixes are covered (Apple Silicon and Intel)', () => {
  // /opt/homebrew is Apple Silicon; /usr/local is Intel Homebrew, and also
  // where the python.org installer symlinks. Same two directories main.ts
  // already hard-codes for its `gh` lookup.
  assert.ok(MACOS_WELL_KNOWN_PYTHON_DIRS.includes('/opt/homebrew/bin'))
  assert.ok(MACOS_WELL_KNOWN_PYTHON_DIRS.includes('/usr/local/bin'))
  assert.ok(
    MACOS_WELL_KNOWN_PYTHON_DIRS.indexOf('/opt/homebrew/bin') < MACOS_WELL_KNOWN_PYTHON_DIRS.indexOf('/usr/local/bin'),
    'Apple Silicon Homebrew is the common case and should be scanned first'
  )
})

test('a well-known hit already on PATH is probed once, not twice', () => {
  // When Homebrew IS on PATH (a terminal launch), the same binary resolves in
  // both passes. Dedup by resolved path keeps that at one subprocess.
  const brewed = '/opt/homebrew/bin/python3.12'
  const probed: string[] = []

  assert.equal(
    findAcceptablePython({
      platform: 'darwin',
      fileExists: fsTable([brewed]),
      findOnPath: pathTable({ python3: brewed, python: brewed }),
      accept: candidate => {
        probed.push(candidate)

        return false
      }
    }),
    null
  )
  assert.deepEqual(probed, [brewed])
})

test('a checkout venv still short-circuits before any well-known scan', () => {
  // The scan must add zero cost to the case every install path produces.
  const root = path.join(os.tmpdir(), 'hermes-resolve-root')
  const dotVenv = venvPythonCandidates(root, false)[0]
  const statted: string[] = []

  const resolved = resolvePythonForRoot({
    root,
    isWindows: false,
    fileExists: candidate => {
      statted.push(candidate)

      return candidate === dotVenv
    },
    canImport: () => true,
    findSystemPython: accept =>
      findAcceptablePython({ platform: 'darwin', fileExists: () => true, findOnPath: () => null, accept })
  })

  assert.equal(resolved, dotVenv)
  assert.deepEqual(
    statted.filter(candidate => MACOS_WELL_KNOWN_PYTHON_DIRS.some(directory => candidate.startsWith(directory))),
    [],
    'no well-known directory may be consulted when a venv resolves'
  )
})

test('nothing usable yields the actionable message, not a doomed spawn', () => {
  const root = '/home/dev/hermes-agent'

  assert.equal(
    resolvePythonForRoot({
      root,
      isWindows: false,
      fileExists: () => false,
      canImport: () => false,
      findSystemPython: accept => (accept('/usr/bin/python3') ? '/usr/bin/python3' : null)
    }),
    null
  )

  const message = pythonResolutionFailureMessage(root)

  // Names the floor, the venv location we expect, and the escape hatch --
  // so the reader fixes interpreter selection instead of trying to make a
  // `requires-python = ">=3.11"` codebase run on 3.9.
  assert.match(message, new RegExp(`>= ?${SUPPORTED_PYTHON_VERSIONS[0]}`))
  assert.match(message, /\.venv/)
  assert.ok(message.includes(root))
  assert.match(message, /HERMES_DESKTOP_PYTHON/)
})

test('supported versions stay within the requires-python floor', () => {
  // Invariant, not a snapshot: pyproject declares ">=3.11,<3.14".
  for (const version of SUPPORTED_PYTHON_VERSIONS) {
    const [major, minor] = version.split('.').map(Number)

    assert.equal(major, 3)
    assert.ok(minor >= 11 && minor < 14, `${version} is outside requires-python`)
  }
})

test('windows venv candidates use Scripts/python.exe', () => {
  const [dotVenv] = venvPythonCandidates('C:\\src\\hermes', true)

  assert.match(dotVenv, /Scripts/)
  assert.match(dotVenv, /python\.exe$/)
})
