/**
 * Wiring regression for interpreter resolution in main.ts.
 *
 * backend-probes.test.ts covers the resolution *policy* (it can import the
 * pure module). It cannot cover whether main.ts still calls it: main.ts imports
 * electron, so no vitest project loads it. That gap is not theoretical --
 * reverting the one line that hands the fallback rung its probe left the entire
 * suite green, which is how the fallback originally came to be the only rung in
 * the ladder spawning an interpreter nobody validated. An out-of-range
 * `python3` (3.9.6 from macOS CommandLineTools/Xcode; Debian 11 / Ubuntu 20.04
 * system python3) then died inside hermes_cli on PEP 604 syntax.
 *
 * So these assert the wiring at the source level. They are deliberately
 * structural, not snapshots: each one names a call-shape invariant, and the
 * version floor and message text are asserted against the exported constants
 * rather than frozen literals.
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import { test } from 'vitest'

import { SUPPORTED_PYTHON_VERSIONS } from './backend-probes'

const MAIN_TS = fs.readFileSync(path.join(import.meta.dirname, 'main.ts'), 'utf8')

/**
 * The body of a top-level `function <name>(` in main.ts, brace-balanced.
 *
 * The opening brace is located after the parameter list closes, so an inline
 * object type in the signature (`options: { accept: ... }`) isn't mistaken for
 * the body.
 */
function functionBody(source: string, name: string) {
  const start = source.indexOf(`function ${name}(`)

  assert.notEqual(start, -1, `main.ts must still define ${name}()`)

  const paramsOpen = source.indexOf('(', start)
  let parenDepth = 0
  let paramsClose = -1

  for (let i = paramsOpen; i < source.length; i += 1) {
    if (source[i] === '(') {
      parenDepth += 1
    } else if (source[i] === ')') {
      parenDepth -= 1

      if (parenDepth === 0) {
        paramsClose = i

        break
      }
    }
  }

  assert.notEqual(paramsClose, -1, `unbalanced parameter list in ${name}()`)

  const open = source.indexOf('{', paramsClose)
  let depth = 0

  for (let i = open; i < source.length; i += 1) {
    if (source[i] === '{') {
      depth += 1
    } else if (source[i] === '}') {
      depth -= 1

      if (depth === 0) {
        return source.slice(start, i + 1)
      }
    }
  }

  throw new Error(`unbalanced braces while reading ${name}()`)
}

test('findPythonForRoot delegates precedence to the pure resolver', () => {
  const body = functionBody(MAIN_TS, 'findPythonForRoot')

  assert.match(body, /resolvePythonForRoot\(/, 'precedence must come from backend-probes, not a second inline copy')
  assert.match(body, /canImport:/, 'the resolver must be given a real import check')
  assert.match(body, /findSystemPython:\s*accept\s*=>\s*findSystemPython\(\{\s*accept\s*\}\)/)
})

test('findPythonForRoot reports the actionable message when nothing qualifies', () => {
  const body = functionBody(MAIN_TS, 'findPythonForRoot')

  assert.match(body, /pythonResolutionFailureMessage\(root\)/)
  assert.match(body, /rememberLog\(/, 'the message has to reach desktop.log / the boot error detail')
})

test('the root-scoped import check puts the root on PYTHONPATH', () => {
  // On a source checkout hermes_cli is only importable via the root, so a probe
  // without PYTHONPATH would reject every candidate and send a working machine
  // to bootstrap.
  const body = functionBody(MAIN_TS, 'canRunHermesFromRoot')

  assert.match(body, /PYTHONPATH/)
  assert.match(body, /canImportHermesCli\(/)
  assert.match(body, /_hermesImportableCache/, 'the probe spawns a subprocess; repeat resolutions must be cached')
})

test('every findSystemPython call site states its accept policy', () => {
  // `accept` is a required option precisely so this stays visible. Two callers
  // legitimately pass null (pre-bootstrap runtime, uninstall rmtree); the point
  // is that no caller can omit it silently.
  //
  // The signature line itself is excluded; only invocations are inspected.
  const callSites = (MAIN_TS.match(/findSystemPython\(\{[^}]*\}\)/g) || []).filter(site => !site.includes('options:'))

  assert.ok(callSites.length >= 3, `expected the known findSystemPython callers, saw ${callSites.length}`)

  for (const site of callSites) {
    // `{ accept }` shorthand and `{ accept: ... }` both declare the policy.
    assert.match(site, /accept\s*[:}]/, `findSystemPython call site without an explicit accept: ${site}`)
  }

  assert.ok(
    !/findSystemPython\(\)/.test(MAIN_TS),
    'a bare findSystemPython() call is the unvetted-interpreter regression'
  )
})

test('the POSIX walk and the Windows passes share one version floor', () => {
  const body = functionBody(MAIN_TS, 'findSystemPython')

  assert.match(body, /findAcceptablePython\(\{\s*findOnPath,\s*accept[,\s}]/, 'POSIX must use the shared walk')
  assert.match(body, /const SUPPORTED_VERSIONS = SUPPORTED_PYTHON_VERSIONS/, 'Windows must reuse the shared list')

  for (const version of SUPPORTED_PYTHON_VERSIONS) {
    assert.ok(
      !new RegExp(`SUPPORTED_VERSIONS = \\[[^\\]]*'${version}'`).test(body),
      'the version floor must not be re-literalised inside main.ts'
    )
  }
})

test('the POSIX walk is handed what the macOS well-known pass needs', () => {
  // The Homebrew-before-PATH pass in backend-probes is opt-in: it only runs
  // when the caller supplies `fileExists` AND a darwin `platform`. main.ts is
  // the only production caller that can supply them, and main.ts is not loaded
  // by any vitest project -- so without this assertion, dropping either
  // argument silently reverts the whole pass with the suite still green. That
  // is the same coverage hole that let the fallback rung run unprobed.
  const body = functionBody(MAIN_TS, 'findSystemPython')
  const call = body.match(/findAcceptablePython\(\{[^}]*\}\)/)

  assert.ok(call, 'main.ts must still call the shared POSIX walk')
  assert.match(call[0], /fileExists/, 'without fileExists the well-known-directory pass never runs')
  assert.match(call[0], /platform:\s*process\.platform/, 'the pass is gated on a darwin platform')
})

test('Windows candidate hits are validated before being returned', () => {
  // Registry / install-dir / py.exe prove a version, not that Hermes can run on
  // it. Each Windows return path must go through the same accept gate.
  const body = functionBody(MAIN_TS, 'findSystemPython')
  const gated = body.match(/&& acceptCandidate\(/g) || []

  assert.match(body, /const acceptCandidate = candidate => !accept \|\| accept\(candidate\)/)
  // Four Windows return paths: registry hit, per-machine install dir, per-user
  // install dir, and the py.exe launcher resolution.
  assert.equal(gated.length, 4, `expected all 4 Windows returns to be gated, saw ${gated.length}`)
})
