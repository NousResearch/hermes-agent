import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { test } from 'vitest'

// ── Regression guard: module-evaluation order vs. the desktop logger ────────
//
// Live repro (2026-09-03, Fedora, Electron 40.10.2): the app died on every
// launch before it painted a window —
//
//   Uncaught Exception:
//   TypeError: Cannot read properties of undefined (reading 'push')
//       at rememberLog (dist/electron-main.mjs:23178)
//       at readPersistedPoolLimits (dist/electron-main.mjs:23011)
//
// c401756a6a4a (#91545) turned two pure `const` computations into an eager
// `let poolLimits = readPersistedPoolLimits()` at module scope. Every branch
// of that function calls rememberLog(), but rememberLog()'s backing state was
// declared 111 lines further down — inside the temporal dead zone at that
// moment. The bundler lowers module-scope const/let to `var`, so the TDZ
// ReferenceError degraded into `hermesLog` being plain `undefined` and `.push`
// threw. Nothing reached desktop.log either, because the logger itself was
// what died — the crash left no trace in the one file you would go looking in.
//
// The invariant: state that rememberLog() touches must be declared before any
// module-level code that can log while the module is still evaluating.
// Function *declarations* are hoisted and so are exempt; `const`/`let` are not.

const here = path.dirname(fileURLToPath(import.meta.url))
const mainTsSource = fs.readFileSync(path.join(here, 'main.ts'), 'utf8')

/** Module-scope `const`/`let` bindings (column 0), name -> source offset. */
function moduleBindings(source: string): Map<string, number> {
  const found = new Map<string, number>()
  const declaration = /^(?:const|let)\s+([A-Za-z_$][\w$]*)\s*=/gm
  let match: RegExpExecArray | null

  while ((match = declaration.exec(source))) {
    if (!found.has(match[1])) {
      found.set(match[1], match.index)
    }
  }

  return found
}

/** Top-level `function foo(` declarations — hoisted, so order-exempt. */
function localFunctions(source: string): Set<string> {
  return new Set(Array.from(source.matchAll(/^function\s+([A-Za-z_$][\w$]*)\s*\(/gm), m => m[1]))
}

// Slice to the next top-level `function ` declaration — crude but stable for
// the flat function layout main.ts uses, and matching the approach the other
// main.ts source-scanning guards already take.
function extractFunction(source: string, name: string): string {
  const start = source.indexOf(`function ${name}(`)
  assert.notEqual(start, -1, `function ${name} not found in main.ts`)

  const rest = source.slice(start)
  const next = rest.slice(1).search(/\nfunction /)

  return next === -1 ? rest : rest.slice(0, next + 1)
}

function identifiersIn(body: string): Set<string> {
  return new Set(body.match(/[A-Za-z_$][\w$]*/g) ?? [])
}

/**
 * Module-scope bindings rememberLog() reaches, following the local functions
 * it calls directly (scheduleDesktopLogFlush, flushDesktopLogBufferAsync, …).
 * Derived rather than hard-coded so a new binding added to the logger is
 * covered without anyone remembering to update this list.
 */
function loggerStateBindings(): string[] {
  const bindings = moduleBindings(mainTsSource)
  const functions = localFunctions(mainTsSource)
  const body = extractFunction(mainTsSource, 'rememberLog')
  const directCallees = Array.from(identifiersIn(body)).filter(name => functions.has(name) && name !== 'rememberLog')
  const reachable = [body, ...directCallees.map(name => extractFunction(mainTsSource, name))].join('\n')

  return Array.from(identifiersIn(reachable)).filter(name => bindings.has(name))
}

test('logger state is declared before readPersistedPoolLimits() runs at module scope', () => {
  const bindings = moduleBindings(mainTsSource)
  const poolLimitsInit = mainTsSource.search(/^let poolLimits = readPersistedPoolLimits\(\)/m)

  assert.notEqual(
    poolLimitsInit,
    -1,
    'expected a module-level `let poolLimits = readPersistedPoolLimits()` — if this moved, re-point the guard at whatever now logs during module evaluation'
  )

  // All four move together. Hoisting hermesLog alone stops the crash but
  // leaves desktopLogBuffer undefined at that moment, so the pool-limits
  // lines are dropped when its own initializer resets it to '' later in the
  // same evaluation, and the pending flush timer handle is clobbered by
  // desktopLogFlushTimer's initializer.
  for (const name of ['hermesLog', 'desktopLogBuffer', 'desktopLogFlushTimer', 'desktopLogFlushPromise']) {
    const declaredAt = bindings.get(name)

    assert.notEqual(declaredAt, undefined, `expected a module-scope \`${name}\` in main.ts`)
    assert.ok(
      (declaredAt as number) < poolLimitsInit,
      `${name} is declared after readPersistedPoolLimits() runs — that call logs during module ` +
        `evaluation, and the bundler's const->var lowering turns the dead-zone read into undefined, ` +
        `so the main process dies on boot with "Cannot read properties of undefined"`
    )
  }
})

test('no module-level initializer logs before the logger state it depends on exists', () => {
  const bindings = moduleBindings(mainTsSource)
  const functions = localFunctions(mainTsSource)
  const loggerState = loggerStateBindings()

  assert.ok(loggerState.includes('hermesLog'), 'sanity: rememberLog should reach hermesLog')

  // Every `const|let X = someLocalFunction(...)` at module scope: if that
  // function logs, it runs rememberLog() mid-evaluation and every binding the
  // logger touches must already be initialized.
  const initializer = /^(?:const|let)\s+[A-Za-z_$][\w$]*\s*=\s*([A-Za-z_$][\w$]*)\(/gm
  let match: RegExpExecArray | null
  const offenders: string[] = []

  while ((match = initializer.exec(mainTsSource))) {
    const callee = match[1]

    if (!functions.has(callee) || !extractFunction(mainTsSource, callee).includes('rememberLog(')) {
      continue
    }

    for (const name of loggerState) {
      if ((bindings.get(name) as number) > match.index) {
        offenders.push(`${callee}() runs at module scope before \`${name}\` is declared`)
      }
    }
  }

  assert.deepEqual(
    offenders,
    [],
    `module-eval logging reads uninitialized logger state:\n  ${offenders.join('\n  ')}\n` +
      'Move the declaration above the initializer — a dead-zone read here becomes `undefined` ' +
      'after bundling and crashes the main process on boot.'
  )
})
