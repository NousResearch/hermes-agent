import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'

import { checkShadowedImports, shadowedImports } from '../scripts/assert-no-shadowed-imports.mjs'

const IMPORT_AND_DECLARATION = `
import { restorePendingApproval } from './restore-pending-approval'

// The merge resolution kept the import above and restored this local copy.
function restorePendingApproval(response, sessionId) {
  return true
}
`

function makeTree(files) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-shadowed-imports-'))
  for (const [rel, content] of Object.entries(files)) {
    const full = path.join(root, rel)
    fs.mkdirSync(path.dirname(full), { recursive: true })
    fs.writeFileSync(full, content, 'utf8')
  }
  return root
}

// The regression this guard exists for: a catch-up merge resolved an extraction
// hunk by keeping the import and restoring the local function, and the two copies
// had DIVERGED — the local one still carried work the sibling never received — so
// "delete the duplicate" would have built green and silently lost behaviour.
test('flags an imported name that is also declared at module scope', () => {
  assert.deepEqual(shadowedImports(IMPORT_AND_DECLARATION), [
    { name: 'restorePendingApproval', importLine: 2, declarationLine: 5 }
  ])
})

test('checkShadowedImports names the file and both lines', () => {
  const root = makeTree({ 'src/app/session/hooks/index.ts': IMPORT_AND_DECLARATION })
  try {
    const result = checkShadowedImports(path.join(root, 'src'))
    assert.equal(result.ok, false)
    assert.equal(result.violations.length, 1)
    const [violation] = result.violations
    assert.equal(violation.name, 'restorePendingApproval')
    assert.equal(violation.importLine, 2)
    assert.equal(violation.declarationLine, 5)
    assert.match(violation.file, /src\/app\/session\/hooks\/index\.ts$/)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('does not flag a name that is only imported, or only declared', () => {
  const source = `
import { helper } from './helper'

function local() {}
const alsoLocal = () => helper()
`
  assert.deepEqual(shadowedImports(source), [])
})

// An aliased import binds the ALIAS, so `import { a as b }` plus a local `a` is legal
// and must not be reported (and vice versa: a local `b` is the real conflict).
test('compares local binding names, not the exported names', () => {
  assert.deepEqual(shadowedImports("import { original as renamed } from './x'\nfunction original() {}\n"), [])
  assert.deepEqual(shadowedImports("import { original as renamed } from './x'\nfunction renamed() {}\n"), [
    { name: 'renamed', importLine: 1, declarationLine: 2 }
  ])
})

// `import type` is erased at compile time: a value declaration of the same name is
// legal TypeScript, so flagging it would block valid code.
test('ignores type-only imports', () => {
  assert.deepEqual(shadowedImports("import type { Foo } from './foo'\nfunction Foo() {}\n"), [])
})

// Inner scopes legitimately shadow an import; only module scope conflicts.
test('ignores a shadow inside a nested scope', () => {
  const source = `
import { helper } from './helper'

function outer() {
  const helper = 1
  return helper
}
`
  assert.deepEqual(shadowedImports(source), [])
})

// Named-list parsing: commas, `as`, and mixed default+named imports.
test('reads default, namespace and named import bindings', () => {
  const source = `
import React, { useState as useLocal, useEffect } from 'react'
import * as ns from './ns'
function React() {}
const nsLocal = ns
function useEffect() {}
function useLocal() {}
`
  const names = shadowedImports(source).map(v => v.name).sort()
  assert.deepEqual(names, ['React', 'useEffect', 'useLocal'])
})

test('passes on a clean tree and skips node_modules and .d.ts files', () => {
  const root = makeTree({
    'src/clean.ts': "import { thing } from './thing'\nexport const value = thing\n",
    'src/deps/node_modules/pkg/index.ts': IMPORT_AND_DECLARATION,
    'src/types.d.ts': IMPORT_AND_DECLARATION
  })
  try {
    assert.deepEqual(checkShadowedImports(path.join(root, 'src')), { ok: true })
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})
