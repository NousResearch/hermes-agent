import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'

import { checkRootInstall } from '../scripts/assert-root-install.mjs'

const BUILD_CRITICAL = ['vite', 'katex', 'electron', 'electron-builder']

// Build a throwaway repo shaped like this one: an app workspace whose
// dependencies are hoisted to the repo root, which is what the guard walks.
function makeTree({ rootPackages = BUILD_CRITICAL, react = '19.2.7', reactDom = '19.2.7' } = {}) {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-assert-root-'))
  const appDir = path.join(tempRoot, 'apps', 'desktop')
  fs.mkdirSync(appDir, { recursive: true })
  fs.writeFileSync(path.join(appDir, 'package.json'), JSON.stringify({ name: 'desktop' }), 'utf8')

  const writePackage = (name, version) => {
    const dir = path.join(tempRoot, 'node_modules', name)
    fs.mkdirSync(dir, { recursive: true })
    fs.writeFileSync(path.join(dir, 'package.json'), JSON.stringify({ name, version }), 'utf8')
  }
  for (const name of rootPackages) writePackage(name, '1.0.0')
  if (react !== null) writePackage('react', react)
  if (reactDom !== null) writePackage('react-dom', reactDom)

  return { tempRoot, appDir }
}

test('checkRootInstall passes on a complete root install', () => {
  const { tempRoot, appDir } = makeTree()
  try {
    assert.deepEqual(checkRootInstall(appDir, tempRoot), { ok: true })
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

// The regression this guard was widened for: the updater's partial `npm install`
// left katex out while vite was present, so the old vite-only check passed and
// the build died on an unresolved `katex/dist/katex.min.css` (#86443).
test('checkRootInstall fails when katex is missing but vite is present', () => {
  const { tempRoot, appDir } = makeTree({
    rootPackages: BUILD_CRITICAL.filter(name => name !== 'katex')
  })
  try {
    const result = checkRootInstall(appDir, tempRoot)
    assert.equal(result.ok, false)
    assert.match(result.error, /katex/)
    assert.match(result.error, /npm ci/)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('checkRootInstall fails when electron is missing', () => {
  const { tempRoot, appDir } = makeTree({
    rootPackages: BUILD_CRITICAL.filter(name => name !== 'electron')
  })
  try {
    const result = checkRootInstall(appDir, tempRoot)
    assert.equal(result.ok, false)
    assert.match(result.error, /electron/)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('checkRootInstall reports every missing package at once', () => {
  const { tempRoot, appDir } = makeTree({ rootPackages: ['vite'] })
  try {
    const result = checkRootInstall(appDir, tempRoot)
    assert.equal(result.ok, false)
    for (const name of ['katex', 'electron', 'electron-builder']) {
      assert.match(result.error, new RegExp(name))
    }
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

// The original guard's only check — kept, so widening coverage cannot silently
// drop the case it already handled.
test('checkRootInstall still fails when vite is missing', () => {
  const { tempRoot, appDir } = makeTree({
    rootPackages: BUILD_CRITICAL.filter(name => name !== 'vite')
  })
  try {
    const result = checkRootInstall(appDir, tempRoot)
    assert.equal(result.ok, false)
    assert.match(result.error, /vite/)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('checkRootInstall fails on a react/react-dom version split', () => {
  const { tempRoot, appDir } = makeTree({ react: '19.2.7', reactDom: '19.1.0' })
  try {
    const result = checkRootInstall(appDir, tempRoot)
    assert.equal(result.ok, false)
    assert.match(result.error, /#527/)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

// A package installed into the app's own node_modules rather than hoisted to the
// root is still installed. The guard walks upward like Node does, so it must not
// insist on the hoisted location.
test('checkRootInstall accepts a package nested in the app workspace', () => {
  const { tempRoot, appDir } = makeTree({
    rootPackages: BUILD_CRITICAL.filter(name => name !== 'katex')
  })
  const nested = path.join(appDir, 'node_modules', 'katex')
  fs.mkdirSync(nested, { recursive: true })
  fs.writeFileSync(path.join(nested, 'package.json'), JSON.stringify({ name: 'katex' }), 'utf8')
  try {
    assert.deepEqual(checkRootInstall(appDir, tempRoot), { ok: true })
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})
