const test = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')

const { resolveBackendRoot } = require('./resolve-backend-root.cjs')

const HOME = '/Users/x/.hermes'
const RUNTIME = path.join(HOME, 'runtime', 'hermes-agent')
const DEV = path.join(HOME, 'hermes-agent')

// helper: build deps where a given set of roots are "usable" (source root + venv)
function deps(usableRoots, onFallback) {
  const set = new Set(usableRoots)
  return {
    hermesHome: HOME,
    activeRoot: DEV,
    isSourceRoot: (r) => set.has(r),
    hasVenv: (r) => set.has(r),
    onFallback: onFallback || (() => {}),
  }
}

test('default: runtime tree present -> runtime', () => {
  const r = resolveBackendRoot({ ...deps([RUNTIME, DEV]) })
  assert.equal(r.root, RUNTIME)
  assert.equal(r.venvRoot, path.join(RUNTIME, 'venv'))
  assert.equal(r.tier, 'runtime')
})

test('runtime absent -> dev fallback, onFallback fired with runtime-unavailable', () => {
  let fb = null
  const r = resolveBackendRoot({ ...deps([DEV], (reason, root) => { fb = { reason, root } }) })
  assert.equal(r.root, DEV)
  assert.equal(r.venvRoot, path.join(DEV, 'venv'))
  assert.equal(r.tier, 'dev')
  assert.equal(fb.reason, 'runtime-unavailable')
  assert.equal(fb.root, RUNTIME)
})

test('override usable -> override wins over runtime', () => {
  const OVR = '/custom/tree'
  const r = resolveBackendRoot({ ...deps([OVR, RUNTIME, DEV]), override: OVR })
  assert.equal(r.root, OVR)
  assert.equal(r.venvRoot, path.join(OVR, 'venv'))
  assert.equal(r.tier, 'override')
})

test('override set but UNUSABLE -> falls through to runtime, fires override-unusable', () => {
  let reasons = []
  const r = resolveBackendRoot({ ...deps([RUNTIME, DEV], (reason) => reasons.push(reason)), override: '/typo/tree' })
  assert.equal(r.root, RUNTIME)
  assert.equal(r.tier, 'runtime')
  assert.ok(reasons.includes('override-unusable'))
})

test('override unusable AND runtime absent -> dev, both fallbacks fired (same surfaced path)', () => {
  let reasons = []
  const r = resolveBackendRoot({ ...deps([DEV], (reason) => reasons.push(reason)), override: '/typo' })
  assert.equal(r.root, DEV)
  assert.equal(r.tier, 'dev')
  assert.ok(reasons.includes('override-unusable'))
  assert.ok(reasons.includes('runtime-unavailable'))
})

test('broken runtime venv (source root present, no venv) -> dev fallback (AC-5 broken-venv trigger)', () => {
  // RUNTIME is a source root but has NO venv -> not "usable" -> fall back, same path as absent tree
  let fb = null
  const d = {
    hermesHome: HOME,
    activeRoot: DEV,
    isSourceRoot: (r) => r === RUNTIME || r === DEV, // runtime IS a source root
    hasVenv: (r) => r === DEV,                        // but its venv is broken/missing
    onFallback: (reason, root) => { fb = { reason, root } },
  }
  const r = resolveBackendRoot(d)
  assert.equal(r.root, DEV)
  assert.equal(r.tier, 'dev')
  assert.equal(fb.reason, 'runtime-unavailable') // same surfaced-condition path as absent tree
})

test('site-packages derivation: venvRoot tracks the resolved tree (AC-1 pass-3)', () => {
  // a correct root with a stale venvRoot would mis-spawn; assert venvRoot is always <root>/venv
  const r1 = resolveBackendRoot({ ...deps([RUNTIME, DEV]) })
  assert.equal(r1.venvRoot, path.join(r1.root, 'venv'))
  const r2 = resolveBackendRoot({ ...deps([DEV]) })
  assert.equal(r2.venvRoot, path.join(r2.root, 'venv'))
})
