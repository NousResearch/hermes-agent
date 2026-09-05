import test from 'node:test'
import assert from 'node:assert/strict'
import os from 'node:os'
import { assertTargetAttested, canon } from './guard.mjs'

// Fake CDP whose Runtime.evaluate returns the given descriptor dataRoot.
const fakeCdp = (dataRoot) => ({
  eval: async (expr) => {
    // guard.mjs evaluates: globalThis.__DEBUG_MCP_INSTANCE__ ? ...dataRoot : null
    if (typeof dataRoot === 'string') return dataRoot
    return dataRoot
  }
})

// Fake CDP whose eval THROWS (renderer Runtime error / disconnect).
const throwingCdp = (msg = 'eval failed') => ({
  eval: async () => {
    throw new Error(msg)
  }
})

test('canon resolves equivalent paths to the same string', () => {
  assert.equal(canon('/tmp/x'), canon('/tmp/./x'))
  assert.equal(canon('/tmp/sb'), canon('/tmp/sb/'))
})

test('refuses when EXPECTED_HOME is unset', async () => {
  await assert.rejects(
    () => assertTargetAttested(fakeCdp('/tmp/sb'), { expectedHome: '', defaultHome: '/real' }),
    /REFUSED/
  )
})

test('refuses when target exposes no descriptor (null dataRoot)', async () => {
  await assert.rejects(
    () => assertTargetAttested(fakeCdp(null), { expectedHome: '/tmp/sb', defaultHome: '/real' }),
    /no debug-instance descriptor/
  )
})

test('P1-1: refuses when declared sandbox != realized target home (attacker lies)', async () => {
  await assert.rejects(
    () =>
      assertTargetAttested(fakeCdp('/real/home'), {
        expectedHome: '/tmp/fake-sandbox',
        defaultHome: '/real/home'
      }),
    /protected operator home/
  )
})

test('P1-1: allows when target home matches declared (canonicalized)', async () => {
  await assert.doesNotReject(
    () =>
      assertTargetAttested(fakeCdp('/tmp/sb/'), {
        expectedHome: '/tmp/./sb',
        defaultHome: '/real/home'
      })
  )
})

test('P1-1: incident shape — target fell back to real home → refused', async () => {
  await assert.rejects(
    () =>
      assertTargetAttested(fakeCdp('/Users/me/.hermes'), {
        expectedHome: '/tmp/sandbox',
        defaultHome: '/Users/me/.hermes'
      }),
    /protected operator home/
  )
})

test('P1-1: mismatch with a NON-protected realized home still reports identity mismatch', async () => {
  // Neither side is the operator home; the plain identity check handles it.
  await assert.rejects(
    () =>
      assertTargetAttested(fakeCdp('/other/sandbox'), {
        expectedHome: '/tmp/sandbox',
        defaultHome: '/Users/me/.hermes'
      }),
    /does not match declared/
  )
})

// --- Edge cases found during review of cdp.eval (returnByValue: true) ---

test('EDGE: descriptor present but dataRoot is undefined → REFUSED (fail-closed)', async () => {
  // __DEBUG_MCP_INSTANCE__ = { nonce: 'x' }  (no dataRoot)
  // expr evaluates to `undefined` (dataRoot missing)
  await assert.rejects(
    () => assertTargetAttested(fakeCdp(undefined), { expectedHome: '/tmp/sb', defaultHome: '/real' }),
    /no debug-instance descriptor/
  )
})

test('EDGE: descriptor is a string, not an object → Runtime TypeError caught → REFUSED', async () => {
  // expr: globalThis.__DEBUG_MCP_INSTANCE__.dataRoot  where instance is a string → throws
  await assert.rejects(
    () => assertTargetAttested(throwingCdp('Cannot read properties of string'), { expectedHome: '/tmp/sb', defaultHome: '/real' }),
    /no debug-instance descriptor/
  )
})

test('EDGE: cdp.eval throws (renderer disconnected / runtime error) → REFUSED', async () => {
  await assert.rejects(
    () => assertTargetAttested(throwingCdp('Target closed'), { expectedHome: '/tmp/sb', defaultHome: '/real' }),
    /no debug-instance descriptor/
  )
})

test('EDGE: descriptor returns an object {dataRoot} (not a bare string) → fallback reads .dataRoot', async () => {
  // If the eval expression ever changes to return the whole object, the
  // `d?.dataRoot ?? null` fallback must still extract it.
  const cdpObj = {
    eval: async () => ({ dataRoot: '/tmp/sb' }) // simulates returnByValue of an object
  }
  await assert.doesNotReject(
    () => assertTargetAttested(cdpObj, { expectedHome: '/tmp/sb', defaultHome: '/real' })
  )
})

test('EDGE: "~" in EXPECTED_HOME is NOT expanded by path.resolve → mismatch → REFUSED', async () => {
  // Operator passed ~/sandbox; main.ts would have resolved HERMES_HOME via
  // os.homedir() so the descriptor carries the real path. The declared home
  // with a literal "~" will not match → REFUSED (documents expected behavior).
  await assert.rejects(
    () =>
      assertTargetAttested(fakeCdp('/Users/me/sandbox'), {
        expectedHome: '~/sandbox',
        defaultHome: '/Users/me/.hermes'
      }),
    /does not match declared/
  )
})

test('EDGE: descriptor dataRoot has trailing slash — canon collapses → allowed', async () => {
  // Electron path.resolve('/tmp/sb/') === '/tmp/sb'; declared '/tmp/sb' also resolves.
  await assert.doesNotReject(
    () => assertTargetAttested(fakeCdp('/tmp/sb/'), { expectedHome: '/tmp/sb', defaultHome: '/real' })
  )
})

test('EDGE: relative EXPECTED_HOME resolved against cwd — mismatch with absolute descriptor → REFUSED', async () => {
  // Operator passed a relative path; descriptor carries absolute. They differ.
  await assert.rejects(
    () => assertTargetAttested(fakeCdp('/abs/sandbox'), { expectedHome: 'rel/sandbox', defaultHome: '/real' }),
    /does not match declared/
  )
})
test('EDGE: macOS /tmp symlink — both declared and descriptor use lexical resolve → allowed', async () => {
  // Electron main computes dataRoot via path.resolve(env.HERMES_HOME), and the
  // operator passes the same string to EXPECTED_HOME, so both sides compare
  // lexically (path.resolve). The renderer never reports a realpath form.
  await assert.doesNotReject(
    () =>
      assertTargetAttested(fakeCdp('/tmp/sb'), {
        expectedHome: '/tmp/sb',
        defaultHome: '/Users/me/.hermes'
      })
  )
})


// --- Protected-home refusal (P1, review round 3) ---

test('REFUSED: expected == realized == the protected operator home', async () => {
  // Caller and target AGREE on ~/.hermes — still forbidden: the debug MCP
  // never acts on the operator's real profile, agreement is not permission.
  const cdp = fakeCdp('/Users/tester/.hermes')
  await assert.rejects(
    () => assertTargetAttested(cdp, { expectedHome: '/Users/tester/.hermes', defaultHome: '/Users/tester/.hermes' }),
    /protected operator home/
  )
})

test('REFUSED: realized is the OS default ~/.hermes even when defaultHome is unset', async () => {
  // Server in a container may have no HERMES_HOME env; the literal ~/.hermes
  // of the OS user is still protected.
  const cdp = fakeCdp(os.homedir() + '/.hermes')
  await assert.rejects(
    () => assertTargetAttested(cdp, { expectedHome: os.homedir() + '/.hermes', defaultHome: '' }),
    /protected operator home/
  )
})

test('allowed: isolated realized/expected exact match, defaultHome differs', async () => {
  const cdp = fakeCdp('/tmp/hermes-p1-home')
  await assert.doesNotReject(
    () => assertTargetAttested(cdp, { expectedHome: '/tmp/hermes-p1-home', defaultHome: '/Users/tester/.hermes' })
  )
})

test('protected check runs AFTER descriptor presence (no descriptor still refuses first)', async () => {
  await assert.rejects(
    () => assertTargetAttested(fakeCdp(null), { expectedHome: '/tmp/sb', defaultHome: '/Users/tester/.hermes' }),
    /no debug-instance descriptor/
  )
})
