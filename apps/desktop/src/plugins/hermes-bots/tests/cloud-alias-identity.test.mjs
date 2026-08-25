import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// #89131: a Desktop per-profile Cloud alias ("moxie" → exact Cloud connection
// → backend targetProfile "default") must keep its friendly identity after
// the hosted session activates, and must render as the alias — not generic
// "Hermes" or a hostname label — when Cloud is the only active source.

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function runtime({ connectionId = 'local' } = {}) {
  const atom = value => ({ get: () => value, set: next => { value = next } })
  const jsx = (type, props = {}) => ({ type, props })
  const context = {
    atom,
    jsx,
    jsxs: jsx,
    useQuery: () => ({}),
    useValue: value => (value?.get ? value.get() : value),
    useState: value => [value, () => undefined],
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } },
    host: {
      state: {
        connectionId: { get: () => connectionId, listen: () => undefined },
        profile: { get: () => 'default', listen: () => undefined }
      },
      request: () => undefined
    },
    sdk: new Proxy({}, { get: () => undefined })
  }
  const code = source
    .replace(/^import \* as sdk from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat(
      '\nglobalThis.__indexAliasRoutes = indexAliasRoutes;' +
        '\nglobalThis.__aliasIdentityFor = aliasIdentityFor;' +
        '\nglobalThis.__displayName = displayName;' +
        '\nglobalThis.__botRosterMeta = botRosterMeta;' +
        '\nglobalThis.__botFriendlyNames = botFriendlyNames;' +
        '\nglobalThis.__botMentionTag = botMentionTag;' +
        '\nglobalThis.__botMeta = $botMeta;'
    )
  vm.runInNewContext(code, context)
  return context
}

const MOXIE_ROUTE = { connectionId: 'cloud-abc', mode: 'remote', profile: 'moxie', targetProfile: 'default' }

/** The hosted backend's own roster row after alias handoff: the Cloud
 *  connection answers as its root profile, so the row identity is
 *  (cloud-abc, default) — NOT the alias key. */
const hostedRow = {
  name: 'default',
  connectionId: 'cloud-abc',
  connectionLabel: 'cloud.example.com',
  targetProfile: 'default',
  remoteSource: true,
  sourceScoped: true,
  route: { connectionId: 'cloud-abc', mode: 'remote', profile: 'default', targetProfile: 'default' }
}

test('alias identity survives hosted-session activation (roster refresh)', () => {
  const ctx = runtime()
  ctx.__indexAliasRoutes([
    { connectionId: 'local', mode: 'local', profile: 'default', targetProfile: 'default' },
    MOXIE_ROUTE
  ])

  // No stored title anywhere: the alias NAME is still the identity.
  assert.equal(ctx.__displayName(hostedRow, null), 'Moxie')

  // The alias's Bot Mode title (either meta key generation) claims the row.
  const metaV2 = { 'cloud-abc::moxie': { title: 'Moxie' } }
  assert.equal(ctx.__botRosterMeta(hostedRow, metaV2), metaV2['cloud-abc::moxie'])
  assert.equal(ctx.__displayName(hostedRow, ctx.__botRosterMeta(hostedRow, metaV2)), 'Moxie')

  const metaV1 = { moxie: { title: 'Moxie ✨' } }
  assert.equal(ctx.__displayName(hostedRow, ctx.__botRosterMeta(hostedRow, metaV1)), 'Moxie ✨')
})

test('Cloud-only mode: the sole active-gateway default renders as the alias, not Hermes', () => {
  // Global route is Cloud: the active gateway IS the Cloud connection and
  // profiles.list returns one unannotated rich `default` row.
  const ctx = runtime({ connectionId: 'cloud-abc' })
  ctx.__indexAliasRoutes([MOXIE_ROUTE])

  const soleBot = { name: 'default' }
  assert.equal(ctx.__displayName(soleBot, null), 'Moxie')

  // A user-set title still wins over the alias name.
  assert.equal(ctx.__displayName(soleBot, { title: 'Custom' }), 'Custom')
})

test('alias never leaks to same-named defaults on other connections', () => {
  const ctx = runtime()
  ctx.__indexAliasRoutes([MOXIE_ROUTE])

  const otherDefault = { ...hostedRow, connectionId: 'other-conn', route: null, targetProfile: 'default', connectionLabel: 'Personal' }
  // Different connection: no alias claim — hostname label behavior stands.
  assert.equal(ctx.__aliasIdentityFor(otherDefault), null)
  assert.equal(ctx.__displayName(otherDefault, null), 'Personal')

  // Local default while the ACTIVE gateway is local: untouched "Hermes".
  assert.equal(ctx.__displayName({ name: 'default' }, null), 'Hermes')
})

test('two aliases claiming one backend row fail closed', () => {
  const ctx = runtime()
  ctx.__indexAliasRoutes([
    MOXIE_ROUTE,
    { connectionId: 'cloud-abc', mode: 'remote', profile: 'roxie', targetProfile: 'default' }
  ])

  assert.equal(ctx.__aliasIdentityFor(hostedRow), null)
  // Ambiguous: fall back to the source label, not a guessed alias.
  assert.equal(ctx.__displayName(hostedRow, null), 'cloud.example.com')
})

test('the alias row itself never claims its own alias entry', () => {
  const ctx = runtime()
  ctx.__indexAliasRoutes([MOXIE_ROUTE])

  const aliasRow = {
    name: 'moxie',
    connectionId: 'cloud-abc',
    targetProfile: 'default',
    sourceScoped: true,
    route: { ...MOXIE_ROUTE }
  }
  assert.equal(ctx.__aliasIdentityFor(aliasRow), null)
  assert.equal(ctx.__displayName(aliasRow, null), 'Moxie')
})

test('mentions keep resolving @moxie against the hosted row after handoff', () => {
  const ctx = runtime()
  ctx.__indexAliasRoutes([MOXIE_ROUTE])
  ctx.__botMeta.set({ 'cloud-abc::moxie': { title: 'Moxie' } })

  const friendly = ctx.__botFriendlyNames(hostedRow).filter(Boolean)
  assert.equal(friendly.length, 1)
  assert.equal(friendly[0], 'Moxie')
  assert.equal(ctx.__botMentionTag(hostedRow), 'moxie')
})

test('non-alias routes never enter the index; index refresh replaces stale claims', () => {
  const ctx = runtime()
  ctx.__indexAliasRoutes([
    { connectionId: 'local', mode: 'local', profile: 'rune', targetProfile: 'rune' },
    MOXIE_ROUTE
  ])
  assert.ok(ctx.__aliasIdentityFor(hostedRow))

  // Alias removed from config → next inventory drops the claim.
  ctx.__indexAliasRoutes([{ connectionId: 'local', mode: 'local', profile: 'rune', targetProfile: 'rune' }])
  assert.equal(ctx.__aliasIdentityFor(hostedRow), null)
  assert.equal(ctx.__displayName(hostedRow, null), 'cloud.example.com')
})
