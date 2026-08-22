import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

// Pull the helper out of the plugin and evaluate it standalone — it is pure,
// so it needs none of the plugin's SDK/React surface. Same vm approach the
// other plugin tests use to exercise a slice of this file.
function loadHelper() {
  const start = source.indexOf('const ACTIVATABLE_SOURCE_KINDS')
  assert.notEqual(start, -1, 'ACTIVATABLE_SOURCE_KINDS must exist')
  const end = source.indexOf('\n}', source.indexOf('function canOpenOnOwnSource(bot) {')) + 2

  return vm.runInNewContext(`${source.slice(start, end)}\ncanOpenOnOwnSource`)
}

test('source-scoped rows open on their own source unless the hop is SSH', () => {
  const canOpen = loadHelper()

  for (const connectionKind of ['remote', 'cloud', 'local']) {
    assert.equal(canOpen({ sourceScoped: true, connectionKind }), true, `${connectionKind} should open in place`)
  }

  // Hopping the window onto an SSH tunnel is the expensive switch that made
  // remote rows @mention-only; that cost is unchanged.
  assert.equal(canOpen({ sourceScoped: true, connectionKind: 'ssh' }), false)
})

test('thin rows without source scoping stay @mention-only', () => {
  const canOpen = loadHelper()

  // No sourceScoped means prepareBotSource never activates an owner, so the
  // canonical-chat RPCs would land on whatever gateway happens to be live.
  assert.equal(canOpen({ connectionKind: 'remote' }), false)
  assert.equal(canOpen({ sourceScoped: false, connectionKind: 'remote' }), false)
  assert.equal(canOpen(null), false)

  // Activation is opt-in on a known kind: a row with no connectionKind could
  // be an SSH tunnel, so it keeps the @mention-only path.
  assert.equal(canOpen({ sourceScoped: true }), false)
})

test('shape: both open paths gate the @mention toast on the same predicate', () => {
  const toasts = [...source.matchAll(/Stay in this chat and @\$\{handle\}/g)]

  assert.equal(toasts.length, 2, 'the bot row and the ActiveNowStrip are the two open paths')

  for (const toast of toasts) {
    // The guard sits immediately above its toast in both paths.
    assert.match(source.slice(Math.max(0, toast.index - 300), toast.index), /!canOpenOnOwnSource\(bot\)/)
  }
})

test('shape: the context menu stays thin-row-only regardless of source kind', () => {
  // Unrelated concern, deliberately NOT gated on canOpenOnOwnSource: a thin
  // row has no profile metadata loaded, so edit/delete/pin would mutate
  // whichever backend is active. The click activates the owner first, and the
  // refreshed rich row is what earns the full menu.
  const menu = source.slice(source.indexOf('// Thin rows from another source are navigation targets only.'))

  assert.match(menu.slice(0, 400), /if \(bot\.remoteSource\) \{\n\s*return row\n\s*\}/)
})

test('shape: opening still activates the owner before any canonical-chat RPC', () => {
  // The whole change rests on this ordering: ensureAgent moves the live
  // gateway onto the bot's (connection, profile) backend, and only then does
  // prepareBotSource verify the source actually became active.
  const prepare = source.slice(
    source.indexOf('async function prepareBotSource('),
    source.indexOf('function displayName(')
  )

  assert.match(prepare, /await host\.ensureAgent\(bot\.connectionId, bot\.name\)/)
  assert.ok(
    prepare.indexOf('await host.ensureAgent(') < prepare.indexOf('Still on '),
    'the liveness check must follow activation, not precede it'
  )
})
