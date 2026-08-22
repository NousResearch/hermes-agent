import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function loadOwnershipRule() {
  const start = source.indexOf('function botRowOwnsWorkspace(')
  const end = source.indexOf('// ── bot row', start)

  assert.notEqual(start, -1)
  assert.notEqual(end, -1)

  const context = {
    botRosterKey: bot => `${bot.connectionId || 'local'}::${bot.name}`
  }
  vm.runInNewContext(
    `${source.slice(start, end)}\nglobalThis.botRowOwnsWorkspace = botRowOwnsWorkspace`,
    context
  )
  return context.botRowOwnsWorkspace
}

// The roster highlight and the Routines (Cronjobs) tile must follow the chat
// the user is LOOKING AT — the focused session's owner profile — not the
// gateway socket's home. Tab/tile focus moves without swapping the socket, so
// keying these off `host.state.profile` alone highlighted (and scoped the
// Cronjobs panel to) the wrong bot whenever a focused tab showed another
// profile's chat (community report: Newsanalyst chat open, Hermes highlighted).

test('$focusedBotProfile prefers the focused-session owner atom and falls back to the gateway profile', () => {
  assert.match(
    source,
    /const \$focusedBotProfile = host\.state\.focusedSessionProfile \|\| host\.state\.profile/,
    'feature-detected: newer desktops expose focusedSessionProfile; older builds keep the previous gateway-profile behavior'
  )
})

test('BotRow keys the highlight off the focused profile, not the socket home', () => {
  const rowStart = source.indexOf('function BotRow(')
  assert.ok(rowStart >= 0)
  const row = source.slice(rowStart, rowStart + 2000)

  assert.match(row, /const focusedProfile = useValue\(\$focusedBotProfile\)/)
  assert.match(row, /const botsHomeFronted = useValue\(\$botsHomeFronted\)/)
  assert.match(row, /const isActive = botRowOwnsWorkspace\(/)
})

test('visible Bots home owns selection even when a local chat remains focused underneath', () => {
  const ownsWorkspace = loadOwnershipRule()
  const localBot = { connectionId: 'local', name: 'writer', remoteSource: false }
  const remoteBot = { connectionId: 'work', name: 'research', remoteSource: true }
  const selectedRemote = 'work::research'

  assert.equal(ownsWorkspace(remoteBot, null, true, true, 'writer', selectedRemote), true)
  assert.equal(ownsWorkspace(localBot, null, true, true, 'writer', selectedRemote), false)

  // When the chat returns to the front, its focused local owner takes over.
  assert.equal(ownsWorkspace(remoteBot, null, true, false, 'writer', selectedRemote), false)
  assert.equal(ownsWorkspace(localBot, null, true, false, 'writer', selectedRemote), true)

  // A group room owns the workspace independently of either bot row.
  assert.equal(ownsWorkspace(remoteBot, { name: 'room' }, true, true, 'writer', selectedRemote), false)
})

test('BotRow keeps turn-busy (work mood) a socket fact', () => {
  const rowStart = source.indexOf('function BotRow(')
  const row = source.slice(rowStart, rowStart + 6000)

  // Only the gateway-home profile can actually be mid-turn: the mood must NOT
  // switch to the focus-keyed identity.
  assert.match(row, /const isGatewayHome = !bot\.remoteSource && bot\.name === activeProfile/)
  assert.match(
    row,
    /const botMood = workerActive \|\| \(isGatewayHome && gatewayState === 'busy'\) \? 'work' : 'idle'/
  )
})

test('RoutinesPane scopes the Cronjobs tile to the focused chat owner', () => {
  const paneStart = source.indexOf('function RoutinesPane(')
  assert.ok(paneStart >= 0)
  const pane = source.slice(paneStart, paneStart + 1200)

  assert.match(pane, /const focusedProfile = useValue\(\$focusedBotProfile\)/)
  assert.match(pane, /const bot = \(focusedProfile \|\| selected \|\| 'default'\)\.trim\(\) \|\| 'default'/)
  assert.ok(!/useValue\(host\.state\.profile\)/.test(pane), 'the tile must not read the socket-home atom directly')
})

test('the $selectedBot tracker binds the focused profile ladder (reseed + unbind captured)', () => {
  assert.match(source, /const unbindProfileListener = bindProfileSync\(\$focusedBotProfile\)/)
})
