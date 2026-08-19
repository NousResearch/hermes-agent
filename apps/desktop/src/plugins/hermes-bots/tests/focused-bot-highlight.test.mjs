import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

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
  assert.match(row, /const isActive = !bot\.remoteSource && bot\.name === focusedProfile/)
})

test('BotRow keeps turn-busy (work mood) a socket fact', () => {
  const rowStart = source.indexOf('function BotRow(')
  const row = source.slice(rowStart, rowStart + 3000)

  // Only the gateway-home profile can actually be mid-turn: the mood must NOT
  // switch to the focus-keyed identity.
  assert.match(row, /const isGatewayHome = !bot\.remoteSource && bot\.name === activeProfile/)
  assert.match(row, /const botMood = \(isGatewayHome && gatewayState === 'busy'\) \|\| activeNow \? 'work' : 'idle'/)
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
