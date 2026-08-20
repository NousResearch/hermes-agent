import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

// The roster highlight and the Routines (Cronjobs) tile must follow the chat
// the user is LOOKING AT — the focused session's owner profile — not the
// gateway socket's home. Tab/tile focus moves without swapping the socket, so
// keying these off `host.state.profile` alone highlighted (and scoped the
// Cronjobs panel to) the wrong bot whenever a focused tab showed another
// profile's chat (community report: Newsanalyst chat open, Hermes highlighted).

test('$focusedBotOwner prefers the connection-qualified focused owner atom', () => {
  assert.match(
    source,
    /const \$focusedBotOwner = host\.state\.focusedSessionOwner \|\|/,
    'newer desktops expose a complete focused owner; older builds retain a feature-detected fallback'
  )
})

test('legacy focused-profile-only SDK never pairs foreign focus with ambient connection', () => {
  const ownerStart = source.indexOf('const $focusedBotProfile =')
  const ownerEnd = source.indexOf('/** Optional secondary navigation', ownerStart)
  const activeStart = source.indexOf('function isActiveRosterBot(')
  const activeEnd = source.indexOf('function botSelectionKey(', activeStart)
  const store = value => ({ get: () => value, listen: () => undefined })
  const context = {
    host: {
      activeConnectionId: () => 'source-a',
      state: {
        connectionId: store('source-a'),
        focusedSessionProfile: store('worker'),
        profile: store('default')
      }
    }
  }

  vm.runInNewContext(
    `${source.slice(ownerStart, ownerEnd)}\n${source.slice(activeStart, activeEnd)}\n` +
      'globalThis.result = { owner: focusedRosterOwner($focusedBotOwner.get()), isActiveRosterBot };',
    context
  )

  assert.equal(context.result.owner, null)
  assert.equal(context.result.isActiveRosterBot({
    connectionId: 'source-a',
    name: 'worker',
    remoteSource: true
  }, context.result.owner), false)
  assert.equal(context.result.isActiveRosterBot({ name: 'default' }, context.result.owner), false)
})

test('BotRow keys the highlight off the focused profile, not the socket home', () => {
  const rowStart = source.indexOf('function BotRow(')
  assert.ok(rowStart >= 0)
  const row = source.slice(rowStart, rowStart + 2000)

  assert.match(row, /const focusedOwner = focusedRosterOwner\(useValue\(\$focusedBotOwner\)\)/)
  assert.match(row, /const isActive = !activeGroup && isActiveRosterBot\(bot, focusedOwner\)/)
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

  assert.match(pane, /const focusedOwner = focusedRosterOwner\(useValue\(\$focusedBotOwner\)\)/)
  assert.match(pane, /const owner = resolveRoutineOwner\(\$lastRoster\.get\(\), focusedOwner, selected\)/)
  assert.ok(!/useValue\(host\.state\.profile\)/.test(pane), 'the tile must not read the socket-home atom directly')
})

test('the $selectedBot tracker binds the focused profile ladder (reseed + unbind captured)', () => {
  assert.match(source, /const unbindProfileListener = bindProfileSync\(\$focusedBotOwner\)/)
})
