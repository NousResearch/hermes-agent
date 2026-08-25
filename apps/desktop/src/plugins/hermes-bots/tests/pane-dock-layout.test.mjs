import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

// Bot Mode layout contract:
//  - the Bots pane center-stacks into the sessions zone (SESSIONS | BOTS tab
//    strip), never splits below it, and carries the ENFORCED dock invariant
//    so every boot re-homes a stacked install — no heal token, no
//    user-placed exemption (the retired one-shot heal left users who had
//    dragged panes stuck stacked forever);
//  - the Cronjobs (routines) pane only exists while a BOT CHAT owns the main
//    workspace and the Bots pane is on screen — registered/unregistered
//    through the contribution disposer, driven by the feature-detected
//    host.paneVisibility SDK export, with the always-registered fallback kept
//    for older desktops. Cronjobs are bot-scoped, so the tile must not sit
//    beside the ownerless Bots home or a group chat.

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

test('the Bots pane center-docks into the sessions zone as an enforced invariant', () => {
  assert.match(source, /dock: \{ pane: 'sessions', pos: 'center', enforce: true \}/)
  // The old workaround split must not come back.
  assert.doesNotMatch(source, /pane: 'sessions', pos: 'bottom'/)
  // Neither may the retired one-shot heal token.
  assert.doesNotMatch(source, /heal: 'sessions-tab-v1'/)
})

test('routines registration is a reusable disposer-returning function', () => {
  assert.match(source, /const registerRoutinesPane = \(\) =>\s*\n\s*ctx\.register\(\{\s*\n\s*id: 'routines'/)
  assert.match(source, /dock: \{ pane: 'workspace', pos: 'right', enforce: true \}/)
})

test('routines pane rides Bots visibility via feature-detected host.paneVisibility', () => {
  assert.match(source, /typeof host\.paneVisibility === 'function'/)
  assert.match(source, /host\.paneVisibility\(`\$\{ID\}:pane`\)/)
  // Transitions register/unregister through the tracked disposer.
  assert.match(source, /unregisterRoutines \?\?= registerRoutinesPane\(\)/)
  assert.match(source, /unregisterRoutines\(\)\s*\n\s*unregisterRoutines = null/)
  // Ownership, not mere visibility: an actual bot chat must own the center.
  assert.match(source, /if \(botChatOwnsWorkspace\(\)\) \{/)
  // None of the lifecycle listeners may survive plugin disable.
  assert.match(source, /ctx\.onDispose\(\(\) => \{\s*\n\s*stopSidebarSync\(\)/)
  assert.match(source, /stopGroupSync\(\)/)
  assert.match(source, /stopFocusSync\?\.\(\)/)
})

test('older desktops without the SDK export keep the always-registered pane', () => {
  assert.match(source, /\} else \{\s*\n\s*registerRoutinesPane\(\)\s*\n\s*\}/)
})
