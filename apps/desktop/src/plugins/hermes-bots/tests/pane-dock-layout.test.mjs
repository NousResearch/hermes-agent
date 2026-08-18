import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

// Bot Mode layout contract:
//  - the Bots pane center-stacks into the sessions zone (SESSIONS | BOTS tab
//    strip), never splits below it, and carries a one-shot heal token so
//    installs that adopted under the old 'bottom' split migrate;
//  - the Cronjobs (routines) pane only exists while the Bots pane is on
//    screen — registered/unregistered through the contribution disposer,
//    driven by the feature-detected host.paneVisibility SDK export, with the
//    always-registered fallback kept for older desktops.

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

test('the Bots pane center-docks into the sessions zone with a heal token', () => {
  assert.match(source, /dock: \{ pane: 'sessions', pos: 'center', heal: 'sessions-tab-v1' \}/)
  // The old workaround split must not come back.
  assert.doesNotMatch(source, /pane: 'sessions', pos: 'bottom'/)
})

test('routines registration is a reusable disposer-returning function', () => {
  assert.match(source, /const registerRoutinesPane = \(\) =>\s*\n\s*ctx\.register\(\{\s*\n\s*id: 'routines'/)
})

test('routines pane rides Bots visibility via feature-detected host.paneVisibility', () => {
  assert.match(source, /typeof host\.paneVisibility === 'function'/)
  assert.match(source, /host\.paneVisibility\(`\$\{ID\}:pane`\)/)
  // Transitions register/unregister through the tracked disposer.
  assert.match(source, /unregisterRoutines \?\?= registerRoutinesPane\(\)/)
  assert.match(source, /unregisterRoutines\(\)\s*\n\s*unregisterRoutines = null/)
  // The visibility listener must not survive plugin disable.
  assert.match(source, /ctx\.onDispose\(stopRoutinesSync\)/)
})

test('older desktops without the SDK export keep the always-registered pane', () => {
  assert.match(source, /\} else \{\s*\n\s*registerRoutinesPane\(\)\s*\n\s*\}/)
})
