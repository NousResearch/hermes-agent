import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

// #37: the Routines pane must scope cron.manage to the bot's OWN cron store via
// the core RPC's optional `profile` param (a bot's profile can run a separate
// gateway / keep cron in ~/.hermes/profiles/<name>/cron/). Older gateways
// ignore the unknown param, so the [bot:] tag filter stays the fallback.

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

test('loadRoutines forwards a profile scope to cron.manage list + pause', () => {
  const fn = source.slice(source.indexOf('async function loadRoutines('), source.indexOf('function useRoutines('))
  assert.match(fn, /const scope = profile \? \{ profile \} : \{\}/)
  // list call spreads the scope
  assert.match(fn, /action: 'list', include_disabled: true, \.\.\.scope/)
  // legacy-pause call spreads the scope too
  assert.match(fn, /action: 'pause', name: job\.job_id, \.\.\.scope/)
})

test('useRoutines is bot-keyed so switching bots refetches', () => {
  assert.match(source, /queryKey: \[\.\.\.ROUTINES_KEY, key\]/)
  assert.match(source, /queryFn: \(\) => loadRoutines\(bot\)/)
})

test('RoutineRow toggle and create forward the profile scope', () => {
  // RoutineRow pause/resume
  assert.match(source, /requestForBot\(owner, 'cron\.manage', \{ action, name: job\.job_id, \.\.\.\(profile \? \{ profile \} : \{\}\) \}\)/)
  // create
  assert.match(source, /requestForBot\(bot, 'cron\.manage',[\s\S]*\.\.\.\(profile \? \{ profile \} : \{\}\)/)
})
