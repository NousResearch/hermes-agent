import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

// The Create Cronjob dialog's "Send results to" target picker: source-shape
// tests in the style of the sibling routine tests (the plugin is a single
// direct file; behavior contracts are pinned via source assertions where a
// full DOM harness would be heavier than the seam warrants).
const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

test('dialog offers a delivery target picker with history and bot-chat options', () => {
  assert.match(pluginSource, /Send results to/)
  assert.match(pluginSource, /id: 'history', label: 'Run history only'/)
  assert.match(pluginSource, /id: 'bot-chat'/)
})

test('bot-chat target sends the BARE deliver token on the profile-scoped create', () => {
  // The job is created in the bot's own cron store (profile: bot), so the
  // bare token resolves to that profile machine-locally — a named token
  // built from a Desktop-side alias could name a profile the backend does
  // not have (the #82530 alias trap). Pin the bare form.
  assert.match(pluginSource, /\.\.\.\(target === 'bot-chat' \? \{ deliver: 'bot-chat' \} : \{\}\)/)
  assert.doesNotMatch(pluginSource, /deliver: `bot-chat:\$\{/)
})

test('history target (default) sends no deliver param — behavior unchanged', () => {
  assert.match(pluginSource, /useState\('history'\)/)
  // reset() returns the picker to the default so a reopened dialog never
  // inherits the previous create's target.
  assert.match(pluginSource, /setTarget\('history'\)/)
})
