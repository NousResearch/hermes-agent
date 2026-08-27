import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

// #93572: CreateRoutineDialog's "Send results to" picker built its label with
// `displayName({ name: bot }, $botMeta.get()[bot])`. The dialog's `bot` prop
// is routineCreateTarget() output — an owner OBJECT for roster-scoped bots —
// so the label rendered "[object Object]" and the meta lookup keyed the map
// with an object. The label must resolve owner objects through the
// object-aware botRosterMeta() path and only wrap bare strings.

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

test('CreateRoutineDialog bot-chat label resolves owner objects via botRosterMeta', () => {
  const start = pluginSource.indexOf('function CreateRoutineDialog(')
  assert.ok(start >= 0, 'CreateRoutineDialog must exist')
  const dialog = pluginSource.slice(start, start + 8000)

  const label = dialog.match(/\{ id: 'bot-chat', label: `\$\{displayName\(([^`]+)\)\}/)
  assert.ok(label, 'bot-chat picker label must be built through displayName()')

  // Owner objects pass through untouched; only bare strings are wrapped.
  assert.match(label[1], /typeof bot === 'string' \? \{ name: bot \} : bot/)
  // Meta lookup must go through the object-aware resolver, never index
  // $botMeta with a possibly-object key.
  assert.match(label[1], /botRosterMeta\(bot, \$botMeta\.get\(\)\)/)
  assert.ok(
    !/\$botMeta\.get\(\)\[bot\]/.test(dialog),
    'the dialog must not index bot meta with the raw bot prop'
  )
})
