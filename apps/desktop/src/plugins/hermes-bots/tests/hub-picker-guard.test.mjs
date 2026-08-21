import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

// The Skills Hub picker embed posts {type:'hermes-skill-pick'} messages and
// the plugin installs via skills.manage. The handler checked only
// event.origin — any window on the hub origin (e.g. an OAuth popup returned
// to the hub) could trigger an install while the picker is open, and any
// string reached skills.manage as the install identifier. Picks are now
// pinned to OUR iframe's contentWindow and the identifier is charset-checked.

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

const start = source.indexOf('function HubSkillsSection(')
assert.notEqual(start, -1, 'HubSkillsSection is missing')
const section = source.slice(start, source.indexOf('function emptyAdvancedState', start))

test('regression: hub pick messages are pinned to the embedded frame', () => {
  assert.ok(section.includes('const frameRef = useRef(null)'))
  assert.ok(section.includes('event.source !== frameRef.current.contentWindow'))
  assert.ok(section.includes('ref: frameRef'))
})

test('regression: hub pick identifiers are charset-checked before install', () => {
  assert.ok(section.includes("if (!/^[A-Za-z0-9][A-Za-z0-9._/-]*$/.test(target)) {"))
  assert.ok(section.includes('const target = String(data.identifier || data.name)'))
})
