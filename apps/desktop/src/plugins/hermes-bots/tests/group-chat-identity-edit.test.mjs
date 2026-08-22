import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

// Group chats have an editable identity: a name (renameable after creation)
// and a room picture (settable during creation and later via Group settings).
// Source-contract style, like the sibling group-chat tests.

test('source contract: creation dialog offers a room picture', () => {
  // Shared picture controls (upload / generate / remove) mount in the create
  // dialog, and a chosen image lands on the freshly created room record.
  assert.match(pluginSource, /function GroupImageControls\(/)
  assert.match(pluginSource, /if \(image\) \{\s*\n\s*room\.image = image/)
})

test('source contract: settings dialog edits name and picture after creation', () => {
  assert.match(pluginSource, /function GroupChatSettingsDialog\(/)
  assert.match(pluginSource, /function renameGroupChat\(/)
  assert.match(pluginSource, /function setGroupChatImage\(/)
  // The room header exposes the entry point.
  assert.match(pluginSource, /Group settings — rename \$\{group\} or set a room picture/)
})

test('source contract: rename re-keys the room AND local memberships, keeps sessions', () => {
  // The room record moves wholesale under the new key (sessions included, so
  // members keep resuming their per-group sessions by stored sid).
  assert.match(pluginSource, /const room = all\[oldName\]\s*\n\s*\n?\s*delete all\[oldName\]/)
  // Local members' canonical groups lists swap old → new via ui_meta.
  assert.match(pluginSource, /botGroups\(meta\)\.map\(g => \(g === oldName \? next : g\)\)/)
  // Collisions are rejected, not silently suffixed — rename is explicit intent.
  assert.match(pluginSource, /taken\.delete\(oldName\)/)
})

test('source contract: room picture persists and hydrates with the room record', () => {
  // Durable writes carry image; hydration restores only a real string.
  assert.match(pluginSource, /image: room\.image \|\| null/)
  assert.match(pluginSource, /image: typeof room\.image === 'string' && room\.image \? room\.image : null/)
})

test('source contract: room picture shows in the roster row and room header', () => {
  // Roster row: picture wins over the fanned member faces when set.
  assert.match(pluginSource, /children: room\.image\s*\n\s*\? jsx\('img', \{/)
  // Header: picture leads the title.
  assert.match(pluginSource, /Room picture \(set via Group settings\) leads the title when present/)
})
