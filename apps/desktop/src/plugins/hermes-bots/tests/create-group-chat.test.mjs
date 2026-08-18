import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

// Discord-style creation flow: the header "+" is a dropdown (New Agent /
// New Group Chat), and New Group Chat opens a checkbox-picker modal with
// search, a name input, and a Create button. Source-contract style, like
// the other roster affordance tests.

test('source contract: header + is a dropdown offering agent and group chat', () => {
  assert.match(pluginSource, /DropdownMenuTrigger/)
  assert.match(pluginSource, /'New Agent'/)
  assert.match(pluginSource, /'New Group Chat'/)
})

test('source contract: create-group modal has search, checkboxes, name, create', () => {
  assert.match(pluginSource, /function CreateGroupChatDialog\(/)
  // Reuses the roster search filter so name/@handle/title all match.
  assert.match(pluginSource, /const visible = filterBots\(roster, allMeta, query\)/)
  // Selection is checkbox-driven and capped at the room member limit.
  assert.match(pluginSource, /const atCap = selected\.length >= GROUP_CHAT_MAX_MEMBERS/)
  // Create requires 2+ members and appends to the canonical multi-group field
  // without replacing existing memberships.
  assert.match(pluginSource, /selected\.length >= 2/)
  assert.match(
    pluginSource,
    /groupMembershipPatch\(botRosterMeta\(bot, allMeta\), groupName, true\)/
  )
  // Existing membership labels enumerate every canonical group.
  assert.match(pluginSource, /const currentGroups = botGroups\(meta\)/)
  assert.match(pluginSource, /currentGroups\.map\(group => `“\$\{group\}”`\)\.join\(', '\)/)
  // Creating drops the user straight into the room (main window when the
  // desktop offers host.openWorkspace, in-panel fallback otherwise).
  assert.match(pluginSource, /onCreated: groupName => openGroupChat\(groupName\)/)
})

test('source contract: group name falls back to member names, Discord-style', () => {
  assert.match(pluginSource, /selected\.map\(bot => displayName\(bot, botRosterMeta\(bot, allMeta\)\)\)\.join\(', '\)/)
})

test('source contract: every selected machine is persisted in the durable room record', () => {
  assert.match(pluginSource, /const roomMembers = durableGroupChatMembers\(selected\)/)
  assert.match(pluginSource, /room\.members = roomMembers/)
  assert.doesNotMatch(pluginSource, /const remoteMembers = selected/)
})
