import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

// Newer desktop builds export the WHOLE core Capabilities surface (SkillsView,
// hermes-agent#87317). The bot editor's Advanced section must render it pinned
// to the bot's profile — in Edit Profile directly, and in New Agent behind a
// Capabilities tab that materializes the profile first. Feature-detected so
// the plugin still loads (and keeps its checklist UI) on older desktops.

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

test('resolves SkillsView as an optional SDK namespace export', () => {
  assert.match(source, /import \* as sdk from '@hermes\/plugin-sdk'/)
  assert.match(source, /const SkillsView = typeof sdk === 'undefined' \? undefined : sdk\.SkillsView/)
})

test('Edit Profile renders the pinned Capabilities surface when the export exists', () => {
  assert.match(source, /if \(SkillsView\) \{/)
  assert.match(source, /jsx\(SkillsView, \{ embedded: true, fixedProfile: bot \}\)/)
})

test('New Agent gains a Capabilities tab that materializes the profile first', () => {
  // Tab list swaps to General + Capabilities on newer builds…
  assert.match(source, /\['capabilities', 'Capabilities'\]/)
  // …and opening it creates the profile through the same lazy door MCP setup uses.
  assert.match(source, /id === 'capabilities'/)
  assert.match(source, /ensureAgentCreated\(\)\s*\n?\s*\.then\(created => created && setCreatedForCaps\(created\)\)/)
  assert.match(source, /jsx\(SkillsView, \{\s*embedded: true,\s*fixedProfile: createdForCaps,/)
})

test('remote-target drafts pin the live surface to the target connection', () => {
  // Builds whose SkillsView routes fixedConnection get the live Capabilities
  // tab for remote targets too — pinned to the target machine's backend.
  assert.match(source, /skillsViewRoutesConnections = Boolean\(SkillsView && SkillsView\.supportsFixedConnection\)/)
  assert.match(source, /SkillsView && \(!remoteTarget \|\| skillsViewRoutesConnections\)/)
  assert.match(source, /\.\.\.\(remoteTarget \? \{ fixedConnection: targetConnection \} : \{\}\)/)
})

test('older-build fallback keeps the checklist UI intact', () => {
  // The staged CheckList sections and the hub search section must survive for
  // desktops without the SkillsView export.
  assert.match(source, /jsx\(CheckList, \{ items: visibleSkills/)
  assert.match(source, /jsx\(HubSkillsSection, \{/)
})
