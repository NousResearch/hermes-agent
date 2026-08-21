import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

// The bot editor (Edit Profile → Advanced) must render the REAL core Capabilities
// components — ToolsetConfigPanel (full per-toolset env/keys/model/post-setup)
// and McpTab (per-server enable + OAuth + API-key setup) — scoped to the bot's
// profile, instead of bare checkbox stand-ins. Both are feature-detected so the
// plugin still loads on older desktop builds that don't export them yet.

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

test('resolves optional Capabilities components from the SDK namespace', () => {
  assert.match(source, /import \* as sdk from '@hermes\/plugin-sdk'/)
  assert.match(source, /const \{ McpTab, ToolsetConfigPanel \} = sdk/)
})

test('AdvancedProfileConfig embeds ToolsetConfigPanel per toolset, scoped to the bot profile', () => {
  // Rendered only when the SDK export exists (older builds: just the toggle).
  assert.match(source, /ToolsetConfigPanel\s*\n?\s*\?\s*jsx\('div'/)
  assert.match(source, /jsx\(ToolsetConfigPanel, \{ toolset: tset\.name, profile: bot \}\)/)
})

test('AdvancedProfileConfig embeds the real McpTab with a live gateway + profile, feature-detected', () => {
  assert.match(source, /McpTab && typeof host\.getGateway === 'function'/)
  assert.match(source, /jsx\(McpTab, \{ gateway: host\.getGateway\(\), profile: bot \}\)/)
})

test('older-build fallback: the checkbox MCP list + inline McpSetupButton is still present', () => {
  // The graceful path for desktops without the SDK export must remain intact.
  assert.match(source, /jsx\(McpSetupButton, \{/)
  assert.match(source, /No MCP servers configured or in the catalog\./)
})
