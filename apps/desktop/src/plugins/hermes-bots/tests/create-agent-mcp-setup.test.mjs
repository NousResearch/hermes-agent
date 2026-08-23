import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

// The New Agent dialog must let you configure MCP OAuth / API-key credentials
// DURING creation (not force create-then-edit). It does this by lazily
// materializing the profile on first setup via ensureAgentCreated(), passed to
// McpSetupButton as ensureProfile.

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

test('McpSetupButton accepts an ensureProfile callback and resolves the profile lazily', () => {
  const fn = source.slice(
    source.indexOf('function McpSetupButton('),
    source.indexOf('function botAppearance(')
  )
  // Signature carries ensureProfile.
  assert.match(fn, /function McpSetupButton\(\{ profile, entry, onDone, ensureProfile \}\)/)
  // resolveProfile falls back to ensureProfile() when no profile is set yet.
  assert.match(fn, /const resolveProfile = async \(\) =>/)
  assert.match(fn, /const created = await ensureProfile\(\)/)
  // Both credential paths resolve the profile before adding the server.
  const beginKeys = fn.slice(fn.indexOf('const beginKeys ='), fn.indexOf('const submitKeys ='))
  assert.match(beginKeys, /const profile = await resolveProfile\(\)/)
  const beginOAuth = fn.slice(fn.indexOf('const beginOAuth ='), fn.indexOf('if (supported === false)'))
  assert.match(beginOAuth, /const profile = await resolveProfile\(\)/)
})

test('CreateAgentDialog materializes the profile lazily and idempotently', () => {
  const fn = source.slice(
    source.indexOf('function CreateAgentDialog('),
    source.indexOf('function RoutinesPane(') > source.indexOf('function CreateAgentDialog(')
      ? source.indexOf('function RoutinesPane(')
      : source.length
  )
  // Idempotent creation guard shares the in-flight creation promise.
  assert.match(fn, /const ensureAgentCreated = \(\) =>/)
  assert.match(fn, /singleFlight\(flightRef, async \(\) =>/)
  // The slug ref stays a string for its sibling consumers (taken check,
  // draft discard, MCP setup profile param).
  assert.match(fn, /createdRef\.current = slug/)
  // submit() routes through the same helper (no duplicate profiles.create).
  assert.match(fn, /const slugCreated = await ensureAgentCreated\(\)/)
  // The MCP setup button in Create is wired to lazy creation, not disabled.
  assert.match(fn, /ensureProfile: ensureAgentCreated/)
  // The dead "save the agent first" gate is gone.
  assert.doesNotMatch(source, /save the agent first/)
})
