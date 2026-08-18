import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

// New Agent's lazily-materialized profile (Capabilities tab / MCP setup) is a
// DRAFT until Create Agent is pressed: cancelling the dialog must delete it so
// preconfigure-then-back-out leaves zero residue.

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

test('discardDraft deletes the materialized draft (deleteBot locally, remote CLI for other connections)', () => {
  assert.match(source, /const discardDraft = \(\) => \{/)
  assert.match(source, /: deleteBot\(\{ name: draft \}\)/)
  // Ref cleared FIRST so a re-entrant cancel can't double-delete.
  assert.match(source, /createdRef\.current = null\n\s*flightRef\.current = null\n\s*const discard = remoteTarget/)
})

test('both cancel paths discard the draft (esc/overlay + Cancel button)', () => {
  const dialogClose = /discardDraft\(\)\s*\n\s*reset\(\)\s*\n\s*onClose\(\)/g
  const hits = source.match(dialogClose) || []
  assert.ok(hits.length >= 2, `expected discardDraft before reset/onClose in both cancel paths, found ${hits.length}`)
})

test('successful create does NOT discard: reset clears createdRef before close', () => {
  // reset() nulls the ref, so the dialog-close discard becomes a no-op after
  // a real Create.
  assert.match(source, /setError\(null\)\n\s*createdRef\.current = null\n\s*flightRef\.current = null\n\s*\}/)
})

test('renaming after materialization discards the orphaned draft', () => {
  assert.match(source, /createdRef\.current && createdRef\.current !== slug/)
})

test("the draft's own slug does not read as taken", () => {
  // Both branches of the target-scoped taken check exempt the draft's slug.
  assert.match(source, /roster\.some\(b => !b\.remoteSource && b\.name === slug && b\.name !== createdRef\.current\)/)
  assert.match(
    source,
    /roster\.some\(b => b\.remoteSource && b\.connectionId === targetConnection && b\.name === slug && b\.name !== createdRef\.current\)/
  )
})
