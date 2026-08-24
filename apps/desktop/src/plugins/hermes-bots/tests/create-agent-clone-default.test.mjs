import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

// CreateAgentDialog seeds cloneFrom with useState('default') — "Clone from:
// default (the main profile)" — but reset() restored it to '__none__' ("Fresh
// profile"). The first dialog open cloned the main profile; every later open
// after a create/cancel silently defaulted to a fresh profile. reset() must
// restore the same default the dialog was mounted with.

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

const initialMatch = source.match(/const \[cloneFrom, setCloneFrom\] = useState\('([^']+)'\)/)
assert.ok(initialMatch, 'cloneFrom useState initializer is missing')

test('regression: New Agent reset restores the initial clone-from default', () => {
  const initial = initialMatch[1]
  assert.equal(initial, 'default')
  assert.match(source, new RegExp(`setCloneFrom\\('${initial}'\\)`))
  assert.doesNotMatch(source, /setCloneFrom\('__none__'\)/)
})
