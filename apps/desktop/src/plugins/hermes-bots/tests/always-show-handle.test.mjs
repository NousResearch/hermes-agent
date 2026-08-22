import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function between(start, end, from = 0) {
  const a = source.indexOf(start, from)
  const b = source.indexOf(end, a)
  assert.notEqual(a, -1, `missing ${start}`)
  assert.notEqual(b, -1, `missing ${end}`)
  return source.slice(a, b)
}

function load() {
  const handle = between('function botHandle(', '/** Taggable @-forms')
  const visible = between('function showsHandle(', '// ── canonical bot chat')
  const context = {}
  vm.runInNewContext(`${handle}\n${visible}\nglobalThis.api = { botHandle, showsHandle }`, context)
  return context.api
}

test('every local bot row shows its routable handle even when title and slug match', () => {
  const { showsHandle } = load()
  assert.equal(showsHandle('ada', null, { name: 'ada' }), true)
  assert.equal(showsHandle('zubir', { title: 'Zubir' }, { name: 'zubir' }), true)
})

test('default and source-qualified bots show their callable handles', () => {
  const { botHandle, showsHandle } = load()
  assert.equal(botHandle('default', { name: 'default' }), 'hermes')
  assert.equal(showsHandle('default', null, { name: 'default' }), true)
  assert.equal(showsHandle('ops', null, { name: 'ops', handle: 'ops-singapore' }), true)
})
