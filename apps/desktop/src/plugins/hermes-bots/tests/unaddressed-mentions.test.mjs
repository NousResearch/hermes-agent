import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// #94478 review: exercise the REAL unaddressedGroupMentions from plugin.js
// via the same vm-slice pattern as active-now-strip.test.mjs, plus pin the
// log-index ordering fix (entry ids are UUIDs, NOT monotonic).

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function loadUnaddressed(roomLog) {
  const start = source.indexOf('function unaddressedGroupMentions')
  const end = source.indexOf('/** Drive one bounded round-robin turn for ONE THREAD.')

  assert.ok(start >= 0 && end > start, 'unaddressedGroupMentions block must remain extractable')

  const members = [{ name: 'alpha' }, { name: 'beta' }, { name: 'gamma' }]
  const sandbox = {
    $groupChats: {
      // The production code indexes rooms by name; expose the room under
      // both shapes so either access style resolves.
      get: k => ({ log: roomLog, [k ?? 'g']: { log: roomLog } }),
    },
    groupThreadOf: e => e.thread,
    parseGroupChatMentions: (text, ms) => ({
      mentioned: [...new Set(ms.filter(m => text.includes(`@${m.name}`)).map(m => m.name))],
      everyone: false,
    }),
    groupMemberKey: m => m.name,
    members,
    out: null,
  }
  vm.createContext(sandbox)
  vm.runInContext(`${source.slice(start, end)}\nout = unaddressedGroupMentions('g', members, 't1')`, sandbox)
  return sandbox.out
}

function entry(id, from, text, thread = 't1') {
  return { id, at: 0, from: { kind: 'member', name: from }, text, thread }
}

test('unaddressedGroupMentions flags a cited member with no later post', () => {
  const pending = loadUnaddressed([
    entry('zzzz-0002', 'alpha', 'hello'),
    entry('aaaa-0003', 'beta', 'ping @gamma — take this'),
  ])
  assert.equal(JSON.stringify(pending), JSON.stringify(['gamma']))
})

test('a reply AFTER the citation in log order counts as answered even when its id sorts lower', () => {
  const pending = loadUnaddressed([
    entry('aaaa-0003', 'beta', 'ping @gamma — take this'),
    entry('9999-0004', 'gamma', 'on it'),
  ])
  assert.equal(JSON.stringify(pending), JSON.stringify([]))
})

test('self-citations and user entries never create pending handoffs', () => {
  const pending = loadUnaddressed([
    entry('bbbb-0001', 'alpha', 'I will do @alpha things myself'),
    { id: 'cccc-0002', at: 0, from: { kind: 'user', name: 'You' }, text: '@beta @gamma look here', thread: 't1' },
  ])
  assert.equal(JSON.stringify(pending), JSON.stringify([]))
})
