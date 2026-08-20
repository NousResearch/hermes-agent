import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// Group-composer mentions (#89049): the room's "new thread" composer and the
// reply-in-thread box mount a member-scoped @-mention popover instead of a
// dead plain Input. Source-shape assertions plus a functional check of the
// caret tokenizer, extracted and run in isolation.

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

test('group room composers mount GroupMentionInput, not bare Input', () => {
  // Main composer (new thread) and thread reply box both use the wrapper.
  const mainComposer = source.match(/jsx\(GroupMentionInput, \{\s*'aria-label': `Message \$\{group\}`/)
  const replyComposer = source.match(/jsx\(GroupMentionInput, \{\s*'aria-label': 'Reply in thread'/)
  assert.ok(mainComposer, 'main group composer must render GroupMentionInput')
  assert.ok(replyComposer, 'thread reply box must render GroupMentionInput')

  // Neither composer renders a dead plain Input anymore.
  assert.ok(!/jsx\(Input, \{\s*'aria-label': `Message \$\{group\}`/.test(source))
  assert.ok(!/jsx\(Input, \{\s*'aria-label': 'Reply in thread'/.test(source))

  // Both receive the seated members for the popover scope.
  assert.ok(/GroupMentionInput\(\{ members, onChange, value/.test(source))
})

test('popover offers @everyone/@all and inserts parser-compatible strings', () => {
  const component = source.slice(
    source.indexOf('function GroupMentionInput'),
    source.indexOf('function GroupChatWorkspace')
  )
  assert.ok(component.includes("['everyone', 'all']"), 'quick picks for the room-wide broadcast')
  // Insertion writes exactly "@handle " — the shape parseGroupChatMentions resolves.
  assert.ok(component.includes('`${value.slice(0, token.start)}@${handle} ${value.slice(caret)}`'))
  // Member handles come from the same botHandle used by the parser.
  assert.ok(component.includes('botHandle(member.name, member)'))
  // mousedown insertion must preventDefault so the input keeps focus.
  assert.ok(/onMouseDown: event => \{\s*event\.preventDefault\(\)/.test(component))
})

test('mentionTokenAt finds the active @-token at the caret', () => {
  const start = source.indexOf('function mentionTokenAt')
  const end = source.indexOf('function GroupMentionInput')
  assert.ok(start > 0 && end > start, 'mentionTokenAt must precede GroupMentionInput')

  const ctx = {}
  vm.createContext(ctx)
  vm.runInContext(source.slice(start, end) + '\nthis.fn = mentionTokenAt', ctx)
  const fn = ctx.fn

  // Mid-word @ tokens resolve with the right query + start offset.
  // (Field-wise compare: vm-context objects have a foreign Object prototype,
  // which strict deepEqual rejects.)
  const eq = (actual, expected) => {
    assert.equal(actual?.query, expected?.query)
    assert.equal(actual?.start, expected?.start)
  }
  eq(fn('hey @al', 7), { query: 'al', start: 4 })
  eq(fn('@', 1), { query: '', start: 0 })
  eq(fn('ping @every', 11), { query: 'every', start: 5 })

  // Not a mention context: no token → popover stays closed.
  assert.equal(fn('email me a@b', 12), null)
  assert.equal(fn('plain text', 10), null)
  // Caret before the @ is not inside the token.
  assert.equal(fn('hey @al', 3), null)
})
