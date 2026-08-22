// Per-room limit overrides. Three states per axis (a number, off, or absent)
// and a safety brake that is itself switchable, so the resolver is where a
// wrong default would quietly change how long every room runs.
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function loadLimits() {
  const consts = source.slice(
    source.indexOf('const GROUP_CHAT_MAX_ROUNDS'),
    source.indexOf('/** Per-room limit overrides')
  )
  assert.ok(consts.includes('GROUP_CHAT_LIMIT_CEILINGS'), 'the ceilings block must stay extractable')

  const fn = name => {
    const start = source.indexOf(`function ${name}(`)
    assert.ok(start >= 0, `${name} not found`)
    const end = source.indexOf('\n}\n', start) + 3
    return source.slice(start, end)
  }

  const context = {}
  vm.runInNewContext(
    [
      consts,
      fn('resolveGroupChatLimits'),
      fn('groupChatDriveCaps'),
      fn('durableGroupChatLimits'),
      fn('groupChatBudgetLabel'),
      'globalThis.__resolve = resolveGroupChatLimits',
      'globalThis.__caps = groupChatDriveCaps',
      'globalThis.__durable = durableGroupChatLimits',
      'globalThis.__label = groupChatBudgetLabel',
      'globalThis.__defaults = { GROUP_CHAT_MAX_ROUNDS, GROUP_CHAT_MAX_MESSAGES, GROUP_CHAT_HISTORY_LIMIT, GROUP_CHAT_MAX_MEMBERS }',
      'globalThis.__ceilings = GROUP_CHAT_LIMIT_CEILINGS',
      'globalThis.__safety = GROUP_CHAT_SAFETY_DEFAULTS'
    ].join('\n'),
    context
  )

  return context
}

const { __resolve: resolve, __caps: caps, __durable: durable, __label: label, __defaults: defaults, __ceilings: ceilings, __safety: safety } =
  loadLimits()

/** Objects minted inside the vm carry that realm's prototype, which
 *  deepEqual compares. Compare the data, not the realm. */
const plain = value => (value === null || value === undefined ? value : JSON.parse(JSON.stringify(value)))

test('a room with no overrides gets the shipped defaults', () => {
  for (const room of [undefined, null, {}, { limits: null }, { limits: {} }]) {
    const limits = resolve(room)
    assert.equal(limits.rounds, defaults.GROUP_CHAT_MAX_ROUNDS)
    assert.equal(limits.messages, defaults.GROUP_CHAT_MAX_MESSAGES)
    assert.equal(limits.members, defaults.GROUP_CHAT_MAX_MEMBERS)
    assert.equal(limits.history, defaults.GROUP_CHAT_HISTORY_LIMIT)
  }
})

test('an explicit number wins over the default', () => {
  const limits = resolve({ limits: { rounds: 9, messages: 40, members: 12, history: 60 } })
  assert.deepEqual(
    [limits.rounds, limits.messages, limits.members, limits.history],
    [9, 40, 12, 60]
  )
})

test('null means off, and off is not the same as absent', () => {
  assert.equal(resolve({ limits: { rounds: null } }).rounds, null)
  assert.equal(resolve({ limits: {} }).rounds, defaults.GROUP_CHAT_MAX_ROUNDS)
})

test('a nonsense value falls back instead of disabling the limit', () => {
  for (const bad of [0, -3, 'lots', NaN, {}, [], true]) {
    assert.equal(resolve({ limits: { rounds: bad } }).rounds, defaults.GROUP_CHAT_MAX_ROUNDS, `rounds: ${String(bad)}`)
  }
})

test('a value above the ceiling is clamped, never rejected', () => {
  assert.equal(resolve({ limits: { rounds: 10_000 } }).rounds, ceilings.rounds)
  assert.equal(resolve({ limits: { messages: 10_000 } }).messages, ceilings.messages)
})

test('the safety brake only exists while its axis is off', () => {
  assert.equal(resolve({ limits: { rounds: 5 } }).safetyRounds, null, 'a bounded axis needs no brake')
  assert.equal(resolve({ limits: { rounds: null } }).safetyRounds, safety.rounds)
  assert.equal(resolve({ limits: { rounds: null, safetyRounds: 12 } }).safetyRounds, 12)
  assert.equal(resolve({ limits: { rounds: null, safetyRounds: null } }).safetyRounds, null)
})

test('drive caps collapse the three states into what the loop needs', () => {
  const bounded = caps(resolve({ limits: { rounds: 4 } }))
  assert.equal(bounded.rounds, 4)
  assert.equal(bounded.roundsUnbounded, false)

  const braked = caps(resolve({ limits: { rounds: null, safetyRounds: 20 } }))
  assert.equal(braked.rounds, 20, 'the brake becomes the loop bound')
  assert.equal(braked.roundsUnbounded, false)

  const free = caps(resolve({ limits: { rounds: null, safetyRounds: null } }))
  assert.equal(free.rounds, null, 'nothing bounds the loop by count')
  assert.equal(free.roundsUnbounded, true)
})

test('the stored form keeps only what the room actually set', () => {
  assert.equal(durable(null), null)
  assert.equal(durable({}), null, 'a room on every default stores nothing')
  assert.deepEqual(plain(durable({ rounds: 5 })), { rounds: 5 })
  assert.deepEqual(plain(durable({ rounds: null })), { rounds: null }, 'off must survive a round trip')
  assert.deepEqual(plain(durable({ rounds: 5, bogus: 1 })), { rounds: 5 }, 'unknown axes are dropped')
  assert.equal(durable({ rounds: 0 }), null, 'a nonsense value is not stored')
})

test('a stored override survives the round trip through the resolver', () => {
  const stored = durable({ rounds: null, safetyRounds: null, messages: 40 })
  const limits = resolve({ limits: stored })
  assert.equal(limits.rounds, null)
  assert.equal(limits.safetyRounds, null)
  assert.equal(limits.messages, 40)
})

test('the header label names all three states', () => {
  assert.equal(label({}), `${defaults.GROUP_CHAT_MAX_ROUNDS} rounds · ${defaults.GROUP_CHAT_MAX_MESSAGES} msgs`)
  assert.match(label({ limits: { rounds: null } }), /≤\d+ rounds/)
  assert.match(label({ limits: { rounds: null, safetyRounds: null } }), /∞ rounds/)
  assert.equal(label({ limits: { rounds: 1, messages: 1 } }), '1 round · 1 msg', 'singular reads correctly')
})

test('the drive loop is bounded by the caps, not by the constants', () => {
  const loop = source.slice(source.indexOf('async function runGroupChatRounds('))
  assert.match(loop, /for \(let round = 0; caps\.rounds === null \|\| round < caps\.rounds; round\+\+\)/)
  assert.match(loop, /caps\.messages !== null && posted >= caps\.messages/)
  assert.doesNotMatch(
    loop.slice(0, loop.indexOf('\n}\n')),
    /GROUP_CHAT_MAX_ROUNDS|GROUP_CHAT_MAX_MESSAGES/,
    'the loop must read the room, not the module defaults'
  )
})

test('the drive reports why it ended, not just that it ended', () => {
  // #92213 (AllanGamal): every ending reported as `settled`, so a room cut off
  // by its budget looked like a room that had finished talking. The reason is
  // recorded where the drive actually exits and reported once in `finally`.
  const loop = source.slice(source.indexOf('async function runGroupChatRounds('))

  assert.match(loop, /let stoppedBy = null/)
  assert.match(loop, /stoppedBy = 'settled'\n\s*return \/\/ everyone passed/)
  assert.match(loop, /stoppedBy = 'messages'/)
  assert.match(loop, /stoppedBy = stoppedBy \?\? \(caps\.rounds === null \? 'settled' : 'rounds'\)/)
  assert.match(loop, /stoppedBy === 'rounds' \|\| stoppedBy === 'messages'/)

  // The reason must not be guessed at the top of the last round: a round that
  // settles there would report a limit stop that never happened.
  assert.doesNotMatch(loop, /round === caps\.rounds - 1/)
})

test('the two stop kinds read differently and both stand out', () => {
  const labels = source.slice(source.indexOf('function groupActivityLabel('))
  const body = labels.slice(0, labels.indexOf('\nconst GROUP_ACTIVITY_LABELS'))

  assert.match(body, /kind === 'safety' \|\| kind === 'capped'/)
  assert.match(body, /safety stop: /)
  assert.match(body, /raise it in the room budget/)
  assert.match(source, /capped: 'stopped at the limit'/)
  assert.match(source, /if \(kind === 'safety' \|\| kind === 'capped'\) \{\n {4}return 'text-\(--ui-warning/)
})

test('the editor warns when the message budget, not the round setting, ends the room', () => {
  const editor = source.slice(source.indexOf('function GroupLimitsControls('))
  const body = editor.slice(0, editor.indexOf('\n}\n'))

  // With N members one round costs N messages, so raising rounds alone is a
  // no-op whenever messages/N is the smaller number. That has to be visible.
  assert.match(body, /Math\.ceil\(effectiveMessages \/ memberCount\) < effectiveRounds/)
  assert.match(body, /raise it too, or the round setting will not change anything/)
})

test('the room header label is what the create dialog and settings also write', () => {
  // One component, three mount points: a second editor would drift.
  const mounts = [...source.matchAll(/jsx\(GroupLimitsControls, \{/g)]
  assert.equal(mounts.length, 2, 'create dialog and group settings mount the shared editor')
  assert.match(source, /children: groupChatBudgetLabel\(room\)/, 'the header shows the same resolved budget')
})

test('the budget row mirrors the app\'s own settings row', () => {
  // src/app/settings/primitives.tsx (ListRow / ToggleRow) is the canonical
  // shape for a labelled control. Plugins cannot import it, so the structure
  // is mirrored. Diverging from it is what made the switches unreadable in
  // the first live test.
  const row = source.slice(source.indexOf('function GroupLimitRow('))
  const body = row.slice(0, row.indexOf('\n/** The room budget editor'))

  assert.match(body, /@container/)
  assert.match(body, /text-\[length:var\(--conversation-text-font-size\)\] font-medium text-foreground/)
  assert.match(body, /@xs:justify-self-end/)

  // The app's own rows split at @2xl, which is 672px of container. These rows
  // live in a 448px dialog, so that breakpoint would never fire and every row
  // would stack label-over-control. Split early instead.
  assert.match(body, /@xs:grid-cols-\[minmax\(0,1fr\)_11rem\]/)
  assert.doesNotMatch(body, /@2xl:grid-cols/)

  // CONTROL_TEXT in src/app/settings/constants.ts is 'text-xs', and the
  // standard field fills its column instead of carrying a fixed width.
  assert.match(body, /className: 'min-w-0 flex-1 text-xs'/)
  assert.doesNotMatch(body, /className: 'w-20/)
})

test('switching an axis fires the same haptic the app uses elsewhere', () => {
  const row = source.slice(source.indexOf('function GroupLimitRow('))
  const body = row.slice(0, row.indexOf('\n/** The room budget editor'))
  const taps = [...body.matchAll(/haptic\('tap'\)/g)]

  assert.equal(taps.length, 2, 'both the limit switch and the brake switch give feedback')
})

test('a switched-off axis shows the infinity placeholder, not an empty box', () => {
  const row = source.slice(source.indexOf('function GroupLimitRow('))
  assert.match(row.slice(0, 4000), /placeholder: off \? '\\u221e' : String\(fallback\)/)
})

test('an activity event cannot overwrite its own timestamp', () => {
  // Live test: the limit event carried `at: 1` for "1 round". recordGroupActivity
  // spread the event over its own `at: Date.now()`, so the room rendered the
  // stop as "20688 days ago". The record's fields win over the caller's.
  const record = source.slice(source.indexOf('function recordGroupActivity('))
  const body = record.slice(0, record.indexOf('\n}\n'))

  assert.match(body, /const entry = \{ \.\.\.event, at: Date\.now\(\), epoch: room\.epoch \|\| 0 \}/)
  assert.doesNotMatch(body, /at: Date\.now\(\), epoch: room\.epoch \|\| 0, \.\.\.event/)
})

test('the limit event carries its count under a name of its own', () => {
  const loop = source.slice(source.indexOf('async function runGroupChatRounds('))

  assert.match(loop, /count: stoppedBy === 'rounds' \? caps\.rounds : caps\.messages/)
  assert.doesNotMatch(loop, /at: stoppedBy === 'rounds'/)
  assert.match(source, /const count = event\?\.count \?\? '\?'/)
})

test('every field the room persists is read back on hydrate', () => {
  // Live test: room budgets did not survive a restart. durableGroupChatRooms
  // wrote `limits`, the gateway merge carried it, and the hydrate path rebuilt
  // the room field by field without it. A persisted field has three sites, and
  // two of three is a field that silently resets.
  const durable = source.slice(source.indexOf('function durableGroupChatRooms('))
  const durableBody = durable.slice(0, durable.indexOf('\n}\n'))

  const hydrateAt = source.indexOf("ctx.storage?.get?.('group-chats')")
  assert.ok(hydrateAt > 0, 'the hydrate path must stay findable')
  const hydrateBody = source.slice(hydrateAt, hydrateAt + 2000)

  // Fields written inside the `durable[name] = { … }` literal.
  const written = [...durableBody.matchAll(/^\s{6}(\w+):/gm)].map(m => m[1])
  assert.ok(written.length >= 6, `expected the durable shape to have fields, saw ${written.length}`)

  // Runtime-only by design: the hydrate path resets these on purpose.
  const runtimeOnly = new Set(['epoch', 'running'])
  const missing = written.filter(f => !runtimeOnly.has(f) && !new RegExp(`\\b${f}:`).test(hydrateBody))

  assert.deepEqual(missing, [], 'these fields are written but never read back')
})

test('a room budget survives the write and read round trip', () => {
  const durable = source.slice(source.indexOf('function durableGroupChatRooms('))
  const durableBody = durable.slice(0, durable.indexOf('\n}\n'))
  assert.match(durableBody, /limits: durableGroupChatLimits\(room\.limits\)/)

  const hydrateAt = source.indexOf("ctx.storage?.get?.('group-chats')")
  assert.match(source.slice(hydrateAt, hydrateAt + 2000), /limits: durableGroupChatLimits\(room\.limits\)/)
})
