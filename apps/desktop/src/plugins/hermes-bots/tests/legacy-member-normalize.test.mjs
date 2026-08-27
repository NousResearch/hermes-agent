import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// #92794: older builds persisted group members with a FRIENDLY name as
// `name` (e.g. `name: '大司命'` for the profile slug `taiyi`). Key matching
// alone seats such a descriptor as a ghost NEXT TO its own live roster row
// ("4 bots" in a 2-bot room — reproduced live), and any path that passes the
// ghost's identity onward targets a profile that does not exist on disk.
// resolveLegacyMemberDescriptor() re-tries an unmatched descriptor by
// friendly name against same-connection roster rows before seating.

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

const NL = String.fromCharCode(10)

function stripImports(source) {
  const out = []
  let inImportBlock = false
  for (const line of source.split(NL)) {
    if (inImportBlock) {
      if (line.includes(' from ')) inImportBlock = false
      continue
    }
    if (line.startsWith('import ')) {
      if (!line.includes(' from ')) inImportBlock = true
      continue
    }
    out.push(line)
  }
  return out.join(NL)
}

function load({ groups, meta = {} }) {
  const start = pluginSource.indexOf('function groupChatMemberBots(')
  const end = pluginSource.indexOf('/** Persist source-qualified identities', start)
  assert.notEqual(start, -1)
  assert.notEqual(end, -1)

  const context = {
    $groupChats: { get: () => groups },
    $botMeta: { get: () => meta },
    botGroups: m => (m && Array.isArray(m.groups) ? m.groups : []),
    botRosterMeta: (bot, metaByName) => metaByName?.[bot?.name] || null,
    botRosterKey: bot => `${bot?.connectionId || 'legacy'}::${bot?.name || 'default'}`,
    botFriendlyNames: bot => [
      bot?.ui_meta?.['hermes-bots']?.title,
      // localTitle: the real impl reads the Bot Mode title from $botMeta for
      // local rows — mirror that so meta-titled bots resolve like production.
      !bot?.remoteSource ? context.$botMeta.get()?.[bot?.name]?.title : null,
      bot?.title,
      bot?.display_name
    ]
  }
  const section = stripImports(pluginSource.slice(start, end))
    .concat(NL + 'globalThis.__m = { groupChatMemberBots, resolveLegacyMemberDescriptor };' + NL)
  vm.runInNewContext(section, context, { filename: 'member-seat.js' })
  return context.__m
}

const TAIYI = {
  connectionId: 'local',
  name: 'taiyi',
  display_name: '大司命',
  remoteSource: false
}
const TESTBOT = { connectionId: 'local', name: 'testbot', display_name: '', remoteSource: false }

test('legacy display-name descriptors seat their live row once, not as extra ghosts', () => {
  const { groupChatMemberBots } = load({
    groups: {
      room: {
        // The legacy shape: friendly names persisted as `name`.
        members: [
          { connectionId: 'local', name: '大司命', handle: '大司命' },
          { connectionId: 'local', name: 'Testbot', handle: 'Testbot' }
        ]
      }
    },
    roster: [TAIYI, TESTBOT],
    meta: {
      taiyi: { groups: ['room'] },
      testbot: { groups: ['room'], title: 'Testbot' }
    }
  })

  const seated = groupChatMemberBots('room', [TAIYI, TESTBOT], {
    taiyi: { groups: ['room'] },
    testbot: { groups: ['room'], title: 'Testbot' }
  })

  // Two members, not four: each legacy descriptor resolved to its live row.
  // (vm-realm arrays fail assert.deepEqual on prototype identity — compare
  // via JSON, per the harness contract in the field notes.)
  assert.equal(seated.length, 2, `seated ${seated.map(b => b.name)}`)
  assert.equal(JSON.stringify(seated.map(b => b.name).sort()), JSON.stringify(['taiyi', 'testbot']))
})

test('a genuinely unknown descriptor still seats as a degraded ghost', () => {
  const { groupChatMemberBots } = load({
    groups: { room: { members: [{ connectionId: 'gone', name: 'vanished' }] } },
    roster: [TAIYI],
    meta: { taiyi: { groups: ['room'] } }
  })

  const seated = groupChatMemberBots('room', [TAIYI], { taiyi: { groups: ['room'] } })
  assert.equal(seated.length, 2)
  assert.ok(seated.some(b => b.name === 'vanished'), 'orphan ghost preserved')
})

test('a connectionless pre-scoping descriptor resolves against local rows', () => {
  // The oldest legacy shape: no connectionId at all (pre-connection-scoping
  // rooms only ever held this machine's bots). Reproduced live: such ghosts
  // keyed as legacy::<display-name> and doubled the seated roster.
  const { resolveLegacyMemberDescriptor } = load({ groups: {}, roster: [] })
  const resolved = resolveLegacyMemberDescriptor({ name: '大司命' }, [TAIYI])
  assert.equal(resolved.name, 'taiyi')
})

test('friendly-name matching never crosses connections', () => {
  const remoteTwin = { connectionId: 'other-box', name: 'shadow', display_name: '大司命', remoteSource: true }
  const { resolveLegacyMemberDescriptor } = load({ groups: {}, roster: [] })

  const resolved = resolveLegacyMemberDescriptor(
    { connectionId: 'local', name: '大司命' },
    [remoteTwin]
  )
  // Same friendly name on ANOTHER connection must not capture the member.
  assert.equal(resolved.name, '大司命')
  assert.equal(resolved.connectionId, 'local')
})

test('exact slug descriptors pass through untouched', () => {
  const { resolveLegacyMemberDescriptor } = load({ groups: {}, roster: [] })
  const descriptor = { connectionId: 'local', name: 'taiyi', route: { targetProfile: 'taiyi' } }
  assert.equal(resolveLegacyMemberDescriptor(descriptor, [TAIYI]), descriptor)
})
