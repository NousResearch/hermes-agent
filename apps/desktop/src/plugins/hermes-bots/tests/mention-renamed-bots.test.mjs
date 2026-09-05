import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// Renamed bots stay taggable (Discord report, Aug 2026): renaming a bot —
// Bot Mode title or `hermes profile rename` display_name — must update what
// the user can @-tag it with, in the mention middleware, group rooms, and
// the composer autocomplete. Old profile handles keep resolving.

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function runtime({ meta } = {}) {
  const context = {
    console,
    setTimeout,
    clearTimeout,
    Date,
    URL,
    atom: initial => {
      let value = initial
      return { get: () => value, set: next => (value = next), listen: () => () => undefined }
    },
    host: {
      request: async () => ({}),
      requestProfile: async () => ({}),
      state: {
        profile: { get: () => 'default', listen: () => undefined },
        connectionId: { get: () => 'local', listen: () => undefined },
        gateway: { listen: () => undefined }
      }
    },
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } }
  }
  const code = source
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat(
      '\nglobalThis.__x = { mentionNameForms, botFriendlyNames, botMentionTag, resolveRosterMentions, parseGroupChatMentions, groupMemberKey, durableGroupChatMembers, $botMeta };\n'
    )
  vm.runInNewContext(code, context, { filename: 'plugin.js' })
  if (meta) {
    context.__x.$botMeta.set(meta)
  }
  return context.__x
}

test('mentionNameForms: slugged + collapsed, reserved tokens dropped', () => {
  const { mentionNameForms } = runtime()

  assert.deepEqual(JSON.parse(JSON.stringify(mentionNameForms('Research Buddy'))), ['research-buddy', 'researchbuddy'])
  assert.deepEqual(JSON.parse(JSON.stringify(mentionNameForms('Ops'))), ['ops'])
  // A bot renamed "Hermes" or "everyone" cannot hijack reserved tags.
  assert.equal(mentionNameForms('Hermes').length, 0)
  assert.equal(mentionNameForms('@everyone').length, 0)
  assert.equal(mentionNameForms('').length, 0)
})

test('botMentionTag: renamed slug wins, profile handle is the fallback', () => {
  const { botMentionTag } = runtime({ meta: { writer: { title: 'Research Buddy' } } })

  assert.equal(botMentionTag({ name: 'writer' }), 'research-buddy')
  assert.equal(botMentionTag({ name: 'ops' }), 'ops')
  assert.equal(botMentionTag({ name: 'default' }), 'hermes')
  // display_name (hermes profile rename) drives the tag too.
  assert.equal(botMentionTag({ name: 'scout', display_name: 'Deal Finder' }), 'deal-finder')
})

test('resolveRosterMentions: renamed bots resolve by friendly tag AND old handle', () => {
  const { resolveRosterMentions } = runtime({ meta: { writer: { title: 'Research Buddy' } } })
  const roster = [{ name: 'default' }, { name: 'writer' }, { name: 'scout', display_name: 'Deal Finder' }]
  const live = { name: 'default', connectionId: 'local' }

  const byTitle = resolveRosterMentions('hey @research-buddy check this', roster, live)
  assert.deepEqual(JSON.parse(JSON.stringify(byTitle.map(b => b.name))), ['writer'])

  const byDisplayName = resolveRosterMentions('ping @dealfinder please', roster, live)
  assert.deepEqual(JSON.parse(JSON.stringify(byDisplayName.map(b => b.name))), ['scout'])

  // The profile name keeps working after a rename.
  const byName = resolveRosterMentions('hey @writer', roster, live)
  assert.deepEqual(JSON.parse(JSON.stringify(byName.map(b => b.name))), ['writer'])
})

test('parseGroupChatMentions: display_name and ui_meta title forms address a member', () => {
  const { parseGroupChatMentions, groupMemberKey } = runtime()
  const members = [
    { name: 'research', title: '' },
    { name: 'builder', title: '', display_name: 'Site Smith' },
    { name: 'ops', title: '', ui_meta: { 'hermes-bots': { title: 'Night Watch' } } }
  ]

  const bySlug = parseGroupChatMentions('@site-smith and @night-watch please', members)
  assert.equal(bySlug.mentioned.has(groupMemberKey(members[1])), true)
  assert.equal(bySlug.mentioned.has(groupMemberKey(members[2])), true)
  assert.equal(bySlug.mentioned.has(groupMemberKey(members[0])), false)

  const collapsed = parseGroupChatMentions('@sitesmith take a look', members)
  assert.equal(collapsed.mentioned.has(groupMemberKey(members[1])), true)
})

test('durableGroupChatMembers persists the friendly identity for renamed-tag mentions', () => {
  const { durableGroupChatMembers, $botMeta } = runtime()
  $botMeta.set({ writer: { title: 'Research Buddy' } })

  const [stored] = durableGroupChatMembers([{ name: 'writer', connectionId: 'local' }])
  assert.equal(stored.title, 'Research Buddy')

  const [remote] = durableGroupChatMembers([
    { name: 'scout', display_name: 'Deal Finder', connectionId: 'mac-mini', remoteSource: true }
  ])
  assert.equal(remote.display_name, 'Deal Finder')
})

test('composer autocomplete offers the renamed tag and matches on the display name', () => {
  const registrations = []
  const atom = value => {
    let current = value
    return { get: () => current, set: next => (current = next), listen: () => () => undefined }
  }
  const jsx = (type, props = {}) => ({ type, props })
  const roster = { profiles: [{ name: 'default' }, { name: 'writer' }, { name: 'ops' }] }
  const context = {
    atom,
    jsx,
    jsxs: jsx,
    useQuery: () => ({}),
    useValue: v => (v?.get ? v.get() : v),
    useState: v => [v, () => undefined],
    setTimeout,
    clearTimeout,
    performance,
    window: {
      setTimeout,
      clearTimeout,
      performance,
      requestAnimationFrame: () => 0,
      cancelAnimationFrame: () => undefined,
      addEventListener: () => undefined,
      removeEventListener: () => undefined
    },
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } },
    queryClient: { getQueryData: () => roster, invalidateQueries: () => undefined, setQueryData: () => undefined },
    host: {
      state: { profile: { get: () => 'default', listen: () => () => undefined }, gateway: { get: () => null, listen: () => () => undefined } },
      request: async () => ({}),
      onEvent: () => () => undefined
    },
    COMPOSER_AREAS: { middleware: 'composer.middleware', atCompletions: 'composer.atCompletions' },
    sdk: new Proxy({}, { get: () => undefined })
  }
  const code = source
    .replace(/^import \* as sdk from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat('\nglobalThis.__botMeta = $botMeta;')
  vm.runInNewContext(code, context)
  context.__botMeta.set({ writer: { title: 'Research Buddy' } })

  try {
    context.plugin.register({
      register: c => registrations.push(c),
      storage: { get: async () => undefined, set: async () => undefined }
    })
  } catch {
    /* registration walks UI surfaces the stub doesn't fully model */
  }

  const provide = registrations.find(c => c.area === 'composer.atCompletions').data.provide

  // The renamed bot completes under its friendly prefix and inserts the tag.
  const renamed = provide('rese')
  assert.deepEqual(JSON.parse(JSON.stringify(renamed.map(i => i.insert))), ['@research-buddy'])

  // Old profile-handle muscle memory still finds it (insert is the new tag).
  const byHandle = provide('wri')
  assert.deepEqual(JSON.parse(JSON.stringify(byHandle.map(i => i.insert))), ['@research-buddy'])

  // Un-renamed bots keep their plain handle.
  const plain = provide('op')
  assert.deepEqual(JSON.parse(JSON.stringify(plain.map(i => i.insert))), ['@ops'])
})
