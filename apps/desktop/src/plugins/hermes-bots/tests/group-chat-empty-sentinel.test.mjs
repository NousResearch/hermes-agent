import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

/** Load the plugin in a vm with a scripted cli.exec so member turns are
 *  deterministic. `turnScript(profile, prompt)` returns the member's reply
 *  text (or throws to simulate a failed turn). */
function load(turnScript, { busyUntilResumeCall, clarifyUntilResumeCall, approvalUntilResumeCall, conflictOnce = false, deferredTimers = false } = {}) {
  const values = new Map()
  const atom = initial => {
    const slot = { get: () => values.get(slot), set: value => {
      if (process.env.TRACE_ATOM && value && typeof value === 'object' && value.Grind) {
        console.error('ATOM SET Grind stranded=', JSON.stringify(value.Grind.stranded), new Error().stack.split('\n').slice(2,5).join(' | '))
      }
      values.set(slot, value)
    } }
    values.set(slot, initial)
    return slot
  }
  const calls = []
  const clarifyResponds = []
  const approvalResponds = []
  const requests = []
  const sessions = new Map()
  const runtimeToStored = new Map()
  const titleToStored = new Map()
  const sharedUiMeta = {}
  const uiMetaRevisions = {}
  let sessionSequence = 0
  const resumeCallCounts = new Map()
  let injectedConflict = false

  const resolveSession = (profile, target) => {
    const stored = runtimeToStored.get(target) || (sessions.has(target) ? target : titleToStored.get(`${profile}::${target}`))
    return stored ? sessions.get(stored) : null
  }
  const context = {
    atom,
    setTimeout: fn => {
      if (deferredTimers) {
        setImmediate(fn)
        return 1
      }
      fn()
      return 0
    },
    clearTimeout: () => undefined,
    PALETTE_AREA: 'palette',
    COMPOSER_AREAS: { middleware: 'middleware' },
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } },
    host: {
      request: async (method, params) => {
        requests.push({ method, params })
        if (method === 'profiles.list') {
          return {
            profiles: [
              {
                name: 'default',
                ui_meta: { ...sharedUiMeta },
                ui_meta_revisions: { ...uiMetaRevisions }
              }
            ]
          }
        }
        if (method === 'profiles.configure') {
          if (conflictOnce && !injectedConflict) {
            injectedConflict = true
            sharedUiMeta['hermes-bots-groups'] = {
              version: 2,
              rooms: {
                Shared: {
                  revision: 1,
                  log: [{ id: 'writer-a:1', from: { kind: 'user', name: 'You' }, text: 'alpha', at: 1 }],
                  members: [{ name: 'alpha' }]
                }
              }
            }
            uiMetaRevisions['hermes-bots-groups'] = 1
            return {
              applied: {
                ui_meta: false,
                ui_meta_conflicts: { 'hermes-bots-groups': { expected: 0, actual: 1 } },
                ui_meta_revisions: { 'hermes-bots-groups': 1 }
              }
            }
          }
          const expected = params.ui_meta_expected_revisions || null
          if (expected) {
            for (const key of Object.keys(params.ui_meta || {})) {
              if ((uiMetaRevisions[key] || 0) !== expected[key]) {
                return {
                  applied: {
                    ui_meta: false,
                    ui_meta_conflicts: {
                      [key]: { expected: expected[key], actual: uiMetaRevisions[key] || 0 }
                    }
                  }
                }
              }
            }
          }
          for (const [key, value] of Object.entries(params.ui_meta || {})) {
            if (value === null) {
              delete sharedUiMeta[key]
            } else {
              sharedUiMeta[key] = value
            }
            uiMetaRevisions[key] = (uiMetaRevisions[key] || 0) + 1
          }
          return { applied: { ui_meta: true, ui_meta_revisions: { ...uiMetaRevisions } } }
        }
        if (method === 'session.create') {
          sessionSequence += 1
          const stored = `sid-${params.profile}-${sessionSequence}`
          const runtime = `rt-${params.profile}-${sessionSequence}`
          const session = { stored, runtime, profile: params.profile, title: params.title, messages: [] }
          sessions.set(stored, session)
          runtimeToStored.set(runtime, stored)
          titleToStored.set(`${params.profile}::${params.title}`, stored)
          return { session_id: runtime, stored_session_id: stored, message_count: 0, messages: [] }
        }
        if (method === 'session.resume') {
          const session = resolveSession(params.profile, params.session_id)
          if (!session) {
            const err = new Error(`session not found: ${params.session_id}`)
            err.code = 4007
            throw err
          }
          const profile = params.profile
          const seen = (resumeCallCounts.get(profile) || 0) + 1
          resumeCallCounts.set(profile, seen)
          const limit = busyUntilResumeCall && busyUntilResumeCall[profile]
          const busy = Boolean(limit && seen <= limit)
          const clarify = clarifyUntilResumeCall && clarifyUntilResumeCall[profile]
          const pendingClarify = clarify && seen <= clarify.until ? clarify.payload : null
          const approval = approvalUntilResumeCall && approvalUntilResumeCall[profile]
          const pendingApproval = approval && seen <= approval.until ? approval.payload : null
          return {
            session_id: session.runtime,
            session_key: session.stored,
            message_count: session.messages.length,
            messages: [...session.messages],
            inflight: busy,
            running: busy,
            ...(pendingClarify ? { pending_clarify: pendingClarify } : {}),
            ...(pendingApproval ? { pending_approval: pendingApproval } : {})
          }
        }
        if (method === 'prompt.submit') {
          const session = resolveSession(null, params.session_id)
          if (!session) {
            throw new Error(`runtime session not found: ${params.session_id}`)
          }
          session.messages.push({ role: 'user', content: params.text })
          calls.push({
            profile: session.profile,
            prompt: params.text,
            runtime: session.runtime,
            stored: session.stored,
            title: session.title
          })
          const reply = turnScript(session.profile, params.text, calls.length, session)
          session.messages.push({ role: 'assistant', content: reply })
          return {}
        }
        if (method === 'clarify.respond') {
          clarifyResponds.push({ ...params })
          return { ok: true }
        }
        if (method === 'approval.respond') {
          approvalResponds.push({ ...params })
          return { resolved: true }
        }
        return {}
      },
      state: { profile: { get: () => 'default', listen: () => undefined }, gateway: { listen: () => undefined } },
      notify: () => undefined,
      notifyError: () => undefined
    }
  }
  const source = pluginSource
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat(
      '\nglobalThis.__gc = { sendToGroupChat, runGroupChatRounds, harvestStrandedGroupReply, resolveGroupResponders, parseGroupChatMentions, rotateGroupSpeakers, isGroupPassText, formatGroupChatLine, groupSpeakerLabel, buildGroupChatTurnPrompt, trimGroupChatLog, groupChatSyncSnapshot, groupChatGatewayJsonSize, mergeGroupChatSyncSnapshots, mergeRemoteGroupChatSnapshotIntoRooms, scheduleGroupChatServerSync, disbandGroupChat, renameGroupChat, updateGroupChat, durableGroupChatRooms, persistGroupChatRooms, ensureGroupChatSession, uniqueGroupChatName, liveGroupChatNames, groupChatNames, openGroupChat, closeGroupChatMainTab, shouldRenderGroupChatInPane, syncGroupClarify, clearGroupClarify, answerGroupClarify, $groupClarify, $groupChats, $groupNeedsYou, $groupChatWorkspace, $groupMainTabsRev, $botMeta, $lastRoster, GROUP_CHAT_MAX_ROUNDS, GROUP_CHAT_MAX_MESSAGES };\n'
    )
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  const storageWrites = new Map()
  context.plugin.register({
    storage: { get: () => null, set: (key, value) => storageWrites.set(key, value) },
    register: () => undefined
  })
  return { ...context.__gc, approvalResponds, calls, clarifyResponds, host: context.host, requests, sessions, storageWrites, sharedUiMeta, uiMetaRevisions }
}

function roomLog(gc, group) {
  return (gc.$groupChats.get()[group] || { log: [] }).log
}

const FRIENDLY = '⚠️ The model returned no response after processing tool results. This can happen with some models — try again or rephrase your question.'

async function drain(gc, group) {
  for (let i = 0; i < 200 && (gc.$groupChats.get()[group] || {}).running; i++) {
    await new Promise(resolve => setImmediate(resolve))
  }
}

test('(empty) sentinel: a "(empty)" member reply is converted like the gateway, never appended raw', async () => {
  const gc = load((profile, prompt) => {
    if (profile === 'research') return '(empty)'
    return '(pass)'
  })

  gc.sendToGroupChat('Sentinel', [{ name: 'research', title: '' }], '@research thoughts?')
  await drain(gc, 'Sentinel')

  const memberEntries = roomLog(gc, 'Sentinel').filter(e => e.from.kind === 'member')
  assert.ok(memberEntries.length === 1, `expected exactly 1 member entry, got ${memberEntries.length}`)
  assert.equal(memberEntries[0].text, FRIENDLY)
  assert.equal(memberEntries[0].text.includes('(empty)'), false)
})

test('(empty) sentinel: normal replies are untouched', async () => {
  const gc = load((profile, prompt) => {
    if (profile === 'research') return 'I am not empty.'
    return '(pass)'
  })

  gc.sendToGroupChat('Sentinel2', [{ name: 'research', title: '' }], '@research hi')
  await drain(gc, 'Sentinel2')

  const memberEntries = roomLog(gc, 'Sentinel2').filter(e => e.from.kind === 'member')
  assert.equal(memberEntries.length, 1)
  assert.equal(memberEntries[0].text, 'I am not empty.')
})
