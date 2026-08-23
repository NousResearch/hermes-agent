import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { existsSync, readFileSync, rmSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function load(request = async () => ({ jobs: [] })) {
  const values = new Map()
  const atom = initial => {
    const slot = { get: () => values.get(slot), set: value => values.set(slot, value) }
    values.set(slot, initial)
    return slot
  }
  const context = {
    atom, PALETTE_AREA: 'palette', COMPOSER_AREAS: { middleware: 'middleware' },
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } },
    host: {
      request,
      state: {
        profile: { listen: () => undefined },
        gateway: { listen: () => undefined }
      }
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
      '\nglobalThis.__routines = { isLegacyDelegatedRoutine, loadRoutines, normalizedProfileName, routineInputError, routinePrompt };\n'
    )
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  return context
}

test('unit: direct execution is selected for the active bot profile', () => {
  const { __routines } = load()
  assert.equal(__routines.normalizedProfileName(' Default '), 'default')
  assert.equal(__routines.routinePrompt('default', 'Health', 'Collect status', ' DEFAULT '), 'Collect status')
})

test('integration: a different active profile retains the delegated routine wrapper', () => {
  const { __routines } = load()
  const prompt = __routines.routinePrompt('research', 'Digest', 'Summarize findings', 'default')
  assert.match(prompt, /hermes -p 'research' chat/)
  assert.match(prompt, /\[Scheduled routine\] Summarize findings/)
})

test('security: delegated routine arguments remain literal shell values', () => {
  const { __routines } = load()
  const title = 'Audit $(printf TITLE_EXPANDED) `printf TITLE_TICK` \'quoted\''
  const instruction = 'Line one $(printf TASK_EXPANDED) `printf TASK_TICK`\nLine two \'quoted\''
  const prompt = __routines.routinePrompt('research', title, instruction, 'default')
  const command = prompt.slice(prompt.indexOf('hermes '), prompt.lastIndexOf('\n\nIf the command'))
  const script = `hermes() { printf '%s\\037' "$@"; }\n${command}`
  const result = spawnSync('sh', ['-c', script], { encoding: 'utf8' })

  assert.equal(result.status, 0, result.stderr)
  assert.deepEqual(result.stdout.split('\x1f').slice(0, -1), [
    '-p',
    'research',
    'chat',
    '-c',
    `Routine: ${title}`,
    '-q',
    `[Scheduled routine] ${instruction}`
  ])
})

test('security: upgrade pauses persisted delegated routines before they can execute', async () => {
  const titleSentinel = `/tmp/hermes-bot-mode-title-${process.pid}`
  const taskSentinel = `/tmp/hermes-bot-mode-task-${process.pid}`
  rmSync(titleSentinel, { force: true })
  rmSync(taskSentinel, { force: true })

  const title = `Audit $(touch ${titleSentinel})`
  const instruction = `Inspect $(touch ${taskSentinel})`
  const oldPrompt =
    `You are running the scheduled routine "${title}" for agent 'research'. ` +
    `Execute it AS that agent so the run lands in its own history: run this in the terminal and relay the output:\n\n` +
    `hermes -p research chat -c "Routine: ${title}" -q ${JSON.stringify(`[Scheduled routine] ${instruction}`)}\n\n` +
    `If the command fails, report the error instead.`
  const persisted = JSON.parse(
    JSON.stringify({
      job_id: 'legacy-job',
      name: '[bot:research] Audit',
      prompt: oldPrompt,
      prompt_preview: oldPrompt.slice(0, 100),
      schedule: 'every 2h',
      enabled: true,
      state: 'scheduled',
      repeat: { times: 3, completed: 1 }
    })
  )
  const requests = []
  const runtime = load(async (method, params) => {
    requests.push([method, params])

    if (params.action === 'list') {
      return { jobs: [persisted] }
    }

    if (params.action === 'pause' && params.name === persisted.job_id) {
      persisted.enabled = false
      persisted.state = 'paused'
      return { success: true }
    }

    throw new Error(`Unexpected request: ${method} ${JSON.stringify(params)}`)
  })

  const result = await runtime.__routines.loadRoutines()
  assert.deepEqual(JSON.parse(JSON.stringify(requests)), [
    ['cron.manage', { action: 'list', include_disabled: true }],
    ['cron.manage', { action: 'pause', name: 'legacy-job' }]
  ])
  assert.equal(result.jobs[0].job_id, 'legacy-job')
  assert.equal(result.jobs[0].schedule, 'every 2h')
  assert.deepEqual(result.jobs[0].repeat, { times: 3, completed: 1 })
  assert.equal(result.jobs[0].enabled, false)
  assert.equal(result.jobs[0].state, 'paused')
  assert.equal(existsSync(titleSentinel), false)
  assert.equal(existsSync(taskSentinel), false)

  await runtime.__routines.loadRoutines()
  assert.deepEqual(JSON.parse(JSON.stringify(requests.slice(2))), [
    ['cron.manage', { action: 'list', include_disabled: true }]
  ])
  assert.match(pluginSource, /disabled: busy \|\| legacyUnsafe/)
  assert.match(pluginSource, /Paused for security: delete and recreate this legacy cronjob/)

  const recreated = runtime.__routines.routinePrompt('research', title, instruction, 'default')
  assert.equal(runtime.__routines.isLegacyDelegatedRoutine({ ...persisted, prompt_preview: recreated.slice(0, 100) }), false)
  const command = recreated.slice(recreated.indexOf('hermes '), recreated.lastIndexOf('\n\nIf the command'))
  const script = `hermes() { printf '%s\\037' "$@"; }\n${command}`
  const executed = spawnSync('sh', ['-c', script], { encoding: 'utf8' })
  assert.equal(executed.status, 0, executed.stderr)
  assert.deepEqual(executed.stdout.split('\x1f').slice(0, -1), [
    '-p',
    'research',
    'chat',
    '-c',
    `Routine: ${title}`,
    '-q',
    `[Scheduled routine] ${instruction}`
  ])
  assert.equal(existsSync(titleSentinel), false)
  assert.equal(existsSync(taskSentinel), false)
})

test('robustness: routine input rejects NUL before cron creation', () => {
  const { __routines } = load()
  assert.equal(__routines.routineInputError('Normal title', 'Normal instruction'), null)
  assert.match(__routines.routineInputError('Bad\0title', 'Normal instruction'), /NUL.*U\+0000/)
  assert.match(__routines.routineInputError('Normal title', 'Bad\0instruction'), /NUL.*U\+0000/)
  assert.match(
    pluginSource,
    /const inputError = routineInputError\(title, task\)[\s\S]*if \(inputError\)[\s\S]*setError\(inputError\)[\s\S]*return[\s\S]*requestForBot\(bot, 'cron\.manage'/
  )
})

test('regression: Create Cronjob passes the active profile to routinePrompt', () => {
  assert.match(pluginSource, /prompt: routinePrompt\(profile, title, task, activeProfile\)/)
  const { __routines } = load()
  const instruction = 'Keep "quoted" output intact'
  assert.equal(__routines.routinePrompt('ops', 'Check', instruction, 'ops'), instruction)
  assert.doesNotMatch(__routines.routinePrompt('ops', 'Check', instruction, 'ops'), /hermes -p/)
})

test('system: the direct-file plugin still registers its panes', () => {
  const runtime = load()
  const registered = []
  runtime.plugin.register({ storage: { get: () => null }, register: entry => registered.push(entry) })
  assert.equal(registered.some(entry => entry.id === 'pane'), true)
  assert.equal(registered.some(entry => entry.id === 'routines'), true)
})

test('performance: direct prompt selection remains bounded', t => {
  const { __routines } = load()
  const start = Date.now()
  for (let index = 0; index < 10000; index += 1) __routines.routinePrompt('ops', 'Check', 'Inspect', 'ops')
  const elapsed = Date.now() - start
  t.diagnostic(`10,000 direct prompt selections: ${elapsed} ms`)
  assert.ok(elapsed < 1000)
})