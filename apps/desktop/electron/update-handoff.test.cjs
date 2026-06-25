const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const test = require('node:test')

const {
  buildHermesUpdateArgs,
  buildManualHermesUpdateCommand,
  buildVisibleWindowsUpdaterScript,
  createUpdaterLaunchPlan
} = require('./update-handoff.cjs')

test('builds Hermes CLI update args with a forced pre-update backup', () => {
  assert.deepEqual(buildHermesUpdateArgs({ assumeYes: true }), ['update', '--yes', '--backup'])
  assert.deepEqual(buildHermesUpdateArgs({ assumeYes: true, branch: 'release/x' }), [
    'update',
    '--yes',
    '--backup',
    '--branch',
    'release/x'
  ])
})

test('builds manual update commands with a forced pre-update backup', () => {
  assert.equal(buildManualHermesUpdateCommand(), 'hermes update --backup')
  assert.equal(buildManualHermesUpdateCommand('feature/x'), 'hermes update --backup --branch feature/x')
})

test('builds a visible Windows updater script with human-readable progress', () => {
  const script = buildVisibleWindowsUpdaterScript(String.raw`C:\Program Files\Hermes\hermes-setup.exe`, [
    '--update',
    '--branch',
    'main'
  ])

  assert.match(script, /title Hermes update/)
  assert.match(script, /Closing Hermes runtime processes/)
  assert.match(script, /Running Hermes updater/)
  assert.match(script, /Relaunching Hermes when the updater finishes/)
  assert.match(script, /Update command finished/)
  assert.match(script, /timeout \/t 2 \/nobreak/)
  assert.match(script, /"C:\\Program Files\\Hermes\\hermes-setup\.exe" "--update" "--branch" "main"/)
})

test('creates a cmd.exe launch plan for Windows updater handoff', () => {
  const handoffDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-update-handoff-test-'))
  try {
    const plan = createUpdaterLaunchPlan({
      handoffDir,
      isWindows: true,
      updater: String.raw`C:\Program Files\Hermes\hermes-setup.exe`,
      updaterArgs: ['--update', '--branch', 'main']
    })

    assert.equal(plan.command, 'cmd.exe')
    assert.deepEqual(plan.args.slice(0, 3), ['/d', '/s', '/c'])
    assert.equal(plan.detached, true)
    assert.equal(plan.windowsHide, false)
    assert.equal(path.dirname(plan.scriptPath), handoffDir)
    assert.ok(fs.existsSync(plan.scriptPath))
    assert.match(fs.readFileSync(plan.scriptPath, 'utf8'), /Running Hermes updater/)
  } finally {
    fs.rmSync(handoffDir, { recursive: true, force: true })
  }
})

test('keeps non-Windows updater launches direct', () => {
  const plan = createUpdaterLaunchPlan({
    handoffDir: '/tmp',
    isWindows: false,
    updater: '/Applications/Hermes.app/Contents/MacOS/hermes-setup',
    updaterArgs: ['--update', '--branch', 'main']
  })

  assert.equal(plan.command, '/Applications/Hermes.app/Contents/MacOS/hermes-setup')
  assert.deepEqual(plan.args, ['--update', '--branch', 'main'])
  assert.equal(plan.scriptPath, null)
  assert.equal(plan.detached, true)
  assert.equal(plan.windowsHide, false)
})
