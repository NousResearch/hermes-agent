import assert from 'node:assert/strict'
import { execFile, execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { afterEach, test } from 'vitest'

import { resolveDesktopHermesHome } from './data-paths'
import { readUiPreferences, uiPreferencesPath, uiPreferencesRoot, writeUiPreferences } from './ui-preferences'

// Native Desktop (this module) and the Webapp's Python server must land on ONE
// file and merge into it without losing each other's keys (#145213).
const repo = path.resolve(import.meta.dirname, '../../..')
const python = process.env.HERMES_PYTHON || 'python3'
const homes: string[] = []

afterEach(() => {
  for (const home of homes.splice(0)) {
    fs.rmSync(home, { force: true, recursive: true })
  }
})

function scratchHome(): string {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-ui-prefs-'))
  homes.push(home)

  return home
}

function pythonEnv(home: string, hermesHome: string): NodeJS.ProcessEnv {
  return { HERMES_HOME: hermesHome, HOME: home, PATH: process.env.PATH, PYTHONPATH: repo }
}

const PY_PATH = 'from hermes_cli.ui_preferences import store_path; print(store_path())'

/** The file main's IPC registers for this HERMES_HOME (registerUiPreferencesIpc). */
const electronStorePath = (home: string, hermesHome: string) =>
  uiPreferencesPath(uiPreferencesRoot(resolveDesktopHermesHome({ env: { HERMES_HOME: hermesHome }, home })))

test('native Desktop and the Webapp server resolve the same store from a profile home', () => {
  const home = scratchHome()
  const root = path.join(home, 'custom-root')
  const profileHome = path.join(root, 'profiles', 'coder')
  fs.mkdirSync(profileHome, { recursive: true })

  const electronFile = electronStorePath(home, profileHome)

  const pythonFile = execFileSync(python, ['-c', PY_PATH], {
    cwd: repo,
    encoding: 'utf8',
    env: pythonEnv(home, profileHome)
  }).trim()

  assert.equal(electronFile, path.join(fs.realpathSync(root), 'ui-preferences.json'))
  assert.equal(pythonFile, electronFile)
})

test.skipIf(process.platform === 'win32')(
  'a symlinked profile directory still resolves its root, the same store on both sides',
  () => {
    const home = scratchHome()
    const root = path.join(home, '.hermes')
    const elsewhere = path.join(home, 'elsewhere', 'coder')
    fs.mkdirSync(path.join(root, 'profiles'), { recursive: true })
    fs.mkdirSync(elsewhere, { recursive: true })
    fs.symlinkSync(elsewhere, path.join(root, 'profiles', 'coder'))
    const profileHome = path.join(root, 'profiles', 'coder')

    const pythonFile = execFileSync(python, ['-c', PY_PATH], {
      cwd: repo,
      encoding: 'utf8',
      env: pythonEnv(home, profileHome)
    }).trim()

    assert.equal(electronStorePath(home, profileHome), path.join(fs.realpathSync(root), 'ui-preferences.json'))
    assert.equal(pythonFile, electronStorePath(home, profileHome))
  }
)

test('a custom home inside the default home resolves the same store on both sides', () => {
  const home = scratchHome()
  const sandbox = path.join(home, '.hermes', 'sandbox')
  fs.mkdirSync(sandbox, { recursive: true })

  const electronFile = electronStorePath(home, sandbox)

  const pythonFile = execFileSync(python, ['-c', PY_PATH], {
    cwd: repo,
    encoding: 'utf8',
    env: pythonEnv(home, sandbox)
  }).trim()

  assert.equal(electronFile, path.join(fs.realpathSync(sandbox), 'ui-preferences.json'))
  assert.equal(fs.realpathSync(path.dirname(pythonFile)), fs.realpathSync(path.dirname(electronFile)))
})

test('concurrent native and Webapp writers keep every key the other wrote', async () => {
  const home = scratchHome()
  const root = path.join(home, 'root')
  fs.mkdirSync(root)
  const file = uiPreferencesPath(root)

  const pyWrites = `
from hermes_cli.ui_preferences import write_preferences
for n in range(25):
    write_preferences({f"hermes.desktop.py{n}": "web"})
`

  const pythonRun = promisify(execFile)(python, ['-c', pyWrites], { cwd: repo, env: pythonEnv(home, root) })

  for (let n = 0; n < 25; n += 1) {
    await writeUiPreferences(file, { values: { [`hermes.desktop.native${n}`]: 'native' } })
  }

  await pythonRun

  const { revision, values } = readUiPreferences(file)

  assert.equal(Object.keys(values).length, 50)
  assert.equal(revision, 50)

  for (let n = 0; n < 25; n += 1) {
    assert.equal(values[`hermes.desktop.native${n}`], 'native')
    assert.equal(values[`hermes.desktop.py${n}`], 'web')
  }

  assert.equal(fs.existsSync(`${file}.lock`), false)
})

test('a client’s localStorage import only adds the keys a store lacks', async () => {
  const root = scratchHome()
  const file = uiPreferencesPath(root)

  await writeUiPreferences(file, { values: { 'hermes-desktop-mode-v1': 'dark' } })

  const seeded = await writeUiPreferences(file, {
    seed: true,
    values: { 'hermes-desktop-mode-v1': 'light', 'hermes-desktop-theme-v2': 'nous' }
  })

  assert.deepEqual(seeded.values, { 'hermes-desktop-mode-v1': 'dark', 'hermes-desktop-theme-v2': 'nous' })
  assert.deepEqual(readUiPreferences(file).values, seeded.values)
})

// Only a missing file is "no store": merging over a store that exists but
// cannot be read (EACCES, a Windows AV lock, EMFILE) dropped every other key.
test.skipIf(process.platform === 'win32' || process.getuid?.() === 0)(
  'neither writer merges over a store it cannot read',
  async () => {
    const home = scratchHome()
    const file = uiPreferencesPath(home)
    await writeUiPreferences(file, { values: { 'hermes-desktop-theme-v2': 'nous', 'hermes.desktop.keybinds': '{}' } })
    const before = fs.readFileSync(file, 'utf8')
    fs.chmodSync(file, 0)

    try {
      await assert.rejects(writeUiPreferences(file, { values: { 'hermes-desktop-mode-v1': 'dark' } }), /unreadable/)

      const pythonWrite = promisify(execFile)(
        python,
        ['-c', 'from hermes_cli.ui_preferences import write_preferences as w; w({"hermes-desktop-mode-v1": "dark"})'],
        { cwd: repo, env: pythonEnv(home, home) }
      )

      await assert.rejects(pythonWrite, /PreferencesBusy: preferences store is unreadable/)
    } finally {
      fs.chmodSync(file, 0o600)
    }

    assert.equal(fs.readFileSync(file, 'utf8'), before)
  }
)
