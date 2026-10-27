import assert from 'node:assert/strict'
import { execFile, execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { afterEach, test } from 'vitest'

import { resolveDesktopHermesHome } from './data-paths'
import { readUiPreferences, uiPreferencesPath, writeUiPreferences } from './ui-preferences'

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

test('native Desktop and the Webapp server resolve the same store from a profile home', () => {
  const home = scratchHome()
  const root = path.join(home, 'custom-root')
  const profileHome = path.join(root, 'profiles', 'coder')
  fs.mkdirSync(profileHome, { recursive: true })

  const electronFile = uiPreferencesPath(resolveDesktopHermesHome({ env: { HERMES_HOME: profileHome }, home }))

  const pythonFile = execFileSync(python, ['-c', PY_PATH], {
    cwd: repo,
    encoding: 'utf8',
    env: pythonEnv(home, profileHome)
  }).trim()

  assert.equal(electronFile, path.join(root, 'ui-preferences.json'))
  assert.equal(pythonFile, electronFile)
})

test('a custom home inside the default home resolves the same store on both sides', () => {
  const home = scratchHome()
  const sandbox = path.join(home, '.hermes', 'sandbox')
  fs.mkdirSync(sandbox, { recursive: true })

  const electronFile = uiPreferencesPath(resolveDesktopHermesHome({ env: { HERMES_HOME: sandbox }, home }))

  const pythonFile = execFileSync(python, ['-c', PY_PATH], {
    cwd: repo,
    encoding: 'utf8',
    env: pythonEnv(home, sandbox)
  }).trim()

  assert.equal(electronFile, path.join(sandbox, 'ui-preferences.json'))
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

test('the one-time localStorage import never overwrites a store that exists', async () => {
  const root = scratchHome()
  const file = uiPreferencesPath(root)

  await writeUiPreferences(file, { values: { 'hermes-desktop-mode-v1': 'dark' } })
  const seeded = await writeUiPreferences(file, { seed: true, values: { 'hermes-desktop-mode-v1': 'light' } })

  assert.deepEqual(seeded, { exists: true, revision: 1, values: { 'hermes-desktop-mode-v1': 'dark' } })
  assert.deepEqual(readUiPreferences(file).values, { 'hermes-desktop-mode-v1': 'dark' })
})
