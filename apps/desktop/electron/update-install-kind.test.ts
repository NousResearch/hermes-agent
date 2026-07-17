import assert from 'node:assert/strict'

import { test } from 'vitest'

import { classifyWindowsManualUpdate } from './update-install-kind'

const REPO_ROOT = 'C:\\Users\\u\\AppData\\Local\\hermes\\hermes-agent'
const REPO_EXE = `${REPO_ROOT}\\apps\\desktop\\release\\win-unpacked\\Hermes.exe`
const NSIS_EXE = 'C:\\Users\\u\\AppData\\Local\\Programs\\Hermes\\Hermes.exe'

test('packaged exe outside the managed checkout is an installer deployment (#66095)', () => {
  assert.equal(
    classifyWindowsManualUpdate({
      execPath: NSIS_EXE,
      updateRoot: REPO_ROOT,
      isPackaged: true,
      hasInstallerLog: false
    }),
    'installer'
  )
})

test('packaged exe inside the managed checkout is a CLI install', () => {
  assert.equal(
    classifyWindowsManualUpdate({
      execPath: REPO_EXE,
      updateRoot: REPO_ROOT,
      isPackaged: true,
      hasInstallerLog: false
    }),
    'cli'
  )
})

test('Hermes-Setup breadcrumb wins even when the exe lives inside the checkout', () => {
  // The Tauri installer points shortcuts at the repo-tree packed exe, so when
  // its best-effort self-copy failed the exe location alone looks like a CLI
  // install. The installer log it always writes is the tiebreaker.
  assert.equal(
    classifyWindowsManualUpdate({
      execPath: REPO_EXE,
      updateRoot: REPO_ROOT,
      isPackaged: true,
      hasInstallerLog: true
    }),
    'installer'
  )
})

test('path containment is case-insensitive and separator-agnostic on Windows', () => {
  assert.equal(
    classifyWindowsManualUpdate({
      execPath: 'c:/users/U/appdata/local/HERMES/hermes-agent/apps/desktop/release/win-unpacked/Hermes.exe',
      updateRoot: REPO_ROOT,
      isPackaged: true,
      hasInstallerLog: false
    }),
    'cli'
  )
})

test('a sibling directory sharing the checkout prefix is NOT inside it', () => {
  // "...\hermes-agent-nightly" starts with the "...\hermes-agent" string but is
  // a different directory; a naive startsWith would misclassify it as CLI.
  assert.equal(
    classifyWindowsManualUpdate({
      execPath: `${REPO_ROOT}-nightly\\Hermes.exe`,
      updateRoot: REPO_ROOT,
      isPackaged: true,
      hasInstallerLog: false
    }),
    'installer'
  )
})

test('unpackaged (dev) runs stay on the CLI message', () => {
  // `npm run dev` launches node_modules electron.exe, which is outside any
  // checkout; without the isPackaged gate every dev run would misclassify.
  assert.equal(
    classifyWindowsManualUpdate({
      execPath: 'C:\\src\\hermes-agent\\node_modules\\electron\\dist\\electron.exe',
      updateRoot: 'C:\\src\\hermes-agent',
      isPackaged: false,
      hasInstallerLog: false
    }),
    'cli'
  )
})

test('missing exec path or update root degrades to the CLI message', () => {
  assert.equal(
    classifyWindowsManualUpdate({ execPath: '', updateRoot: REPO_ROOT, isPackaged: true, hasInstallerLog: false }),
    'cli'
  )
  assert.equal(
    classifyWindowsManualUpdate({ execPath: NSIS_EXE, updateRoot: '', isPackaged: true, hasInstallerLog: false }),
    'cli'
  )
})
