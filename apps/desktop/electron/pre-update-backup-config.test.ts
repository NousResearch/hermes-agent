import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, it } from 'vitest'

import { preUpdateBackupEnabled, readPreUpdateBackupEnabled } from './pre-update-backup-config'

const temporaryDirectories: string[] = []

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    fs.rmSync(directory, { recursive: true, force: true })
  }
})

describe('preUpdateBackupEnabled', () => {
  it.each([false, null, 'off', 'false', 'none', 'disabled', ' DISABLED '])(
    'disables the desktop backup for the Python updater off alias %j',
    value => {
      expect(preUpdateBackupEnabled({ updates: { pre_update_backup: value } })).toBe(false)
    }
  )

  it.each([true, 'quick', 'full', 'zip', 'true', 'unexpected', 0, {}, undefined])(
    'keeps the safety backup for %j',
    value => {
      expect(preUpdateBackupEnabled({ updates: { pre_update_backup: value } })).toBe(true)
    }
  )

  it('keeps the safety backup when the updates section is absent', () => {
    expect(preUpdateBackupEnabled({ model: { default: 'test' } })).toBe(true)
  })
})

describe('readPreUpdateBackupEnabled', () => {
  it('reads the nested setting from config.yaml', () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-pre-update-backup-'))
    const configPath = path.join(directory, 'config.yaml')

    temporaryDirectories.push(directory)
    fs.writeFileSync(configPath, 'updates:\n  pre_update_backup: false\n')

    expect(readPreUpdateBackupEnabled(configPath)).toBe(false)
  })

  it.each(['updates: [', 'missing-config.yaml'])('fails safe for an unreadable configuration: %s', contents => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-pre-update-backup-'))
    const configPath = path.join(directory, 'config.yaml')

    temporaryDirectories.push(directory)

    if (contents !== 'missing-config.yaml') {
      fs.writeFileSync(configPath, contents)
    }

    expect(readPreUpdateBackupEnabled(configPath)).toBe(true)
  })
})