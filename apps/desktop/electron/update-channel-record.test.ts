/**
 * update-channel-record.test.ts
 *
 * Contract tests for the install-scoped update channel record shared with
 * the CLI updater (hermes_cli/update_channel.py):
 *   - path mirrors the CLI's get_default_hermes_root() profile unwrapping
 *   - missing/invalid reads as stable, and a read never writes
 *   - only explicit selection persists; the write is atomic (no .tmp litter)
 *   - profile-scoped HERMES_HOME values resolve to ONE shared install record
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterEach, beforeEach, describe, it } from 'vitest'

import {
  parseUpdateChannelRecord,
  readUpdateChannel,
  updateChannelRecordPath,
  writeUpdateChannel,
  UPDATE_CHANNEL_RECORD_FILENAME
} from './update-channel-record'

function tempRoot(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-channel-record-'))
}

function makePathModule(root: string) {
  return {
    ...path,
    resolve: (...parts: string[]) => path.resolve(root, ...parts),
    dirname: (value: string) => path.dirname(value),
    basename: (value: string) => path.basename(value),
    join: (...parts: string[]) => path.join(...parts)
  }
}

describe('updateChannelRecordPath', () => {
  it('lives at the install root, not in a profile directory', () => {
    const root = tempRoot()
    const pathModule = makePathModule(root)

    const direct = updateChannelRecordPath(path.join(root, 'hermes'), { pathModule })
    const profile = updateChannelRecordPath(path.join(root, 'hermes', 'profiles', 'work'), { pathModule })

    assert.equal(direct, path.join(root, 'hermes', UPDATE_CHANNEL_RECORD_FILENAME))
    assert.equal(profile, direct, 'a profile home must resolve to the SAME install record')
  })
})

describe('readUpdateChannel', () => {
  let root: string
  let home: string

  beforeEach(() => {
    root = tempRoot()
    home = path.join(root, 'hermes')
    fs.mkdirSync(home, { recursive: true })
  })

  afterEach(() => {
    fs.rmSync(root, { force: true, recursive: true })
  })

  it('missing record reads stable and never writes', () => {
    assert.equal(readUpdateChannel(home), 'stable')
    assert.equal(fs.existsSync(path.join(home, UPDATE_CHANNEL_RECORD_FILENAME)), false)
  })

  it('malformed record reads stable', () => {
    fs.writeFileSync(path.join(home, UPDATE_CHANNEL_RECORD_FILENAME), '{not json', 'utf8')
    assert.equal(readUpdateChannel(home), 'stable')
  })

  it('invalid channel value reads stable', () => {
    fs.writeFileSync(
      path.join(home, UPDATE_CHANNEL_RECORD_FILENAME),
      JSON.stringify({ schema_version: 1, channel: 'rc' }),
      'utf8'
    )
    assert.equal(readUpdateChannel(home), 'stable')
  })

  it('beta round-trips', () => {
    writeUpdateChannel(home, 'beta')
    assert.equal(readUpdateChannel(home), 'beta')
  })

  it('unknown write input normalizes to stable', () => {
    const record = writeUpdateChannel(home, 'nonsense' as 'stable' | 'beta')
    assert.equal(record.channel, 'stable')
  })
})

describe('writeUpdateChannel', () => {
  let root: string
  let home: string

  beforeEach(() => {
    root = tempRoot()
    home = path.join(root, 'hermes')
    fs.mkdirSync(home, { recursive: true })
  })

  afterEach(() => {
    fs.rmSync(root, { force: true, recursive: true })
  })

  it('persists the schema the CLI reads', () => {
    writeUpdateChannel(home, 'beta')
    const raw = JSON.parse(fs.readFileSync(path.join(home, UPDATE_CHANNEL_RECORD_FILENAME), 'utf8'))

    assert.deepEqual(raw, { schema_version: 1, channel: 'beta' })
  })

  it('leaves no torn temp file behind', () => {
    writeUpdateChannel(home, 'stable')
    assert.equal(fs.existsSync(path.join(home, `${UPDATE_CHANNEL_RECORD_FILENAME}.tmp`)), false)
  })

  it('overwrites a previous selection atomically', () => {
    writeUpdateChannel(home, 'beta')
    writeUpdateChannel(home, 'stable')

    assert.equal(readUpdateChannel(home), 'stable')
  })
})

describe('parseUpdateChannelRecord', () => {
  it('accepts the documented schema', () => {
    assert.deepEqual(parseUpdateChannelRecord('{"schema_version":1,"channel":"stable"}'), {
      schemaVersion: 1,
      channel: 'stable'
    })
  })

  it('rejects anything that is not stable or beta', () => {
    assert.equal(parseUpdateChannelRecord('{"schema_version":1,"channel":"main"}'), null)
    assert.equal(parseUpdateChannelRecord('[]'), null)
    assert.equal(parseUpdateChannelRecord('nope'), null)
  })
})
