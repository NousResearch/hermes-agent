import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, it } from 'vitest'

import { readPreviewUblockSettings, writePreviewUblockSettings } from './preview-ublock-settings'

const tempDirectories: string[] = []

function tempSettingsPath(): string {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-preview-ublock-settings-'))
  tempDirectories.push(directory)
  return path.join(directory, 'nested', 'preview-ublock.json')
}

afterEach(() => {
  for (const directory of tempDirectories.splice(0)) {
    fs.rmSync(directory, { force: true, recursive: true })
  }
})

describe('preview uBlock settings', () => {
  it('defaults missing or malformed settings to disabled', () => {
    const settingsPath = tempSettingsPath()

    expect(readPreviewUblockSettings(settingsPath)).toEqual({ enabled: false })

    fs.mkdirSync(path.dirname(settingsPath), { recursive: true })
    fs.writeFileSync(settingsPath, JSON.stringify({ enabled: 'false' }), 'utf8')

    expect(readPreviewUblockSettings(settingsPath)).toEqual({ enabled: false })
  })

  it('round-trips a valid disabled setting and creates its parent directory', () => {
    const settingsPath = tempSettingsPath()

    writePreviewUblockSettings(settingsPath, { enabled: false })

    expect(readPreviewUblockSettings(settingsPath)).toEqual({ enabled: false })
    expect(JSON.parse(fs.readFileSync(settingsPath, 'utf8'))).toEqual({ enabled: false })
  })
})
