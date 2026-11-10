import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, describe, expect, it } from 'vitest'
import { parse } from 'yaml'

import { seedConfigFrom, withPlatformsDisabled } from './perf/lib/seed-home.mjs'

const dirs = []

const tempDir = () => {
  const dir = mkdtempSync(join(tmpdir(), 'perf-seed-'))
  dirs.push(dir)

  return dir
}

afterEach(() => {
  for (const dir of dirs.splice(0)) {
    rmSync(dir, { force: true, recursive: true })
  }
})

describe('isolated perf home seeding', () => {
  it('pins every known platform off, including ones enabled only by .env or auth.json evidence', () => {
    const config = parse(
      withPlatformsDisabled(
        `
model:
  provider: openai-codex
discord:
  require_mention: true
platforms:
  photon:
    enabled: true
  telegram:
    streaming: true
`,
        ['discord', 'photon', 'telegram', 'slack']
      )
    )

    expect(config.platforms.photon.enabled).toBe(false)
    expect(config.platforms.telegram).toEqual({ enabled: false, streaming: true })
    // Top-level `discord:` and absent `slack` have no `enabled` key, so .env credentials would
    // turn them on; only an explicit platforms.<name>.enabled: false outranks that.
    expect(config.platforms.discord).toEqual({ enabled: false })
    expect(config.platforms.slack).toEqual({ enabled: false })
    expect(config.discord).toEqual({ require_mention: true })
    expect(config.model.provider).toBe('openai-codex')
  })

  it('seeds the isolated home with bundled plugin platforms pinned off and credentials copied', () => {
    const source = tempDir()
    const target = tempDir()
    const repo = tempDir()

    mkdirSync(join(repo, 'plugins', 'platforms', 'photon'), { recursive: true })
    writeFileSync(join(repo, 'plugins', 'platforms', 'photon', '__init__.py'), '')
    writeFileSync(join(source, 'config.yaml'), 'platforms:\n  photon:\n    enabled: true\n')
    writeFileSync(join(source, '.env'), 'TELEGRAM_BOT_TOKEN=x\n')
    writeFileSync(join(source, 'auth.json'), '{}')

    seedConfigFrom(source, target, repo)

    const platforms = parse(readFileSync(join(target, 'config.yaml'), 'utf8')).platforms

    expect(platforms.photon.enabled).toBe(false)
    expect(platforms.telegram.enabled).toBe(false)
    expect(readFileSync(join(target, '.env'), 'utf8')).toBe('TELEGRAM_BOT_TOKEN=x\n')
    expect(readFileSync(join(target, 'auth.json'), 'utf8')).toBe('{}')
  })
})
