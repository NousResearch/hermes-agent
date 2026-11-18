import { mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, describe, expect, it } from 'vitest'
import { parse } from 'yaml'

import { readDesktopLaunchConfig } from '../electron/renderer-heap-flags.ts'

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
telegram:
  enabled: true
  require_mention: true
gateway:
  platforms:
    slack:
      enabled: true
  photon:
    enabled: true
  streaming:
    enabled: true
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
    expect(config.telegram).toEqual({ enabled: false, require_mention: true })
    expect(config.gateway.photon.enabled).toBe(false)
    expect(config.gateway.streaming.enabled).toBe(true)
    expect(config.model.provider).toBe('openai-codex')
  })

  it('preserves model and auth scalar values for YAML 1.1 and 1.2 consumers', () => {
    const untouched =
      '# model and auth settings\nmodel:\n  default: "no"\n  provider: y\n' +
      'auth:\n  mode: "off"\n  FLAG: "on"\n  mask: 0755\n  channel: 1234567890123456789\n'

    const seeded = withPlatformsDisabled(untouched, ['telegram'])
    for (const version of ['1.1', '1.2']) {
      const options = { version, intAsBigInt: true }
      expect(parse(seeded, options)).toMatchObject(parse(untouched, options))
    }
  })

  it('pins aliased platform blocks without changing their shared defaults', () => {
    const config = parse(
      withPlatformsDisabled(
        'defaults: &bot {enabled: true, token: "on", label: n}\n' +
          'telegram: *bot\ngateway:\n  discord: *bot\nplatforms:\n  photon: *bot\n',
        ['telegram', 'discord', 'photon']
      )
    )

    expect(config.defaults).toEqual({ enabled: true, token: 'on', label: 'n' })
    for (const block of [config.telegram, config.gateway.discord, config.platforms.photon]) {
      expect(block).toEqual({ enabled: false, token: 'on', label: 'n' })
    }
  })

  it('pins shorthand blocks that inherit enabled through a YAML merge', () => {
    const config = parse(
      withPlatformsDisabled(
        'defaults: &bot {enabled: true, token: t}\ntelegram:\n  <<: *bot\ngateway:\n  discord:\n    <<: *bot\n',
        ['telegram', 'discord']
      ),
      { version: '1.1' }
    )

    expect(config.defaults).toEqual({ enabled: true, token: 't' })
    expect(config.telegram).toEqual({ enabled: false, token: 't' })
    expect(config.gateway.discord).toEqual({ enabled: false, token: 't' })
  })

  it.each([
    ['<<: *group', false],
    ['gateway: {<<: *group}', true],
    ['wrapper: &wrapper {gateway: *group}\n<<: *wrapper', true]
  ])('pins a platform block inherited via %s without editing the template', (section, nested) => {
    const config = `group: &group {telegram: {enabled: true, require_mention: true, label: "no"}}\n${section}\n`
    const result = parse(withPlatformsDisabled(config, ['telegram']), { version: '1.1' })

    expect(result.group.telegram.enabled).toBe(true)
    expect((nested ? result.gateway : result).telegram).toEqual({ enabled: false, require_mention: true, label: 'no' })
  })

  it('keeps a long one-line electron_flags list readable by the Desktop launch reader', () => {
    const flags = [
      '--disable-gpu-compositing',
      '--disable-features=Vulkan',
      '--force-device-scale-factor=1',
      '--ozone-platform=x11'
    ]

    const source = `desktop:\n  electron_flags: [${flags.join(', ')}]\n`

    expect(readDesktopLaunchConfig(withPlatformsDisabled(source, ['telegram'])).electronFlags).toEqual(flags)
  })

  it('seeds the isolated home with bundled plugin platforms pinned off and credentials copied', () => {
    const source = tempDir()
    const target = tempDir()
    const repo = tempDir()

    mkdirSync(join(repo, 'plugins', 'platforms', 'photon'), { recursive: true })
    writeFileSync(join(repo, 'plugins', 'platforms', 'photon', '__init__.py'), '')
    mkdirSync(join(repo, 'plugins', 'platforms', 'not_a_platform'))
    writeFileSync(join(source, 'config.yaml'), 'platforms:\n  photon:\n    enabled: true\n')
    writeFileSync(join(source, '.env'), 'TELEGRAM_BOT_TOKEN=x\n')
    writeFileSync(join(source, 'auth.json'), '{}')

    seedConfigFrom(source, target, repo)

    const platforms = parse(readFileSync(join(target, 'config.yaml'), 'utf8')).platforms

    expect(platforms.photon.enabled).toBe(false)
    expect(platforms.telegram.enabled).toBe(false)
    expect(platforms).not.toHaveProperty('not_a_platform')
    expect(readFileSync(join(target, '.env'), 'utf8')).toBe('TELEGRAM_BOT_TOKEN=x\n')
    expect(readFileSync(join(target, 'auth.json'), 'utf8')).toBe('{}')
  })

  it('pins core adapters with credentials only and no bundled package tree', () => {
    const source = tempDir()
    const target = tempDir()
    writeFileSync(join(source, '.env'), 'TELEGRAM_BOT_TOKEN=x\n')

    seedConfigFrom(source, target, tempDir())

    expect(parse(readFileSync(join(target, 'config.yaml'), 'utf8')).platforms.telegram.enabled).toBe(false)
    expect(readFileSync(join(target, '.env'), 'utf8')).toBe('TELEGRAM_BOT_TOKEN=x\n')
    expect(readdirSync(target).sort()).toEqual(['.env', 'config.yaml'])
  })
})
