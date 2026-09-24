import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { DEFAULT_SKIN_COLORS, readConfiguredSkinName, readLocalSkin, readSkinFile } from './local-skin'

describe('readConfiguredSkinName', () => {
  it('reads display.skin among other keys, including PyYAML indentless lists', () => {
    const config = [
      'model:',
      '  default: some/model',
      'toolsets:',
      '- web',
      '- terminal',
      'display:',
      '  pet:',
      '    enabled: true',
      '  skin: "neon"  # picked by /skin',
      '  compact: false',
      'memory:',
      '  enabled: true'
    ].join('\n')

    expect(readConfiguredSkinName(config)).toBe('neon')
  })

  it('defaults to `default` when display.skin is unset or blank, like init_skin_from_config', () => {
    expect(readConfiguredSkinName('model:\n  default: x\n')).toBe('default')
    expect(readConfiguredSkinName('display:\n  compact: true\n')).toBe('default')
    expect(readConfiguredSkinName("display:\n  skin: ''\n")).toBe('default')
  })

  it('handles plain, single-quoted and CRLF values', () => {
    expect(readConfiguredSkinName('display:\r\n  skin: mono\r\n')).toBe('mono')
    expect(readConfiguredSkinName("display:\n    skin: 'forest'\n")).toBe('forest')
  })

  it('refuses a value it cannot read faithfully', () => {
    expect(readConfiguredSkinName('display:\n  skin: "a\\u0041"\n')).toBeNull()
  })
})

describe('readSkinFile', () => {
  it('merges the skin colors over the default skin, as _build_skin_config', () => {
    const skin = readSkinFile(
      ['name: neon', 'description: test', 'colors:', '  background: "#171720"', "  ui_accent: '#ff2d95'"].join('\n')
    )

    expect(skin).toEqual({
      name: 'neon',
      colors: { ...DEFAULT_SKIN_COLORS, background: '#171720', ui_accent: '#ff2d95' }
    })
  })

  it('uses the name inside the file, not the file name', () => {
    expect(readSkinFile('name: other\ncolors:\n  ui_text: "#ffffff"\n')?.name).toBe('other')
  })

  it('treats an unquoted hex as the YAML comment it is, and null as unset', () => {
    const colors = readSkinFile('name: x\ncolors:\n  ui_accent: #ff0000\n  ui_error: null\n')?.colors

    expect(colors).not.toHaveProperty('ui_error')
    expect(colors).not.toHaveProperty('ui_accent')
  })

  it('rejects files load_skin would not accept, and shapes it cannot read', () => {
    expect(readSkinFile('colors:\n  ui_text: "#ffffff"\n')).toBeNull() // no name
    expect(readSkinFile('name: x\ncolors: {ui_text: "#fff"}\n')).toBeNull() // flow mapping
    expect(readSkinFile('name: x\ncolors:\n    ui_text: "#fff"\n  ui_accent: "#000"\n')).toBeNull() // bad indent
  })
})

describe('readLocalSkin', () => {
  let home: string

  beforeEach(() => {
    home = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-local-skin-'))
  })

  afterEach(() => {
    fs.rmSync(home, { recursive: true, force: true })
  })

  const write = (rel: string, text: string) => {
    fs.mkdirSync(path.dirname(path.join(home, rel)), { recursive: true })
    fs.writeFileSync(path.join(home, rel), text)
  }

  it('resolves the configured user skin from the home skins dir', () => {
    write('config.yaml', 'display:\n  skin: neon\n')
    write('skins/neon.yaml', 'name: neon\ncolors:\n  background: "#171720"\n')

    expect(readLocalSkin(home, null)?.colors?.background).toBe('#171720')
  })

  it('reads a named profile from profiles/<name>/', () => {
    write('config.yaml', 'display:\n  skin: neon\n')
    write('profiles/work/config.yaml', 'display:\n  skin: forest\n')
    write('profiles/work/skins/forest.yaml', 'name: forest\ncolors:\n  background: "#001100"\n')

    expect(readLocalSkin(home, 'work')?.name).toBe('forest')
    expect(readLocalSkin(home, 'default')?.name).toBe('neon')
  })

  it('returns just the name when there is no user file (built-in skins)', () => {
    write('config.yaml', 'display:\n  skin: mono\n')

    expect(readLocalSkin(home, null)).toEqual({ name: 'mono' })
  })

  it('returns null for the default skin, a missing config, or a path-like name', () => {
    expect(readLocalSkin(home, null)).toBeNull()
    write('config.yaml', 'display:\n  skin: default\n')
    expect(readLocalSkin(home, null)).toBeNull()
    write('config.yaml', 'display:\n  skin: ../evil\n')
    expect(readLocalSkin(home, null)).toBeNull()
  })
})

it('DEFAULT_SKIN_COLORS matches the Python default skin', () => {
  const source = fs.readFileSync(
    path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../hermes_cli/skin_engine.py'),
    'utf8'
  )

  const block = /"default": \{[\s\S]*?"colors": \{([\s\S]*?)\}/.exec(source)?.[1] ?? ''
  const python = Object.fromEntries([...block.matchAll(/"(\w+)": "([^"]+)"/g)].map(([, k, v]) => [k, v]))

  expect(Object.keys(python).length).toBeGreaterThan(10)
  expect(DEFAULT_SKIN_COLORS).toEqual(python)
})
