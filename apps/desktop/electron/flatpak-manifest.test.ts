import { readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, test } from 'vitest'

const desktopRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

const manifest = readFileSync(
  path.join(desktopRoot, 'flatpak', 'com.nousresearch.Hermes.yml'),
  'utf8',
)

const metainfo = readFileSync(
  path.join(desktopRoot, 'flatpak', 'com.nousresearch.Hermes.metainfo.xml'),
  'utf8',
)

const desktopPackage = JSON.parse(readFileSync(path.join(desktopRoot, 'package.json'), 'utf8'))
const projectToml = readFileSync(path.join(desktopRoot, '..', '..', 'pyproject.toml'), 'utf8')
const projectVersion = projectToml.match(/^version = "([^"]+)"$/m)?.[1]

const finishArgs = manifest
  .slice(manifest.indexOf('finish-args:'), manifest.indexOf('modules:'))
  .split('\n')
  .map(line => line.trim())
  .filter(line => line.startsWith('- '))
  .map(line => line.slice(2))

describe('Flatpak release metadata', () => {
  test('uses the canonical Hermes package version', () => {
    expect(desktopPackage.version).toBe(projectVersion)
    expect(metainfo).toContain(`<release version="${projectVersion}"`)
  })
})

describe('Flatpak sandbox permissions', () => {
  test('keeps the Electron defaults with host filesystem access for git repos', () => {
    expect(finishArgs).toEqual(
      expect.arrayContaining([
        '--share=ipc',
        '--share=network',
        '--socket=wayland',
        '--socket=fallback-x11',
        '--socket=pulseaudio',
        '--socket=session-bus',
        '--device=dri',
        '--filesystem=host',
        '--talk-name=org.freedesktop.Notifications',
        '--talk-name=org.freedesktop.portal.Desktop',
        '--talk-name=org.freedesktop.portal.FileChooser',
      ]),
    )
    expect(finishArgs).not.toContain('--filesystem=home')
    expect(finishArgs).not.toContain('--filesystem=home:ro')
  })
})
