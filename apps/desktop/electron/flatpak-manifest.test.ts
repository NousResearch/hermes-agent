import { readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, test } from 'vitest'

const desktopRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

const manifest = readFileSync(
  path.join(desktopRoot, 'flatpak', 'com.nousresearch.Hermes.yml'),
  'utf8',
)

const finishArgs = manifest
  .slice(manifest.indexOf('finish-args:'), manifest.indexOf('modules:'))
  .split('\n')
  .map(line => line.trim())
  .filter(line => line.startsWith('- '))
  .map(line => line.slice(2))

describe('Flatpak sandbox permissions', () => {
  test('keeps the Electron defaults without broad host filesystem access', () => {
    expect(finishArgs).toEqual(
      expect.arrayContaining([
        '--share=ipc',
        '--share=network',
        '--socket=wayland',
        '--socket=fallback-x11',
        '--socket=pulseaudio',
        '--device=dri',
        '--filesystem=home',
        '--talk-name=org.freedesktop.Notifications',
      ]),
    )
    expect(finishArgs).not.toContain('--filesystem=host:ro')
    expect(finishArgs).not.toContain('--filesystem=host-os:ro')
    expect(finishArgs).not.toContain('--filesystem=/run/media:ro')
  })
})
