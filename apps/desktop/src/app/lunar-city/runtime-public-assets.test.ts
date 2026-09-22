import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, it } from 'vitest'

import { collectRuntimePublicAssets } from '../../../scripts/lunar-city/runtime-public-assets'

it('packages referenced assets and ordinary desktop files without authoring sources or obsolete exports', () => {
  const root = mkdtempSync(join(tmpdir(), 'lunar-runtime-pack-'))

  try {
    const file = (path: string, value: string) => {
      const destination = join(root, path)
      mkdirSync(join(destination, '..'), { recursive: true })
      writeFileSync(destination, value)
    }

    const manifest = {
      models: [{ uri: 'models/current.glb' }],
      textures: [{ uri: 'textures/atlas.png' }],
      navigation: { meshUri: 'models/nav.glb' },
      characterAssets: { sharedResourceStrategy: { textureAtlas: 'textures/atlas.png' } }
    }

    file('icon.svg', 'icon')
    file('lunar-city/v2/world-manifest.v2.json', JSON.stringify(manifest))
    file('lunar-city/v2/models/current.glb', 'current')
    file('lunar-city/v2/models/nav.glb', 'nav')
    file('lunar-city/v2/textures/atlas.png', 'atlas')
    file('lunar-city/v2/models/obsolete.glb', 'old')
    file('lunar-city/source/master.blend', 'authoring')
    const assets = collectRuntimePublicAssets(root)
    expect(assets.has('icon.svg')).toBe(true)

    for (const name of ['models/current.glb', 'models/nav.glb', 'textures/atlas.png']) {
      expect(assets.has(`lunar-city/v2/${name}`)).toBe(true)
    }

    expect(assets.has('lunar-city/v2/models/obsolete.glb')).toBe(false)
    expect(assets.has('lunar-city/source/master.blend')).toBe(false)
    manifest.models[0]!.uri = 'models/../../../source/master.blend'
    file('lunar-city/v2/world-manifest.v2.json', JSON.stringify(manifest))
    expect(() => collectRuntimePublicAssets(root)).toThrow('Invalid runtime asset URI')
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})
