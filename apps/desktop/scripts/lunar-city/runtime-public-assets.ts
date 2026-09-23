import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { join, relative, resolve, sep } from 'node:path'

/** Package the asset graph, keeping authoring files and superseded exports outside the renderer. */
export function collectRuntimePublicAssets(publicDir: string): Map<string, Buffer> {
  const assets = new Map<string, Buffer>()
  const add = (path: string) => assets.set(relative(publicDir, path).split(sep).join('/'), readFileSync(path))

  const walk = (directory: string) => {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const path = join(directory, entry.name)

      if (entry.isDirectory()) {
        walk(path)
      } else if (entry.isFile()) {
        add(path)
      }
    }
  }

  if (!existsSync(publicDir)) {
    return assets
  }

  for (const entry of readdirSync(publicDir, { withFileTypes: true })) {
    if (entry.name === 'lunar-city') {
      continue
    }

    const path = join(publicDir, entry.name)

    if (entry.isDirectory()) {
      walk(path)
    } else if (entry.isFile()) {
      add(path)
    }
  }

  for (const pack of ['v2', 'v2-review']) {
    const root = join(publicDir, 'lunar-city', pack),
      manifestPath = join(root, 'world-manifest.v2.json')

    if (!existsSync(manifestPath)) {
      continue
    }

    const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'))

    const uris = new Set<string>([
      ...manifest.models.map((entry: { uri: string }) => entry.uri),
      ...manifest.textures.map((entry: { uri: string }) => entry.uri),
      manifest.navigation.meshUri,
      manifest.characterAssets.sharedResourceStrategy.textureAtlas,
      ...(manifest.reviewLeaderAssets ?? []).map((entry: { uri: string }) => entry.uri),
      ...(manifest.reviewWorkerAssets ?? []).map((entry: { uri: string }) => entry.uri),
      ...(manifest.externalDecorations ?? []).map((entry: { uri: string }) => entry.uri)
    ])

    add(manifestPath)

    for (const uri of uris) {
      if (typeof uri !== 'string' || !/^(models|textures)\/[a-zA-Z0-9_./-]+\.(glb|png|jpg|jpeg|webp|ktx2)$/.test(uri)) {
        throw new Error(`Invalid runtime asset URI: ${uri}`)
      }

      const path = resolve(root, uri)

      if (!path.startsWith(resolve(root) + sep)) {
        throw new Error(`Asset escapes runtime pack: ${uri}`)
      }

      add(path)
    }

    for (const receipt of ['review-sources.json', 'environment-build.json', 'source-reference.v2.json']) {
      const path = join(root, receipt)

      if (existsSync(path)) {
        add(path)
      }
    }
  }

  return assets
}

export function runtimePublicAssetsPlugin() {
  let publicDir = ''

  return {
    name: 'hermes:lunar-runtime-public-assets',
    apply: 'build' as const,
    configResolved(config: { publicDir: string }) {
      publicDir = config.publicDir
    },
    generateBundle(this: { emitFile(asset: { type: 'asset'; fileName: string; source: Uint8Array }): unknown }) {
      for (const [fileName, source] of collectRuntimePublicAssets(publicDir)) {
        this.emitFile({ type: 'asset', fileName, source })
      }
    }
  }
}
