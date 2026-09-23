import type { WorldManifestV2 } from './model'

/** Must never be converted into an optional rendering/navigation fallback. */
export class RuntimeAssetIntegrityError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'RuntimeAssetIntegrityError'
  }
}

export interface RuntimeAssetFetchOptions {
  fetch?: typeof globalThis.fetch
  signal?: AbortSignal
}

type IntegrityManifest = Pick<WorldManifestV2, 'models' | 'reviewLeaderAssets' | 'reviewWorkerAssets'> & { navigation?: WorldManifestV2['navigation'] }

/** Receipt paths outside this manifest cannot add or replace trusted asset digests. */
export function buildRuntimeAssetDigests(
  manifest: IntegrityManifest,
  manifestUrl: string,
  reviewSources?: unknown
): ReadonlyMap<string, string> {
  const digests = new Map<string, string>()

  const add = (uri: string, sha256: unknown): void => {
    if (typeof sha256 !== 'string' || !/^[a-f\d]{64}$/iu.test(sha256)) {
      throw new RuntimeAssetIntegrityError(`Lunar City asset has no valid SHA-256: ${uri}`)
    }

    const url = new URL(uri, manifestUrl).href
    const normalized = sha256.toLowerCase()

    if (digests.has(url) && digests.get(url) !== normalized) {
      throw new RuntimeAssetIntegrityError(`Lunar City asset has conflicting digests: ${uri}`)
    }

    digests.set(url, normalized)
  }

  for (const model of manifest.models) {
    add(model.uri, model.statistics.sha256)
  }

  if (manifest.navigation?.sha256) {add(manifest.navigation.meshUri, manifest.navigation.sha256)}
  const reviews = [...(manifest.reviewLeaderAssets ?? []), ...(manifest.reviewWorkerAssets ?? [])]

  if (reviews.length === 0) {
    return digests
  }

  if (!Array.isArray(reviewSources)) {
    throw new RuntimeAssetIntegrityError('Lunar City review asset hash receipts are required')
  }

  const allowed = new Set(reviews.map(asset => new URL(asset.uri, manifestUrl).href))

  for (const row of reviewSources) {
    if (!row || typeof row !== 'object' || typeof row.uri !== 'string') {
      continue
    }

    const url = new URL(row.uri, manifestUrl).href

    if (allowed.has(url)) {
      add(row.uri, row.sha256)
    }
  }

  for (const url of allowed) {
    if (!digests.has(url)) {
      throw new RuntimeAssetIntegrityError(`Lunar City review asset hash receipt is missing: ${url}`)
    }
  }

  return digests
}

/**
 * Import the exact verified bytes, never re-fetch the original URL after validation.
 * Current packs are self-contained GLBs. The explicit extension selects Babylon's
 * GLB plugin for blob URLs; the URL lives until import (including textures) completes.
 * Assets without a declared hash retain their existing loader contract.
 */
export async function importRuntimeAsset<T>(
  url: string,
  digests: ReadonlyMap<string, string>,
  importAsset: (url: string, pluginExtension?: '.glb') => Promise<T>,
  options: RuntimeAssetFetchOptions = {}
): Promise<T> {
  const expected = digests.get(url)

  if (!expected) {
    return importAsset(url)
  }

  options.signal?.throwIfAborted()
  const response = await (options.fetch ?? globalThis.fetch)(url, { signal: options.signal })

  if (!response.ok) {
    throw new RuntimeAssetIntegrityError(`Lunar City asset fetch failed (${response.status}): ${url}`)
  }

  const bytes = await response.arrayBuffer()
  options.signal?.throwIfAborted()
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes)
  const actual = Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, '0')).join('')

  if (actual !== expected) {
    throw new RuntimeAssetIntegrityError(`Lunar City asset digest mismatch: ${url}`)
  }

  options.signal?.throwIfAborted()
  const objectUrl = URL.createObjectURL(new Blob([bytes], { type: 'model/gltf-binary' }))

  try {
    return await importAsset(objectUrl, '.glb')
  } finally {
    URL.revokeObjectURL(objectUrl)
  }
}
