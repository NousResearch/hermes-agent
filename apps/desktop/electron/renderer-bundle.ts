/**
 * Renderer bundle generation check.
 *
 * `index.html` and the hashed chunks under `dist/assets/` are ONE generation:
 * every `lazy()` route resolves to a filename baked into that generation's
 * module graph. A self-update that replaces the package while its files are
 * locked (antivirus, a still-running instance, an interrupted Windows replace)
 * can leave the two copies electron-builder ships — inside `app.asar` and,
 * because `asarUnpack` lists `dist/**`, beside it in `app.asar.unpacked` —
 * from DIFFERENT generations. The window then loads an `index.html` whose
 * chunks are gone and dies on the first lazy import:
 *
 *   Failed to fetch dynamically imported module:
 *   …/app.asar/dist/assets/shiki-block-COiz1pEN.js
 *
 * The app looks permanently broken (every relaunch reloads the same torn copy),
 * yet the OTHER copy is usually intact. This makes that checkable, so the
 * loader can prefer a complete generation and only report a repair when both
 * are torn.
 *
 * Pure + injectable so it is testable without booting Electron. `fs` here is
 * Electron's asar-aware fs: paths inside `app.asar` read like real files.
 */

import fs from 'node:fs'
import path from 'node:path'

// The modules the browser fetches before any app code runs: Vite emits them as
// `<script type="module" src>` plus `<link rel="modulepreload" href>`.
const TAG_WITH_URL = /<(?:script|link)\b[^>]*\b(?:src|href)=["']([^"']+)["'][^>]*>/gi
const MODULE_TAG = /\btype=["']module["']|\brel=["']modulepreload["']/i

export function parseModuleAssetRefs(html: string): string[] {
  const refs: string[] = []

  for (const [tag, href] of String(html ?? '').matchAll(TAG_WITH_URL)) {
    // Absolute/CDN URLs aren't part of this bundle's generation.
    if (MODULE_TAG.test(tag) && !/^[a-z]+:|^\/\//i.test(href)) {
      refs.push(href.replace(/^\.\//, '').split(/[?#]/)[0])
    }
  }

  return refs
}

export interface RendererBundleDeps {
  readFileSync?: (file: string, encoding: 'utf8') => string
  existsSync?: (file: string) => boolean
}

/**
 * The module files `indexPath` declares but that do not exist beside it.
 *
 * Empty ⇒ a complete generation (or an index naming nothing checkable — the
 * caller's own existence gate owns unreadable/missing files). Non-empty ⇒ torn:
 * loading it produces the "Failed to fetch dynamically imported module" crash.
 */
export function missingRendererAssets(indexPath: string, deps: RendererBundleDeps = {}): string[] {
  const { readFileSync = fs.readFileSync, existsSync = fs.existsSync } = deps
  const dir = path.dirname(indexPath)

  let html: string

  try {
    html = readFileSync(indexPath, 'utf8')
  } catch {
    return []
  }

  return parseModuleAssetRefs(html).filter(ref => !existsSync(path.join(dir, ref)))
}
