import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { zipSync } from 'fflate'
import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  createCompatibilityExtensionModule,
  createCompatibilityWorker,
  createCompatibilityWrapper,
  createPreviewUblockInstaller,
  extractArchive,
  prepareExtension,
  PREVIEW_UBLOCK_ARCHIVE_SHA256,
  PREVIEW_UBLOCK_ARCHIVE_URL,
  PREVIEW_UBLOCK_CACHE_NAME,
  PREVIEW_UBLOCK_COMPATIBILITY_REVISION,
  PREVIEW_UBLOCK_VERSION,
  prunePreviewUblockCache,
  validateExtensionDirectory
} from './preview-ublock-installer'

const temporaryDirectories: string[] = []

function temporaryDirectory(): string {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-preview-ublock-'))
  temporaryDirectories.push(directory)

  return directory
}

function archiveFiles(version = PREVIEW_UBLOCK_VERSION, alteredFormatting = false, popupPath?: string): Uint8Array {
  const indentation = alteredFormatting ? '  ' : '    '
  const background = `${indentation}const untouched = true;\n`

  const manifest = JSON.stringify({
    ...(popupPath === undefined ? {} : { action: { default_popup: popupPath } }),
    background: { service_worker: '/js/background.js', type: 'module' },
    dashboard: '/dashboard.html',
    manifest_version: 3,
    name: 'uBlock Origin Lite',
    version
  })

  return zipSync({
    'LICENSE.txt': new TextEncoder().encode('MPL-2.0'),
    'dashboard.html': new TextEncoder().encode('<!doctype html>'),
    'js/background.js': new TextEncoder().encode(background),
    'js/ext-compat.js': new TextEncoder().encode('export const webext = self.browser || self.chrome;\n'),
    ...(popupPath === 'popup.html' ? { 'popup.html': new TextEncoder().encode('<!doctype html>') } : {}),
    'manifest.json': new TextEncoder().encode(manifest),
    'rulesets/main/easylist.json': new TextEncoder().encode('[]'),
    'rulesets/main/easyprivacy.json': new TextEncoder().encode('[]'),
    'rulesets/main/ublock-filters.json': new TextEncoder().encode('[]')
  })
}

function makeValidCachedInstall(userDataPath: string): string {
  const cache = path.join(userDataPath, PREVIEW_UBLOCK_CACHE_NAME)
  const directory = path.join(cache, 'versions', `${PREVIEW_UBLOCK_VERSION}-${PREVIEW_UBLOCK_COMPATIBILITY_REVISION}`)
  const archive = archiveFiles()
  extractArchive(archive, directory, PREVIEW_UBLOCK_VERSION)
  prepareExtension(directory, PREVIEW_UBLOCK_VERSION)
  fs.writeFileSync(
    path.join(directory, 'installed.json'),
    JSON.stringify({
      archiveSha256: PREVIEW_UBLOCK_ARCHIVE_SHA256,
      archiveUrl: PREVIEW_UBLOCK_ARCHIVE_URL,
      compatibilityRevision: PREVIEW_UBLOCK_COMPATIBILITY_REVISION,
      schemaVersion: 2,
      version: PREVIEW_UBLOCK_VERSION
    })
  )
  fs.writeFileSync(
    path.join(cache, 'active.json'),
    JSON.stringify({ cacheKey: path.basename(directory), schemaVersion: 2 })
  )

  return directory
}

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    fs.rmSync(directory, { force: true, recursive: true })
  }
})

describe('preview uBlock installer', () => {
  it('uses only the pinned official URL and reports monotonic streamed progress', async () => {
    const archive = archiveFiles()
    const progress: Array<{ receivedBytes: number; totalBytes: number | null }> = []

    const request = vi.fn(async (url: string, options: { onProgress?: (value: (typeof progress)[number]) => void }) => {
      expect(url).toBe(PREVIEW_UBLOCK_ARCHIVE_URL)
      options.onProgress?.({ receivedBytes: 0, totalBytes: archive.length })
      options.onProgress?.({ receivedBytes: 12, totalBytes: archive.length })
      progress.push({ receivedBytes: 0, totalBytes: archive.length }, { receivedBytes: 12, totalBytes: archive.length })

      return archive
    })

    const installer = createPreviewUblockInstaller({ request, userDataPath: temporaryDirectory() })

    await expect(installer.resolve('pinned')).rejects.toThrow(/checksum/i)
    expect(request).toHaveBeenCalledOnce()
    expect(progress.map(item => item.receivedBytes)).toEqual([0, 12])
  })

  it('does not use the network for cached startup and validates the active pointer', async () => {
    const userDataPath = temporaryDirectory()
    const expectedPath = makeValidCachedInstall(userDataPath)
    const request = vi.fn()
    const installer = createPreviewUblockInstaller({ request, userDataPath })

    await expect(installer.resolve('cached')).resolves.toEqual({ path: expectedPath, version: PREVIEW_UBLOCK_VERSION })
    expect(request).not.toHaveBeenCalled()
  })

  it('invalidates the old schema-v1 mutable cache and interrupted staging debris', async () => {
    const userDataPath = temporaryDirectory()
    const cache = path.join(userDataPath, PREVIEW_UBLOCK_CACHE_NAME)
    fs.mkdirSync(path.join(cache, 'current'), { recursive: true })
    fs.mkdirSync(path.join(cache, '.staging-interrupted'), { recursive: true })
    fs.writeFileSync(path.join(cache, 'installed.json'), '{}')
    createPreviewUblockInstaller({ request: vi.fn(), userDataPath })

    expect(fs.existsSync(path.join(cache, 'current'))).toBe(false)
    expect(fs.existsSync(path.join(cache, '.staging-interrupted'))).toBe(false)
    expect(fs.existsSync(path.join(cache, 'installed.json'))).toBe(false)
  })

  it('removes a validly named orphan while preserving the validated active version', () => {
    const userDataPath = temporaryDirectory()
    const activePath = makeValidCachedInstall(userDataPath)
    const orphanPath = path.join(
      userDataPath,
      PREVIEW_UBLOCK_CACHE_NAME,
      'versions',
      `${PREVIEW_UBLOCK_VERSION}-${PREVIEW_UBLOCK_COMPATIBILITY_REVISION}-orphan`
    )
    fs.mkdirSync(orphanPath, { recursive: true })

    createPreviewUblockInstaller({ request: vi.fn(), userDataPath })

    expect(fs.existsSync(activePath)).toBe(true)
    expect(fs.existsSync(orphanPath)).toBe(false)
  })

  it.each(['not json', JSON.stringify({ cacheKey: 'missing-active', schemaVersion: 2 })])(
    'preserves valid versions when the active pointer is invalid: %s',
    pointer => {
      const userDataPath = temporaryDirectory()
      const activePath = makeValidCachedInstall(userDataPath)
      const cache = path.join(userDataPath, PREVIEW_UBLOCK_CACHE_NAME)
      const orphanPath = path.join(
        cache,
        'versions',
        `${PREVIEW_UBLOCK_VERSION}-${PREVIEW_UBLOCK_COMPATIBILITY_REVISION}-orphan`
      )
      fs.mkdirSync(orphanPath, { recursive: true })
      fs.writeFileSync(path.join(cache, 'active.json'), pointer)

      createPreviewUblockInstaller({ request: vi.fn(), userDataPath })

      expect(fs.existsSync(activePath)).toBe(true)
      expect(fs.existsSync(orphanPath)).toBe(true)
    }
  )

  it('prunes an old active version only after the new pointer is committed', () => {
    const userDataPath = temporaryDirectory()
    const oldPath = makeValidCachedInstall(userDataPath)
    const cache = path.join(userDataPath, PREVIEW_UBLOCK_CACHE_NAME)
    const newKey = `${PREVIEW_UBLOCK_VERSION}-${PREVIEW_UBLOCK_COMPATIBILITY_REVISION}-new`
    const newPath = path.join(cache, 'versions', newKey)
    fs.cpSync(oldPath, newPath, { recursive: true })
    fs.writeFileSync(path.join(cache, 'active.json'), JSON.stringify({ cacheKey: newKey, schemaVersion: 2 }))

    prunePreviewUblockCache(cache)

    expect(fs.existsSync(newPath)).toBe(true)
    expect(fs.existsSync(oldPath)).toBe(false)
  })

  it('allows a cleanup failure to be retried during the next recovery', () => {
    const userDataPath = temporaryDirectory()
    const oldPath = makeValidCachedInstall(userDataPath)
    const cache = path.join(userDataPath, PREVIEW_UBLOCK_CACHE_NAME)
    const newKey = `${PREVIEW_UBLOCK_VERSION}-${PREVIEW_UBLOCK_COMPATIBILITY_REVISION}-new`
    const newPath = path.join(cache, 'versions', newKey)
    fs.cpSync(oldPath, newPath, { recursive: true })
    fs.writeFileSync(path.join(cache, 'active.json'), JSON.stringify({ cacheKey: newKey, schemaVersion: 2 }))

    const originalRemove = fs.rmSync
    const remove = vi.spyOn(fs, 'rmSync').mockImplementation(((target: fs.PathLike) => {
      if (String(target) === oldPath) {
        throw new Error('temporary cleanup failure')
      }

      return originalRemove(target, { force: true, recursive: true })
    }) as typeof fs.rmSync)

    expect(() => prunePreviewUblockCache(cache)).toThrow('temporary cleanup failure')
    remove.mockRestore()
    expect(fs.existsSync(oldPath)).toBe(true)

    createPreviewUblockInstaller({ request: vi.fn(), userDataPath })

    expect(fs.existsSync(newPath)).toBe(true)
    expect(fs.existsSync(oldPath)).toBe(false)
  })

  it('reports cache recovery failures without throwing from installer construction', async () => {
    const userDataPath = temporaryDirectory()
    const cache = path.join(userDataPath, PREVIEW_UBLOCK_CACHE_NAME)
    fs.writeFileSync(cache, 'not a directory')
    const installer = createPreviewUblockInstaller({ request: vi.fn(), userDataPath })

    await expect(installer.resolve('cached')).rejects.toMatchObject({ code: 'storage' })
  })

  it('rejects a checksum failure without replacing an existing active cache', async () => {
    const userDataPath = temporaryDirectory()
    const activePath = makeValidCachedInstall(userDataPath)
    const cache = path.join(userDataPath, PREVIEW_UBLOCK_CACHE_NAME)
    const request = vi.fn().mockResolvedValue(new Uint8Array([1, 2, 3]))
    const installer = createPreviewUblockInstaller({ request, userDataPath })

    await expect(installer.resolve('pinned')).resolves.toEqual({ path: activePath, version: PREVIEW_UBLOCK_VERSION })
    expect(fs.existsSync(path.join(cache, 'active.json'))).toBe(true)
  })

  it.each(['../escape.txt', '/absolute.txt', 'C:/drive.txt', 'nested\\escape.txt', 'nested/../escape.txt'])(
    'rejects unsafe ZIP path %s before writing files',
    unsafePath => {
      const staging = path.join(temporaryDirectory(), 'staging')
      const archive = zipSync({ [unsafePath]: new TextEncoder().encode('blocked') })
      expect(() => extractArchive(archive, staging, PREVIEW_UBLOCK_VERSION)).toThrow(/unsafe path/i)
      expect(fs.existsSync(staging)).toBe(false)
    }
  )

  it('rewrites only the manifest and is independent of upstream source formatting', () => {
    const root = path.join(temporaryDirectory(), 'extension')
    extractArchive(archiveFiles(PREVIEW_UBLOCK_VERSION, true), root, PREVIEW_UBLOCK_VERSION)
    const originalWorker = fs.readFileSync(path.join(root, 'js/background.js'), 'utf8')
    prepareExtension(root, PREVIEW_UBLOCK_VERSION)
    validateExtensionDirectory(root, PREVIEW_UBLOCK_VERSION)
    expect(fs.readFileSync(path.join(root, 'js/background.js'), 'utf8')).toBe(originalWorker)
    expect(JSON.parse(fs.readFileSync(path.join(root, 'manifest.json'), 'utf8')).background.service_worker).toBe(
      'js/hermes-service-worker.js'
    )
    expect(fs.readFileSync(path.join(root, 'js/ext-compat.js'))).toEqual(
      createCompatibilityExtensionModule(fs.readFileSync(path.join(root, 'js/hermes-ext-compat-source.js')))
    )
  })

  it('returns a validated popup path inside the extension directory', () => {
    const root = path.join(temporaryDirectory(), 'extension')
    extractArchive(archiveFiles(PREVIEW_UBLOCK_VERSION, false, 'popup.html'), root, PREVIEW_UBLOCK_VERSION)

    expect(validateExtensionDirectory(root, PREVIEW_UBLOCK_VERSION, false)).toBe('popup.html')
  })

  it.each(['../popup.html', 'nested\\popup.html', 'nested/../popup.html'])(
    'rejects an unsafe declared popup path %s',
    popupPath => {
      const root = path.join(temporaryDirectory(), 'extension')
      expect(() =>
        extractArchive(archiveFiles(PREVIEW_UBLOCK_VERSION, false, popupPath), root, PREVIEW_UBLOCK_VERSION)
      ).toThrow(/unsafe path/i)
    }
  )

  it('rejects a declared popup that is missing or not a regular file', () => {
    const missingRoot = path.join(temporaryDirectory(), 'missing-popup')
    expect(() =>
      extractArchive(archiveFiles(PREVIEW_UBLOCK_VERSION, false, 'missing.html'), missingRoot, PREVIEW_UBLOCK_VERSION)
    ).toThrow(/required extension file is missing.*missing\.html/i)

    const directoryRoot = path.join(temporaryDirectory(), 'directory-popup')
    const directoryArchive = zipSync({
      ...Object.fromEntries(
        Object.entries({
          'LICENSE.txt': '<!doctype html>',
          'dashboard.html': '<!doctype html>',
          'js/background.js': 'const untouched = true;\n',
          'js/ext-compat.js': 'export const webext = self.browser || self.chrome;\n',
          'manifest.json': JSON.stringify({
            action: { default_popup: 'popup' },
            background: { service_worker: 'js/background.js', type: 'module' },
            manifest_version: 3,
            name: 'uBlock Origin Lite',
            version: PREVIEW_UBLOCK_VERSION
          }),
          'rulesets/main/easylist.json': '[]',
          'rulesets/main/easyprivacy.json': '[]',
          'rulesets/main/ublock-filters.json': '[]'
        }).map(([name, value]) => [name, new TextEncoder().encode(value)])
      ),
      'popup/': new Uint8Array()
    })
    expect(() => extractArchive(directoryArchive, directoryRoot, PREVIEW_UBLOCK_VERSION)).toThrow(
      /required extension file is not regular.*popup/i
    )
  })

  it('rejects a declared popup symlink without affecting releases that omit one', () => {
    const root = path.join(temporaryDirectory(), 'symlink-popup')
    extractArchive(archiveFiles(PREVIEW_UBLOCK_VERSION, false, 'popup.html'), root, PREVIEW_UBLOCK_VERSION)
    fs.unlinkSync(path.join(root, 'popup.html'))
    fs.symlinkSync('dashboard.html', path.join(root, 'popup.html'))

    expect(() => validateExtensionDirectory(root, PREVIEW_UBLOCK_VERSION, false)).toThrow(/symlink|regular/i)

    const noPopupRoot = path.join(temporaryDirectory(), 'no-popup')
    extractArchive(archiveFiles(), noPopupRoot, PREVIEW_UBLOCK_VERSION)
    expect(validateExtensionDirectory(noPopupRoot, PREVIEW_UBLOCK_VERSION, false)).toBeNull()
  })

  it('builds a static module worker from an exact shim-plus-source composition', () => {
    const source = new TextEncoder().encode('export const original = true;\n')
    const worker = createCompatibilityWorker(source).toString('utf8')

    expect(worker).toBe(createCompatibilityWrapper() + Buffer.from(source).toString('utf8'))
    expect(worker).not.toMatch(/import\(/)
    expect(worker).not.toMatch(/await\s+import/)
  })

  it('provides strict extension-page permission fallback behavior', async () => {
    const previousChrome = (globalThis as typeof globalThis & { chrome?: unknown }).chrome
    const chrome = { commands: {}, permissions: {} as Record<string, unknown> }
    ;(globalThis as typeof globalThis & { chrome?: unknown }).chrome = chrome

    try {
      const source = createCompatibilityExtensionModule('export const fixture = true;\n').toString('utf8')
      await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`)

      await expect(
        (chrome.permissions.request as (details: unknown) => Promise<boolean>)({ origins: ['<all_urls>'] })
      ).resolves.toBe(true)
      await expect(
        (chrome.permissions.request as (details: unknown) => Promise<boolean>)({
          origins: ['*://*.example.com/*']
        })
      ).resolves.toBe(true)
      await expect(
        (chrome.permissions.request as (details: unknown) => Promise<boolean>)({
          origins: ['<all_urls>'],
          permissions: ['tabs']
        })
      ).rejects.toThrow(/unsupported permission/i)
    } finally {
      ;(globalThis as typeof globalThis & { chrome?: unknown }).chrome = previousChrome
    }
  })
})
