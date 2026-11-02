import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { expect, it, vi } from 'vitest'

import {
  PACKAGED_MAIN_MODULE,
  READY_TOKENS,
  assertBackendReadyArtifactSourceAcceptsBothTokens,
  assertPackagedBackendReadyArtifact,
  extractPackagedMainSource,
  resolvePackagedAsarPath
} from './backend-ready-artifact.mjs'

const CURRENT_SOURCE = 'const re = /HERMES_(?:BACKEND|DASHBOARD)_READY[^\\n]*port=(\\d+)/m\n'
// The pre-#55923 matcher: only the legacy token. A packaged bundle carrying
// this kills a current backend after the port-announcement timeout (#60772).
const STALE_SOURCE = 'const re = /HERMES_DASHBOARD_READY port=(\\d+)/m\n'

async function packedAppRoot() {
  const root = await mkdtemp(path.join(os.tmpdir(), 'hermes-ready-artifact-'))
  const resources = path.join(root, 'resources')
  await mkdir(resources, { recursive: true })
  return { root, resources }
}

async function writeAsar(resources, source) {
  // app.asar is an archive we cannot write by hand; the guard reads the
  // .unpacked mirror first, and electron-builder always lays the bundled
  // main out there when asarUnpack applies — and the test controls which
  // path the guard takes, so exercising the unpacked mirror is the real
  // flow for the asarUnpack-eligible bundle.
  const unpacked = path.join(resources, 'app.asar.unpacked', 'dist')
  await mkdir(unpacked, { recursive: true })
  await writeFile(path.join(unpacked, 'electron-main.mjs'), source)
  await writeFile(path.join(resources, 'app.asar'), 'stub archive')
  return path.join(resources, 'app.asar')
}

it('resolves the asar path per platform, including the branded macOS bundle', () => {
  expect(resolvePackagedAsarPath({ appOutDir: '/out/win', electronPlatformName: 'win32' }))
    .toBe(path.join('/out/win', 'resources', 'app.asar'))
  expect(resolvePackagedAsarPath({
    appOutDir: '/out/mac',
    electronPlatformName: 'darwin',
    packager: { appInfo: { productFilename: 'Hermes Preview' } }
  })).toBe(path.join('/out/mac', 'Hermes Preview.app', 'Contents', 'Resources', 'app.asar'))
  expect(() => resolvePackagedAsarPath({ electronPlatformName: 'linux' }))
    .toThrow('missing appOutDir')
})

it('accepts a packaged bundle whose matcher handles both tokens', async () => {
  const { resources } = await packedAppRoot()
  try {
    const asar = await writeAsar(resources, CURRENT_SOURCE)
    const result = assertPackagedBackendReadyArtifact(asar)
    expect(result.module).toBe(PACKAGED_MAIN_MODULE)
    expect(result.tokens).toEqual(READY_TOKENS)
  } finally {
    await rm(resources, { recursive: true, force: true })
  }
})

it('rejects the stale dashboard-only matcher — the #60772 artifact skew', async () => {
  const { resources } = await packedAppRoot()
  try {
    const asar = await writeAsar(resources, STALE_SOURCE)
    expect(() => assertPackagedBackendReadyArtifact(asar)).toThrow(
      'does not contain a packaged readiness matcher accepting HERMES_BACKEND_READY and HERMES_DASHBOARD_READY'
    )
  } finally {
    await rm(resources, { recursive: true, force: true })
  }
})

it('fails the pack when the packaged app.asar is missing', async () => {
  const { resources } = await packedAppRoot()
  try {
    expect(() => assertPackagedBackendReadyArtifact(path.join(resources, 'app.asar')))
      .toThrow('Missing packaged app.asar')
  } finally {
    await rm(resources, { recursive: true, force: true })
  }
})

it('extracts through @electron/asar when no unpacked mirror exists', async () => {
  const { resources } = await packedAppRoot()
  try {
    const asar = path.join(resources, 'app.asar')
    await writeFile(asar, 'stub archive')
    const extracted = vi.fn(() => Buffer.from(CURRENT_SOURCE, 'utf8'))
    const source = extractPackagedMainSource(asar, { asarModule: { extractFile: extracted } })
    expect(source).toBe(CURRENT_SOURCE)
    expect(extracted).toHaveBeenCalledWith(asar, PACKAGED_MAIN_MODULE)
    expect(() => assertPackagedBackendReadyArtifact(asar, { asarModule: { extractFile: () => Buffer.from(STALE_SOURCE) } }))
      .toThrow('readiness matcher')
  } finally {
    await rm(resources, { recursive: true, force: true })
  }
})

it('the CURRENT source-tree parser itself passes the guard', async () => {
  // The shipped parser (electron/backend-ready.ts) must satisfy the very
  // property the packaged bundle is asserted to keep.
  const { readFile } = await import('node:fs/promises')
  const backendReady = await readFile(
    path.resolve(import.meta.dirname, '..', 'electron', 'backend-ready.ts'), 'utf8')
  expect(() => assertBackendReadyArtifactSourceAcceptsBothTokens(backendReady, 'backend-ready.ts'))
    .not.toThrow()
})
