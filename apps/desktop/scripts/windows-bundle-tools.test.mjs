import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterEach, expect, test, vi } from 'vitest'
import { ensureWindowsBundleTools } from './windows-bundle-tools.mjs'
import { batchSignAppTree } from './batch-sign-binaries.mjs'

const directories = []
afterEach(() => {
  for (const directory of directories.splice(0)) fs.rmSync(directory, { recursive: true, force: true })
})

// The adapter models the builder's installer, not an existing cache. The
// native smoke runs its real downloader and all three executable tools.
test('both bundle modes provision pinned tools rather than search a prefilled cache', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'bundle-tools-'))
  directories.push(root)
  const calls = []
  const materialize = (name, files) => {
    const directory = path.join(root, name)
    for (const file of files) {
      const destination = path.join(directory, file)
      fs.mkdirSync(path.dirname(destination), { recursive: true })
      fs.writeFileSync(destination, 'test installer output')
    }
    return directory
  }
  const builder = {
    WIN_CODESIGN_LATEST: 'owned-by-builder',
    getWindowsKitsBundle: async ({ winCodeSign, resourcesDir }) => {
      calls.push(['kit', winCodeSign, resourcesDir])
      return { kit: materialize('kit/x64', ['makeappx.exe', 'signtool.exe']) }
    },
    getAtsBundleDir: async version => {
      calls.push(['ats', version])
      return materialize('ats', ['x64/Azure.CodeSigning.Dlib.dll'])
    },
    getDotnetRuntimeDir: async version => {
      calls.push(['dotnet', version])
      return materialize('dotnet', ['dotnet.exe'])
    },
  }
  const load = async () => builder
  const resourcesDir = path.join(root, 'resources')
  const store = await ensureWindowsBundleTools({ load, config: {}, resourcesDir })
  expect(calls).toEqual([['kit', undefined, resourcesDir]])
  expect(store.dlib).toBeNull()
  expect(store.dotnetRoot).toBeNull()
  expect(fs.existsSync(store.makeappx)).toBe(true)
  const signed = await ensureWindowsBundleTools({ load, config: {}, resourcesDir, signing: true })
  expect(calls.slice(1)).toEqual([
    ['kit', undefined, resourcesDir], ['ats', builder.WIN_CODESIGN_LATEST], ['dotnet', builder.WIN_CODESIGN_LATEST],
  ])
  expect(path.basename(signed.dlib)).toBe('Azure.CodeSigning.Dlib.dll')
  expect(fs.existsSync(path.join(signed.dotnetRoot, 'dotnet.exe'))).toBe(true)
  const failed = vi.fn().mockRejectedValue(new Error('pinned download failed'))
  await expect(ensureWindowsBundleTools({ load: async () => ({ ...builder, getWindowsKitsBundle: failed }), config: {}, resourcesDir })).rejects.toThrow('pinned download failed')
})

test.runIf(process.platform === 'win32')('a cold payload signer provisions executable SDK tools and reuses them warm', { timeout: 240_000 }, async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'bundle-tools-native-'))
  directories.push(root)
  const previous = process.env.ELECTRON_BUILDER_CACHE
  process.env.ELECTRON_BUILDER_CACHE = path.join(root, 'empty-cache')
  try {
    const app = path.join(root, 'app')
    fs.mkdirSync(app)
    const binary = path.join(app, 'payload.exe')
    fs.copyFileSync(process.execPath, binary)
    const signingBoundary = new Error('signing boundary reached')
    let requested
    // Stop before Azure. Downloads and SDK execution below are real.
    await expect(batchSignAppTree(app, path.join(app, 'Hermes.exe'), {
      env: {
        ELECTRON_BUILDER_CACHE: process.env.ELECTRON_BUILDER_CACHE,
        LOCALAPPDATA: root, USERPROFILE: root, TEMP: root,
        AZURE_SIGN_ENDPOINT: 'https://test.invalid',
        AZURE_SIGN_ACCOUNT: 'account', AZURE_SIGN_PROFILE: 'profile',
      },
      cache: null,
      exec: async (signtool, args, options) => {
        expect(args[0]).toBe('sign')
        expect(args).toContain(binary)
        requested = {
          signtool, dlib: args[args.indexOf('/dlib') + 1],
          dotnetRoot: options.env.DOTNET_ROOT,
        }
        throw signingBoundary
      },
    })).rejects.toBe(signingBoundary)
    const tools = await ensureWindowsBundleTools({ signing: true })
    expect(requested).toEqual({ signtool: tools.signtool, dlib: tools.dlib, dotnetRoot: tools.dotnetRoot })
    for (const file of [tools.makeappx, tools.signtool, tools.dlib, tools.dotnetRoot]) {
      expect(path.relative(root, file).startsWith('..')).toBe(false)
    }
    // Verify an actual SDK binary, not a dummy file or a help-only exit.
    const verified = execFileSync(tools.signtool, ['verify', '/pa', tools.makeappx], { encoding: 'utf8', timeout: 60_000 })
    expect(verified).toContain('Successfully verified')
    const runtime = execFileSync(path.join(tools.dotnetRoot, 'dotnet.exe'), ['--list-runtimes'], {
      env: { ...process.env, DOTNET_ROOT: tools.dotnetRoot }, encoding: 'utf8', timeout: 60_000,
    })
    expect(runtime).toContain('Microsoft.NETCore.App')
    const warm = await ensureWindowsBundleTools({ signing: true })
    expect(warm).toEqual(tools)
  } finally {
    if (previous === undefined) delete process.env.ELECTRON_BUILDER_CACHE
    else process.env.ELECTRON_BUILDER_CACHE = previous
  }
})
