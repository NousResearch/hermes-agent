import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { createDesktopMediaProtocolRuntime, ensureWslWindowsFonts, writeFileAtomic } from './desktop-host-utilities'

test('media registration keeps remote cookie and claim routing on their own paths', async () => {
  let media: any
  const calls: string[] = []

  const runtime = createDesktopMediaProtocolRuntime({
    createMediaProtocolHandler: (options: any) => options,
    protocol: { handle: (scheme: string, handler: any) => { assert.equal(scheme, 'hermes-media'); media = handler } },
    MEDIA_PROTOCOL: 'hermes-media',
    ensureNativeAccessToken: async () => 'token',
    fetchLocalMedia: () => 'local',
    electronNet: { fetch: async () => 'remote' },
    getOauthSessionForUrl: () => ({ fetch: async (_url: string, opts: any) => opts.credentials }),
    resolveReadableFileForIpc: async () => ({ resolvedPath: 'C:/safe.mp4' }),
    backendDialClaims: { run: async (_key: string, fn: () => Promise<any>) => { calls.push('claim');

 return fn() } },
    backendScopeKey: () => 'scope',
    ensureRegistryBackend: async () => { calls.push('registry');

 return 'registry' },
    ensureBackend: async () => { calls.push('primary');

 return 'primary' }
  } as any)

  runtime.registerMediaProtocol()
  assert.equal(await media.fetchRemoteWithCookies('https://remote.example', {}, 'GET'), 'include')
  assert.equal(await media.resolveRemoteConnection({ connectionId: 'host-a', profile: 'work' }), 'registry')
  assert.deepEqual(calls, ['claim', 'registry'])
})

test('WSL font wiring is idempotent and only touches an available Windows Fonts mount', () => {
  const writes: string[] = []
  const logs: string[] = []
  let configured = ''

  const deps = {
    isWsl: true,
    fs: {
      statSync: (candidate: string) => ({ isDirectory: () => candidate === '/mnt/c/Windows/Fonts' }),
      readFileSync: () => configured,
      mkdirSync: () => {},
      writeFileSync: (_name: string, content: string) => { configured = content; writes.push(content) }
    },
    path,
    app: { getPath: () => '/home/test' },
    spawn: () => ({ on: () => {}, unref: () => {} }),
    rememberLog: (line: string) => { logs.push(line) }
  }

  ensureWslWindowsFonts(deps as any)
  ensureWslWindowsFonts(deps as any)
  assert.equal(writes.length, 1)
  assert.match(writes[0], /\/mnt\/c\/Windows\/Fonts/)
  assert.equal(logs.length, 1)
})

test('atomic write leaves a complete renamed file', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-e57-'))
  const target = path.join(dir, 'config.json')

  try {
    writeFileAtomic(target, '{"ok":true}', 'utf8')
    assert.equal(fs.readFileSync(target, 'utf8'), '{"ok":true}')
    assert.equal(fs.existsSync(target + '.tmp'), false)
  } finally {
    if (fs.existsSync(target)) {
      fs.unlinkSync(target)
    }

    fs.rmdirSync(dir)
  }
})
