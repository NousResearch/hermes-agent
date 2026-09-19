import { createHash } from 'node:crypto'
import { mkdir, mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises'
import { createServer } from 'node:http'
import { once } from 'node:events'
import os from 'node:os'
import path from 'node:path'

import { afterEach, expect, test } from 'vitest'

import { downloadPinnedArtifact } from './artifact'
import { verifyPreparedChannelInstaller } from './channel-windows-host'
import type { ChannelTarget } from './channel'
import type { NativeCommandResult } from './channel-native'

const directories: string[] = []
afterEach(async (): Promise<void> => {
  for (const directory of directories.splice(0)) {
    await rm(directory, { recursive: true, force: true })
  }
})

test('ordinary Windows preparation owns temporary bytes while a resumable download retains its durable file', async (): Promise<void> => {
  const directory: string = await mkdtemp(path.join(os.tmpdir(), 'hermes-artifact-lifetime-'))
  directories.push(directory)
  const bytes: Buffer = Buffer.from('digest-bound download')
  const server = createServer((_request, response): void => { response.end(bytes) })
  server.listen(0, '127.0.0.1')
  await once(server, 'listening')
  const address = server.address()
  if (!address || !(address instanceof Object)) { throw new Error('Expected TCP server') }
  const url: string = `http://127.0.0.1:${address.port}/stable.msixbundle`
  const identity = { token: 'a'.repeat(16), displayName: 'Preview', appId: 'chat.nous.preview', appNamePascal: 'Preview',
    artifactNamePascal: 'Preview', cliName: 'preview', windowsExecutableName: 'preview', msixAppIdWithOrg: 'NousResearch.Preview' }
  const target: ChannelTarget = {
    channel: { schema: 1, state: 'active', name: 'preview', repository: 'NousResearch/hermes-agent', policy: 'preview',
      revision: 1, nextSequence: 2, head: null, identity },
    manifest: { schema: 1, packages: [], request: { schema: 1, buildId: 'b'.repeat(32), channel: 'preview', sequence: 1,
      repository: 'NousResearch/hermes-agent', commit: 'c'.repeat(40), sourceVersion: '1.0.0', version: '0.0.1',
      windowsVersion: '0.0.1.0', identity, bundleEnv: {}, publicBase: 'https://example.com' } },
    package: { platform: 'win32', arch: 'x64', variant: 'bundled', version: '0.0.1.0', identity: identity.msixAppIdWithOrg,
      publisher: 'CN=Test', artifact: { key: 'stable.msixbundle', sha256: createHash('sha256').update(bytes).digest('hex'), size: bytes.length },
      feed: { key: 'stable.appinstaller', channel: 'stable' } },
    artifactUrl: url, feedUrl: 'https://example.com/stable.appinstaller', manifestSha256: 'd'.repeat(64)
  }
  try {
    const file: string = path.join(directory, 'stable.appinstaller')
    await writeFile(file, 'native descriptor checked by command')
    const command = async (_script: string, input: string): Promise<NativeCommandResult> => {
      const payload: { artifact: string } = JSON.parse(input)
      expect(await readFile(payload.artifact)).toEqual(bytes)
      return { stdout: '', stderr: '' }
    }
    await verifyPreparedChannelInstaller(file, target, command)
    expect(await readdir(directory)).toEqual(['stable.appinstaller'])
    await expect(verifyPreparedChannelInstaller(file, target, async (): Promise<never> => {
      throw new Error('native signature refused')
    })).rejects.toThrow('signature refused')
    expect(await readdir(directory)).toEqual(['stable.appinstaller'])
    const durable: string = path.join(directory, 'durable-download')
    await mkdir(durable, { mode: 0o700 })
    const artifact = { url, sha256: target.package.artifact.sha256, size: bytes.length, format: 'msixbundle' as const }
    const downloaded: string = await downloadPinnedArtifact(durable, artifact)
    expect(await downloadPinnedArtifact(durable, artifact)).toBe(downloaded)
    expect(await readFile(downloaded)).toEqual(bytes)
  } finally { server.close(); await once(server, 'close') }
})
