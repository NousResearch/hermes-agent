import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  AppleMailDropCapabilityRegistry,
  appleMailOsaScriptArgs,
  cleanupAllAppleMailExports,
  decodeAppleMailMessageUri,
  exportSelectedAppleMailMessage,
  isTrustedDesktopRendererUrl,
  removeAppleMailExport
} from './apple-mail-drop'

const tempDirs: string[] = []

afterEach(async () => {
  vi.restoreAllMocks()
  await Promise.all(tempDirs.splice(0).map(dir => fs.promises.rm(dir, { force: true, recursive: true })))
})

describe('decodeAppleMailMessageUri', () => {
  it('decodes the percent-encoded Message-ID emitted by Apple Mail', () => {
    expect(decodeAppleMailMessageUri('message:%3Cexample-id%40example.com%3E')).toBe('example-id@example.com')
  })

  it('accepts a valid Message-ID whose local part starts with a dash', () => {
    expect(decodeAppleMailMessageUri('message:%3C-e%40example.com%3E')).toBe('-e@example.com')
  })

  it('normalizes the message:// URI form emitted by some Apple Mail versions', () => {
    expect(decodeAppleMailMessageUri('message://%3Cexample-id%40example.com%3E')).toBe('example-id@example.com')
  })

  it('rejects non-Mail URLs and malformed Message-IDs', () => {
    expect(decodeAppleMailMessageUri('https://example.com/mail')).toBeNull()
    expect(decodeAppleMailMessageUri('message:%E0%A4%A')).toBeNull()
    expect(decodeAppleMailMessageUri('message:%3Cnot-an-id%3E')).toBeNull()
  })
})

describe('isTrustedDesktopRendererUrl', () => {
  const packaged = 'file:///Applications/Hermes.app/Contents/Resources/app.asar/dist/index.html'

  it('accepts only the configured development origin', () => {
    expect(isTrustedDesktopRendererUrl('http://127.0.0.1:5174/#/', 'http://127.0.0.1:5174', packaged)).toBe(true)
    expect(isTrustedDesktopRendererUrl('http://localhost:5174/#/', 'http://127.0.0.1:5174', packaged)).toBe(false)
  })

  it('requires the exact packaged file host and pathname', () => {
    expect(isTrustedDesktopRendererUrl(`${packaged}#/chat`, undefined, packaged)).toBe(true)
    expect(
      isTrustedDesktopRendererUrl(
        'file://foreign-host/Applications/Hermes.app/Contents/Resources/app.asar/dist/index.html',
        undefined,
        packaged
      )
    ).toBe(false)
    expect(
      isTrustedDesktopRendererUrl(
        'file:///Applications/Hermes.app/Contents/Resources/app.asar/dist/other.html',
        undefined,
        packaged
      )
    ).toBe(false)
  })
})

describe('appleMailOsaScriptArgs', () => {
  it('places an option terminator before the untrusted Message-ID argument', () => {
    const args = appleMailOsaScriptArgs('-e@example.com')

    expect(args.slice(-2)).toEqual(['--', '-e@example.com'])
  })
})

describe('AppleMailDropCapabilityRegistry', () => {
  it('consumes a matching drop capability exactly once', () => {
    const registry = new AppleMailDropCapabilityRegistry(10_000)
    const uri = 'message:%3Cexample-id%40example.com%3E'

    expect(registry.register(42, uri, 'token-123', 1_000)).toBe(true)
    expect(registry.consume(42, uri, 'token-123', 2_000)).toBe(true)
    expect(registry.consume(42, uri, 'token-123', 2_001)).toBe(false)
  })

  it('matches equivalent message: and message:// spellings by normalized Message-ID', () => {
    const registry = new AppleMailDropCapabilityRegistry(10_000)

    expect(registry.register(42, 'message://%3Cexample-id%40example.com%3E', 'token-123', 1_000)).toBe(true)
    expect(registry.consume(42, 'message:%3Cexample-id%40example.com%3E', 'token-123', 2_000)).toBe(true)
  })

  it('rejects mismatched and expired capabilities', () => {
    const registry = new AppleMailDropCapabilityRegistry(10_000)
    const uri = 'message:%3Cexample-id%40example.com%3E'

    expect(registry.register(42, uri, 'token-123', 1_000)).toBe(true)
    expect(registry.consume(42, 'message:%3Cother%40example.com%3E', 'token-123', 2_000)).toBe(false)
    expect(registry.register(42, uri, 'token-expired', 1_000)).toBe(true)
    expect(registry.consume(42, uri, 'token-expired', 11_001)).toBe(false)
  })

  it('sweeps expired unconsumed capabilities when registering a fresh drop', () => {
    const registry = new AppleMailDropCapabilityRegistry(10_000)
    const uri = 'message:%3Cexample-id%40example.com%3E'

    expect(registry.register(42, uri, 'expired-token', 1_000)).toBe(true)
    expect(registry.register(42, uri, 'fresh-token', 12_000)).toBe(true)
    expect(registry.size).toBe(1)
    expect(registry.consume(42, uri, 'expired-token', 12_001)).toBe(false)
    expect(registry.consume(42, uri, 'fresh-token', 12_001)).toBe(true)
  })

  it('rejects registration for malformed message references and tokens', () => {
    const registry = new AppleMailDropCapabilityRegistry(10_000)

    expect(registry.register(42, 'https://example.com', 'token-123', 1_000)).toBe(false)
    expect(registry.register(42, 'message:%3Cexample%40example.com%3E', '', 1_000)).toBe(false)
  })
})

describe('exportSelectedAppleMailMessage', () => {
  it('writes the verified selected message source to a private managed eml file', async () => {
    const userDataDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'hermes-apple-mail-test-'))
    tempDirs.push(userDataDir)
    const source = 'From: sender@example.com\r\nSubject: Test\r\n\r\nBody\r\n'
    const readSelectedSource = vi.fn(async () => source)

    const exportedPath = await exportSelectedAppleMailMessage({
      messageUri: 'message:%3Cexample-id%40example.com%3E',
      readSelectedSource,
      userDataDir
    })

    expect(readSelectedSource).toHaveBeenCalledWith('example-id@example.com')
    expect(path.extname(exportedPath)).toBe('.eml')
    expect(path.dirname(exportedPath)).toBe(path.join(await fs.promises.realpath(userDataDir), 'apple-mail-drops'))
    await expect(fs.promises.readFile(exportedPath, 'utf8')).resolves.toBe(source)
    expect((await fs.promises.stat(exportedPath)).mode & 0o777).toBe(0o600)
  })

  it('uses a collision-resistant filename for repeated exports', async () => {
    const userDataDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'hermes-apple-mail-test-'))
    tempDirs.push(userDataDir)

    const options = {
      messageUri: 'message:%3Cexample-id%40example.com%3E',
      readSelectedSource: async () => 'From: sender@example.com\r\n\r\nBody\r\n',
      userDataDir
    }

    const [first, second] = await Promise.all([
      exportSelectedAppleMailMessage(options),
      exportSelectedAppleMailMessage(options)
    ])

    expect(first).not.toBe(second)
  })

  it('removes the export if closing its file handle fails', async () => {
    const userDataDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'hermes-apple-mail-test-'))
    tempDirs.push(userDataDir)

    const unlink = vi.spyOn(fs.promises, 'unlink').mockResolvedValue()

    const close = vi.fn(async () => {
      throw new Error('close failed')
    })

    vi.spyOn(fs.promises, 'open').mockResolvedValue({ close, writeFile: vi.fn(async () => undefined) } as never)

    await expect(
      exportSelectedAppleMailMessage({
        messageUri: 'message:%3Cexample-id%40example.com%3E',
        readSelectedSource: async () => 'From: sender@example.com\r\n\r\nBody\r\n',
        userDataDir
      })
    ).rejects.toThrow('close failed')
    expect(unlink).toHaveBeenCalledOnce()
  })

  it('rejects an empty message source', async () => {
    const userDataDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'hermes-apple-mail-test-'))
    tempDirs.push(userDataDir)

    await expect(
      exportSelectedAppleMailMessage({
        messageUri: 'message:%3Cexample-id%40example.com%3E',
        readSelectedSource: async () => '',
        userDataDir
      })
    ).rejects.toThrow('empty')
  })
})

describe('Apple Mail export cleanup and path safety', () => {
  it('removes only eml files inside the managed export directory', async () => {
    const userDataDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'hermes-apple-mail-test-'))
    tempDirs.push(userDataDir)
    const managedDir = path.join(userDataDir, 'apple-mail-drops')
    await fs.promises.mkdir(managedDir, { recursive: true })
    const managed = path.join(managedDir, 'message.eml')
    const outside = path.join(userDataDir, 'outside.eml')
    await fs.promises.writeFile(managed, 'mail', { mode: 0o600 })
    await fs.promises.writeFile(outside, 'outside', { mode: 0o600 })

    await expect(removeAppleMailExport(userDataDir, managed)).resolves.toBe(true)
    await expect(fs.promises.stat(managed)).rejects.toMatchObject({ code: 'ENOENT' })
    await expect(removeAppleMailExport(userDataDir, outside)).rejects.toThrow('managed Apple Mail export')
    await expect(fs.promises.readFile(outside, 'utf8')).resolves.toBe('outside')
  })

  it('removes every abandoned export on startup', async () => {
    const userDataDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'hermes-apple-mail-test-'))
    tempDirs.push(userDataDir)
    const managedDir = path.join(userDataDir, 'apple-mail-drops')
    await fs.promises.mkdir(managedDir, { recursive: true })
    const first = path.join(managedDir, 'first.eml')
    const second = path.join(managedDir, 'second.eml')
    await fs.promises.writeFile(first, 'first', { mode: 0o600 })
    await fs.promises.writeFile(second, 'second', { mode: 0o600 })

    await expect(cleanupAllAppleMailExports(userDataDir)).resolves.toBe(2)
    await expect(fs.promises.stat(first)).rejects.toMatchObject({ code: 'ENOENT' })
    await expect(fs.promises.stat(second)).rejects.toMatchObject({ code: 'ENOENT' })
  })

  it('rejects a symlinked managed export directory', async () => {
    const userDataDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'hermes-apple-mail-test-'))
    const outsideDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'hermes-apple-mail-outside-'))
    tempDirs.push(userDataDir, outsideDir)
    await fs.promises.symlink(outsideDir, path.join(userDataDir, 'apple-mail-drops'))

    await expect(
      exportSelectedAppleMailMessage({
        messageUri: 'message:%3Cexample-id%40example.com%3E',
        readSelectedSource: async () => 'From: sender@example.com\r\n\r\nBody\r\n',
        userDataDir
      })
    ).rejects.toThrow('symlink')
    await expect(cleanupAllAppleMailExports(userDataDir)).rejects.toThrow('symlink')
  })

  it('refuses to unlink a symlink masquerading as a managed eml export', async () => {
    const userDataDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'hermes-apple-mail-test-'))
    tempDirs.push(userDataDir)
    const managedDir = path.join(userDataDir, 'apple-mail-drops')
    await fs.promises.mkdir(managedDir, { recursive: true })
    const outside = path.join(userDataDir, 'outside.eml')
    const link = path.join(managedDir, 'link.eml')
    await fs.promises.writeFile(outside, 'outside', { mode: 0o600 })
    await fs.promises.symlink(outside, link)

    await expect(removeAppleMailExport(userDataDir, link)).rejects.toThrow('symlink')
    await expect(fs.promises.readFile(outside, 'utf8')).resolves.toBe('outside')
  })
})
