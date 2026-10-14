import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

import { test } from 'vitest'

import { readFileDataUrlForIpc, resolveReadableFileForIpc, resolveRequestedPathForIpc } from './hardening'
import { resolveIpcFileReadPath, resolveMediaStreamFile, resolvePreviewTargetPath } from './local-read-path'
import { isWslBridgeActive, setWslBridgeActive } from './wsl-path-bridge'

test.runIf(process.platform === 'win32')(
  'Windows shell attachments resolve and read through the native preview boundaries',
  async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-shell-preview-'))
    const bridgeWasActive = isWslBridgeActive()
    const source = path.join(dir, 'notes # 100% café.txt')
    const content = Buffer.from('LOCX-29: native preview reads the same attachment bytes.\n')

    const shellPath = (file: string, prefix = '') =>
      `/${prefix}${file[0].toLowerCase()}${file.slice(2).replace(/\\/g, '/')}`

    try {
      fs.writeFileSync(source, content)
      fs.writeFileSync(path.join(dir, '.env'), 'test-only')

      const targets = [
        source,
        pathToFileURL(source).href,
        ...['', 'cygdrive/', 'mnt/'].map(prefix => shellPath(source, prefix))
      ]

      // Native Windows backends can also have the WSL bridge enabled. Shell
      // drive paths must resolve before that bridge can reinterpret them as UNC.
      for (const bridgeActive of [false, true]) {
        setWslBridgeActive(bridgeActive)

        for (const target of targets) {
          const previewPath = resolveRequestedPathForIpc(
            resolvePreviewTargetPath(target, value => value),
            { baseDir: dir }
          )

          assert.equal(previewPath, source)
          assert.equal(fs.statSync(previewPath).isFile(), true)

          for (const readPath of [previewPath, resolveIpcFileReadPath(target), resolveMediaStreamFile(target)]) {
            const readable = await resolveReadableFileForIpc(readPath)
            assert.deepEqual(fs.readFileSync(readable.resolvedPath), content)
            assert.equal(
              await readFileDataUrlForIpc(readPath, { mimeType: 'text/plain' }),
              `data:text/plain;base64,${content.toString('base64')}`
            )
          }
        }
      }

      for (const prefix of ['', 'cygdrive/', 'mnt/']) {
        await assert.rejects(
          resolveReadableFileForIpc(resolveIpcFileReadPath(shellPath(path.join(dir, '.env'), prefix))),
          {
            code: 'sensitive-file'
          }
        )
        await assert.rejects(
          resolveReadableFileForIpc(resolveIpcFileReadPath(shellPath(path.join(dir, 'missing.txt'), prefix))),
          {
            code: 'ENOENT'
          }
        )
        await assert.rejects(
          resolveReadableFileForIpc(resolveIpcFileReadPath(shellPath(source, prefix)), { maxBytes: 1 }),
          { code: 'EFBIG' }
        )
      }

      assert.throws(() => resolveRequestedPathForIpc(resolveIpcFileReadPath('/c/Windows/GlobalRoot/Device/test')), {
        code: 'device-path'
      })
      assert.equal(resolveRequestedPathForIpc('\\\\server\\share\\notes.txt'), '\\\\server\\share\\notes.txt')
    } finally {
      setWslBridgeActive(bridgeWasActive)
      fs.rmSync(dir, { force: true, recursive: true })
    }
  }
)

test.runIf(process.platform !== 'win32')('POSIX shell-looking paths retain their native absolute meaning', () => {
  for (const target of ['/c/Users/notes.txt', '/cygdrive/c/notes.txt', '/mnt/c/notes.txt']) {
    for (const prepared of [
      resolvePreviewTargetPath(target, value => value),
      resolveIpcFileReadPath(target),
      resolveMediaStreamFile(target)
    ]) {
      assert.equal(resolveRequestedPathForIpc(prepared), target)
    }
  }
})
