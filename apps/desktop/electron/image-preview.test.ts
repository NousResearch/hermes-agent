import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

import { test, vi } from 'vitest'

import { readFileDataUrlForIpc } from './hardening'
import { imagePreviewDataUrl } from './image-preview'

// A generated 64x32 solid-colour HEIC; no user photo or metadata.
const heic = Buffer.from(
  'AAAAHGZ0eXBoZWljAAAAAG1pZjFoZWljbWlhZgAAAX1tZXRhAAAAAAAAACFoZGxyAAAAAAAAAABwaWN0AAAAAAAAAAAAAAAAAAAAACJpbG9jAAAAAERAAAEAAQAAAAABoQABAAAAAAAAAB0AAAAjaWluZgAAAAAAAQAAABVpbmZlAgAAAAABAABodmMxAAAAAA5waXRtAAAAAAABAAAA/WlwcnAAAADdaXBjbwAAAHZodmNDAQNwAAAAAAAAAAAAHvAA/P34+AAADwNgAAEAGEABDAH//wNwAAADAJAAAAMAAAMAHroCQGEAAQAqQgEBA3AAAAMAkAAAAwAAAwAeoCCBBZbq5Ka5uAhoMCAAAAMDIAAAAwAhYgABAAZEAcFzwIkAAAATY29scm5jbHgAAQANAAaAAAAAFGlzcGUAAAAAAAAAQAAAAEAAAAAoY2xhcAAAAEAAAAABAAAAIAAAAAEAAAAAAAAAAv///+AAAAACAAAAEHBpeGkAAAAAAwgICAAAABhpcG1hAAAAAAAAAAEAAQWBAgMFhAAAACVtZGF0AAAAGSgBrxOA9SeBP//g/Uf/w754fY6zfnny1PA=',
  'base64'
)

test.skipIf(process.platform !== 'darwin')(
  'preview converts HEIC bytes despite .png extension, but upload preserves originals',
  async () => {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'hermes-preview-test-'))

    try {
      const file = path.join(dir, 'photo.png')
      await fs.writeFile(file, heic)
      const options = { mimeType: 'image/png', maxBytes: 1024 * 1024, imagePreview: true }
      const preview = await readFileDataUrlForIpc(file, options)
      assert.ok(preview.startsWith('data:image/jpeg;base64,'))
      assert.equal(Buffer.from(preview.split(',')[1], 'base64').subarray(0, 3).toString('hex'), 'ffd8ff')
      assert.deepEqual(await fs.readFile(file), heic)
      const upload = await readFileDataUrlForIpc(file, { ...options, imagePreview: false })
      assert.deepEqual(Buffer.from(upload.split(',')[1], 'base64'), heic)
      await assert.rejects(readFileDataUrlForIpc(file, { ...options, maxBytes: 1 }), { code: 'EFBIG' })
      const secret = path.join(dir, '.env')
      await fs.writeFile(secret, heic)
      const link = path.join(dir, 'secret.png')
      await fs.symlink(secret, link)
      await assert.rejects(readFileDataUrlForIpc(link, options), { code: 'sensitive-file' })
    } finally {
      await fs.rm(dir, { recursive: true, force: true })
    }
  }
)


test('preview preserves non-HEIC bytes and cleans temporary files on decoder failure', async () => {
  const ordinary = Buffer.from('not an HEIC image')
  assert.equal(await imagePreviewDataUrl(ordinary, 'image/png'), `data:image/png;base64,${ordinary.toString('base64')}`)

  if (process.platform !== 'darwin') {
    assert.equal(await imagePreviewDataUrl(heic, 'image/heic'), `data:image/heic;base64,${heic.toString('base64')}`)

    return
  }

  const mkdtemp = vi.spyOn(fs, 'mkdtemp')

  try {
    // Valid ftyp, but missing the image payload: exercise real sips failure.
    await assert.rejects(imagePreviewDataUrl(heic.subarray(0, 28), 'image/heic'))
    assert.equal(mkdtemp.mock.results.length, 1)
    const dir = await mkdtemp.mock.results[0].value
    await assert.rejects(fs.stat(dir), { code: 'ENOENT' })
  } finally {
    mkdtemp.mockRestore()
  }
})
