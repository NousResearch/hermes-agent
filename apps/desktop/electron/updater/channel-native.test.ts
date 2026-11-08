import { createHash } from 'node:crypto'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

import { expect, test } from 'vitest'

import { verifyChannelDownload } from './channel-native'

test('download verification binds native bytes to the manifest size and digest before signature preparation', async (): Promise<void> => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'channel-artifact-'))
  const file = path.join(directory, 'update.zip')

  try {
    const content = Buffer.from('real local archive fixture bytes')
    await writeFile(file, content)

    const artifact = {
      key: 'releases/fixture.zip',
      size: content.length,
      sha256: createHash('sha256').update(content).digest('hex')
    }

    await verifyChannelDownload([file], artifact)
    await writeFile(file, Buffer.alloc(content.length))
    await expect(verifyChannelDownload([file], artifact)).rejects.toThrow('digest')
    await expect(verifyChannelDownload([], artifact)).rejects.toThrow('one')
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
