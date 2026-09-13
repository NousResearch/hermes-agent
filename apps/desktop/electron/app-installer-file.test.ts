import * as fs from 'node:fs/promises'
import * as http from 'node:http'
import * as os from 'node:os'
import * as path from 'node:path'

import { expect, test } from 'vitest'

import { stageAppInstallerFile } from './app-installer-file'

test('stages an actual HTTP descriptor and leaves a valid file intact on download failure', async () => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'hermes-installer-file-'))
  const descriptor = '<?xml version="1.0"?><AppInstaller Uri="https://updates.example/app.appinstaller" />'

  const server = http.createServer((request, response) => {
    if (request.url === '/missing') {
      response.writeHead(404).end()
    } else if (request.url === '/large') {
      response.end(Buffer.alloc(1024 * 1024 + 1))
    } else {
      response.end(descriptor)
    }
  })

  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address()

  if (!address || typeof address === 'string') { throw new Error('missing server address') }
  const base = `http://127.0.0.1:${address.port}`

  try {
    const target = await stageAppInstallerFile(`${base}/update`, directory)
    expect(target.endsWith('.appinstaller')).toBe(true)
    expect(await fs.readFile(target, 'utf8')).toBe(descriptor)
    await expect(stageAppInstallerFile(`${base}/missing`, directory)).rejects.toThrow('404')
    await expect(stageAppInstallerFile(`${base}/large`, directory)).rejects.toThrow('1 MiB')
    expect(await fs.readFile(target, 'utf8')).toBe(descriptor)
    expect(await fs.readdir(directory)).toEqual(['update.appinstaller'])
  } finally {
    server.closeAllConnections()
    await new Promise<void>(resolve => server.close(() => resolve()))
    await fs.rm(directory, { recursive: true, force: true })
  }
})
