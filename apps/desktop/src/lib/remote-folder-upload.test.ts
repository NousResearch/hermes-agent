import { strFromU8, unzipSync } from 'fflate'
import { describe, expect, it } from 'vitest'

import {
  type RemoteFolderReader,
  zipLocalFolderForRemoteUpload
} from './remote-folder-upload'

function dataUrl(text: string): string {
  return `data:text/plain;base64,${Buffer.from(text, 'utf8').toString('base64')}`
}

// In-memory local disk: dirs map to entries, files to text.
function memoryReader(tree: Record<string, Array<{ name: string; dir?: boolean }>>, files: Record<string, string>) {
  const reader: RemoteFolderReader = {
    readDir: async path => {
      const entries = tree[path]

      if (!entries) {
        throw new Error(`ENOENT ${path}`)
      }

      return {
        entries: entries.map(e => ({
          name: e.name,
          path: `${path}/${e.name}`,
          isDirectory: Boolean(e.dir)
        }))
      }
    },
    readFileDataUrl: async path => (path in files ? dataUrl(files[path]) : null)
  }

  return reader
}

function unzipNames(dataUrlOut: string): Record<string, string> {
  const base64 = dataUrlOut.split(',').pop() || ''
  const bytes = Uint8Array.from(atob(base64), ch => ch.charCodeAt(0))
  const out: Record<string, string> = {}

  for (const [name, content] of Object.entries(unzipSync(bytes))) {
    out[name] = strFromU8(content as Uint8Array)
  }

  return out
}

describe('zipLocalFolderForRemoteUpload', () => {
  it('zips nested files preserving structure and skips VCS/dependency trees', async () => {
    const reader = memoryReader(
      {
        '/docs': [{ name: 'jan.pdf' }, { name: 'feb', dir: true }, { name: '.git', dir: true }],
        '/docs/feb': [{ name: 'scan.txt' }],
        '/docs/.git': [{ name: 'HEAD' }]
      },
      { '/docs/jan.pdf': 'pdf-bytes', '/docs/feb/scan.txt': 'scanned', '/docs/.git/HEAD': 'ref' }
    )

    const zipped = await zipLocalFolderForRemoteUpload('/docs', reader)

    expect(zipped.filename).toBe('docs.zip')
    expect(zipped.fileCount).toBe(2)
    expect(zipped.dataUrl.startsWith('data:application/zip;base64,')).toBe(true)
    expect(unzipNames(zipped.dataUrl)).toEqual({ 'jan.pdf': 'pdf-bytes', 'feb/scan.txt': 'scanned' })
  })

  it('rejects an empty folder instead of uploading a useless archive', async () => {
    const reader = memoryReader({ '/empty': [] }, {})

    await expect(zipLocalFolderForRemoteUpload('/empty', reader)).rejects.toThrow(/no files/)
  })

  it('fails loudly on an unreadable file instead of a silent partial upload', async () => {
    const reader = memoryReader({ '/docs': [{ name: 'gone.txt' }] }, {})

    await expect(zipLocalFolderForRemoteUpload('/docs', reader)).rejects.toThrow(/Could not read gone\.txt/)
  })

  it('enforces the file-count cap', async () => {
    const reader = memoryReader({ '/docs': [{ name: 'a.txt' }, { name: 'b.txt' }] }, {
      '/docs/a.txt': 'a',
      '/docs/b.txt': 'b'
    })

    await expect(zipLocalFolderForRemoteUpload('/docs', reader, { maxFiles: 1 })).rejects.toThrow(/too many files/)
  })

  it('fails on an unreadable folder root', async () => {
    const reader = memoryReader({}, {})

    await expect(zipLocalFolderForRemoteUpload('/missing', reader)).rejects.toThrow(/Could not read folder/)
  })
})
