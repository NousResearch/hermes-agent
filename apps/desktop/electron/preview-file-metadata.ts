import fs from 'node:fs'

import { resolveReadableFileForIpc } from './hardening'

export function looksBinary(buffer: Buffer): boolean {
  if (!buffer.length) {
    return false
  }

  let suspicious = 0

  for (const byte of buffer) {
    if (byte === 0) {
      return true
    }

    if (byte < 32 && byte !== 9 && byte !== 10 && byte !== 13) {
      suspicious += 1
    }
  }

  return suspicious / buffer.length > 0.12
}

interface PreviewMetadataOptions {
  metadataOnly?: boolean
  maxPreviewBytes: number
}

export async function previewFileMetadata(filePath: string, mimeType: string, options: PreviewMetadataOptions) {
  const { resolvedPath, stat } = await resolveReadableFileForIpc(filePath, { purpose: 'Preview target' })
  let binary = false

  // Mounted transcript cards need type/existence only. Reading even 4 KiB can
  // hydrate a cloud placeholder; explicit previews must also keep I/O off main.
  if (!options.metadataOnly && !mimeType.startsWith('image/')) {
    try {
      const handle = await fs.promises.open(resolvedPath, 'r')

      try {
        const sample = Buffer.alloc(Math.min(stat.size, 4096))
        const { bytesRead } = await handle.read(sample, 0, sample.length, 0)
        binary = looksBinary(sample.subarray(0, bytesRead))
      } finally {
        await handle.close()
      }
    } catch {
      // Metadata is best-effort; the read handlers surface hard errors later.
    }
  }

  return { resolvedPath, binary, byteSize: stat.size, large: stat.size > options.maxPreviewBytes }
}
