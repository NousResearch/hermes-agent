export interface FileReadHandle {
  read: (buffer: Buffer, offset: number, length: number, position: number) => Promise<{ bytesRead: number }>
}

export interface FileReadSnapshot {
  ctimeMs: number
  isFile: () => boolean
  mtimeMs: number
  size: number
}

export interface StableFileReadHandle extends FileReadHandle {
  stat: () => Promise<FileReadSnapshot>
}

export interface CompleteTextFileRead {
  buffer: Buffer
  byteSize: number
}

export async function readTextFileBytes(
  handle: FileReadHandle,
  fileSize: number,
  previewMaxBytes: number,
  complete: boolean
): Promise<Buffer> {
  const requestedBytes = complete ? fileSize : Math.min(fileSize, previewMaxBytes)
  const buffer = Buffer.alloc(requestedBytes)
  let bytesRead = 0

  while (bytesRead < requestedBytes) {
    const result = await handle.read(buffer, bytesRead, requestedBytes - bytesRead, bytesRead)

    if (!result.bytesRead) {
      break
    }

    bytesRead += result.bytesRead
  }

  return buffer.subarray(0, bytesRead)
}

export async function readCompleteTextFileBytes(
  handle: StableFileReadHandle,
  sourceMaxBytes: number
): Promise<CompleteTextFileRead> {
  const before = await handle.stat()

  if (!before.isFile()) {
    throw new Error('Only regular files can be read')
  }

  if (!Number.isSafeInteger(before.size) || before.size < 0 || before.size > sourceMaxBytes) {
    throw new Error('File too large')
  }

  const buffer = await readTextFileBytes(handle, before.size, before.size, true)
  const after = await handle.stat()

  const changed =
    buffer.length !== before.size ||
    after.size !== before.size ||
    after.mtimeMs !== before.mtimeMs ||
    after.ctimeMs !== before.ctimeMs

  if (changed) {
    throw new Error('File changed while reading')
  }

  return { buffer, byteSize: before.size }
}
