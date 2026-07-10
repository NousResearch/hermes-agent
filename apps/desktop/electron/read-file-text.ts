export interface FileReadHandle {
  read: (buffer: Buffer, offset: number, length: number, position: number) => Promise<{ bytesRead: number }>
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
