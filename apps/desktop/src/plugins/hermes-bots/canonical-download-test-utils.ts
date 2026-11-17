import { expect, vi } from 'vitest'

/** Observe a user-facing browser download, not a private cache-writing IPC. */
export function observeDownloads() {
  const blobs = new Map<string, Blob>()
  const downloads: Array<{ blob: Blob; name: string; connected: boolean }> = []

  const create = vi.fn((blob: Blob) => {
    const url = `blob:http://localhost/file-${blobs.size}`
    blobs.set(url, blob)

    return url
  })

  const revoke = vi.fn()
  vi.stubGlobal('URL', class extends URL {
    static createObjectURL = create
    static revokeObjectURL = revoke
  })

  const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
    const blob = blobs.get(this.href)

    if (!blob) {throw new Error('Download did not use the provided bytes')}
    downloads.push({ blob, name: this.download, connected: this.isConnected })
  })

  return { downloads, create, revoke, click }
}

export async function expectDownloaded(
  observed: ReturnType<typeof observeDownloads>, bytes: Uint8Array, name: string, mime: string, index = 0
) {
  const entry = observed.downloads[index]
  expect(entry.name).toBe(name)
  expect(entry.connected).toBe(true)
  expect(entry.blob.type).toBe(mime)

  const buffer = await new Promise<ArrayBuffer>((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as ArrayBuffer)
    reader.onerror = () => reject(reader.error)
    reader.readAsArrayBuffer(entry.blob)
  })

  expect(new Uint8Array(buffer)).toEqual(bytes)
}
