import type { IpcMain } from 'electron'

interface ClipboardItemLike {
  types: string[]
  getType(type: string): Promise<unknown>
}

interface ClipboardLike {
  read(): Promise<ClipboardItemLike[]>
}

interface NativeImageLike {
  isEmpty(): boolean
  toPNG(): Buffer
}

interface NativeImageFactoryLike {
  createFromBuffer(buffer: Buffer): NativeImageLike
}

interface ClipboardImageIpcOptions {
  ipcMain: Pick<IpcMain, 'handle'>
  clipboard: ClipboardLike
  nativeImage: NativeImageFactoryLike
  isWsl: boolean
  readWslWindowsClipboardImage: () => Buffer | null
  writeComposerImage: (buffer: Buffer, ext?: string) => Promise<string>
}

async function readClipboardImage({
  clipboard,
  nativeImage,
  isWsl,
  readWslWindowsClipboardImage,
  writeComposerImage
}: Omit<ClipboardImageIpcOptions, 'ipcMain'>): Promise<string> {
  const items = await clipboard.read()

  for (const item of items) {
    const imageType = item.types.find(type => type.startsWith('image/'))

    if (!imageType) {
      continue
    }

    try {
      // Electron's typings include a bookmark union for the special bookmark
      // MIME type; this branch only asks for an image MIME type, which is Blob.
      const imageBlob = (await item.getType(imageType)) as Blob
      const image = nativeImage.createFromBuffer(Buffer.from(await imageBlob.arrayBuffer()))

      if (!image.isEmpty()) {
        return writeComposerImage(image.toPNG(), '.png')
      }
    } catch {
      // Keep checking other clipboard entries; one unsupported image format
      // should not prevent a later PNG entry from being attached.
    }
  }

  // WSL2/WSLg doesn't bridge clipboard images from the Windows host to the
  // Linux clipboard Electron reads, so pull them from the host as a fallback.
  if (isWsl) {
    const png = readWslWindowsClipboardImage()

    if (png) {
      return writeComposerImage(png, '.png')
    }
  }

  return ''
}

function registerClipboardImageIpc(options: ClipboardImageIpcOptions): void {
  const { ipcMain, ...readerOptions } = options
  ipcMain.handle('hermes:saveClipboardImage', () => readClipboardImage(readerOptions))
}

export { readClipboardImage, registerClipboardImageIpc }
