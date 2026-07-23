import type { ContextMenuParams, MenuItemConstructorOptions } from 'electron'

interface ImageContextMenuActions {
  copyImageAt: (x: number, y: number) => void
  openExternalUrl: (url: string) => void
  reportError: (message: string) => void
  saveImageFromUrl: (url: string) => Promise<unknown>
  writeText: (text: string) => void
}

function imageContextMenuItems(
  params: Pick<ContextMenuParams, 'hasImageContents' | 'mediaType' | 'srcURL' | 'x' | 'y'>,
  actions: ImageContextMenuActions
): MenuItemConstructorOptions[] {
  if (params.mediaType !== 'image' || !params.hasImageContents) {
    return []
  }

  const copyImage: MenuItemConstructorOptions = {
    label: 'Copy Image',
    click: () => actions.copyImageAt(params.x, params.y)
  }

  if (!params.srcURL) {
    return [copyImage]
  }

  return [
    {
      label: 'Open Image',
      click: () => {
        if (!params.srcURL.startsWith('data:')) {
          actions.openExternalUrl(params.srcURL)
        }
      },
      enabled: !params.srcURL.startsWith('data:')
    },
    copyImage,
    {
      label: 'Copy Image Address',
      click: () => actions.writeText(params.srcURL)
    },
    {
      label: 'Save Image As...',
      click: () => {
        void actions
          .saveImageFromUrl(params.srcURL)
          .catch(error => actions.reportError(`Save image failed: ${error.message}`))
      }
    }
  ]
}

export { imageContextMenuItems }
export type { ImageContextMenuActions }
