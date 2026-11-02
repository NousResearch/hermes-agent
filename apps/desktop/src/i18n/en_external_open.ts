import type { ExternalOpenFailedCopy } from './types_external_open'

export const enExternalOpenFailed: ExternalOpenFailedCopy = {
  title: 'Couldn’t open this link',
  message: 'No browser is registered to open this address. Copy the link and open it manually.',
  copyUrl: 'Copy link',
  close: 'Close',
  missing: {
    title: 'File not found',
    message: 'This file does not exist — it may have been deleted or moved, or it lives on another machine.'
  }
}
