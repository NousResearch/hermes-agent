export const CHATGPT_SESSION_PARTITION = 'persist:hermes-chatgpt'

interface DownloadEvent {
  preventDefault: () => void
}

interface RestrictedSession {
  on: (event: 'will-download', handler: (event: DownloadEvent) => void) => void
  setPermissionCheckHandler: (handler: (webContents: unknown, permission: string) => boolean) => void
  setPermissionRequestHandler: (
    handler: (webContents: unknown, permission: string, callback: (allowed: boolean) => void, details: unknown) => void
  ) => void
}

/**
 * The ChatGPT cookie jar is persistent but capability-empty. Login cookies stay
 * in this partition; camera, microphone, location, notifications, clipboard and
 * other browser permissions never cross into Hermes, and remote pages cannot
 * download files behind the user's back.
 */
export function configureChatGptSession(chatGptSession: RestrictedSession): void {
  chatGptSession.setPermissionRequestHandler((_webContents, _permission, callback) => callback(false))
  chatGptSession.setPermissionCheckHandler(() => false)
  chatGptSession.on('will-download', event => event.preventDefault())
}