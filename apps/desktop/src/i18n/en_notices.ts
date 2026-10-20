import { enDesktopCommands } from './en_desktop_commands'
import { enSessionCommands } from './en_session_commands'
import type { Translations } from './types'

// Shell notices (remote-display toast, butterbar, keep-awake offer), spread into en.ts.
export const enNotices = {
  remoteDisplayBanner: {
    message: reason =>
      `Software rendering active — remote display detected (${reason}). GPU acceleration is disabled to prevent flickering.`
  },
  butterbar: {
    goTo: (index, total) => `Show notice ${index} of ${total}`,
    legal: {
      before: 'Use of Hermes Agent is subject to our ',
      terms: 'Terms of Service',
      between: ' and ',
      privacy: 'Privacy Policy',
      after: '.'
    }
  },
  promptNotices: {
    legacySendUnconfirmed:
      'This server could not confirm the earlier send of this message, so it may already have run. Check the conversation before sending it again.',
    mcpReloadDeferred:
      'This chat is busy: the MCP change applies once its turn finishes and MCP reloads again, or when you reopen the chat',
    mcpReloadNotInChat: server => `${server} is not enabled in this chat`,
    mcpReloadUnavailable: servers => `Did not connect: ${servers}`,
    compactedTurnUnavailable:
      "That message was summarized when the chat was compressed, so it can't be restored or edited anymore.",
    staleTurnUnavailable: "That message is no longer in this chat's history.",
    sessionCommands: enSessionCommands,
    desktopCommands: enDesktopCommands
  },
  keepAwakeOffer: {
    title: 'Your computer slept while Hermes was working',
    message:
      'Hermes can keep your computer awake while a task runs. The screen still turns off, and on battery your computer may still sleep.',
    action: 'Keep awake while working'
  }
} satisfies Pick<Translations, 'remoteDisplayBanner' | 'butterbar' | 'promptNotices' | 'keepAwakeOffer'>
