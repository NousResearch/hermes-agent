/** A Bot Chat may show cached messages before its explicit refresh has hydrated.
 * Generic tile/selection focus must not acknowledge its unread watermark. */
const protectedBotChats = new Set<string>()
let hydratedBotChats = new Set<string>()

export function protectBotChatRead(storedSessionId: null | string | undefined) {
  if (storedSessionId) {
    protectedBotChats.add(storedSessionId)
  }
}

export function isBotChatReadProtected(storedSessionId: null | string | undefined) {
  return Boolean(storedSessionId && protectedBotChats.has(storedSessionId) && !hydratedBotChats.has(storedSessionId))
}

export function afterSuccessfulBotChatRefresh(storedSessionIds: readonly string[], action: () => void) {
  const previous = hydratedBotChats
  hydratedBotChats = new Set([...previous, ...storedSessionIds])

  try {
    action()
  } finally {
    hydratedBotChats = previous
  }
}
