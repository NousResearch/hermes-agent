import type { SessionChannelOrigin } from '@/types/hermes'

export function sessionChannelOriginLabel(origin: null | SessionChannelOrigin | undefined): null | string {
  const displayName = origin?.display_name?.trim()

  if (!displayName) {
    return null
  }

  const topic = origin?.chat_topic?.trim()

  if (topic && topic.toLowerCase() !== displayName.toLowerCase()) {
    return `${displayName} · ${topic}`
  }

  return displayName
}
