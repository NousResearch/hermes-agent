import { atom } from 'nanostores'

import { persistStringRecord, storedStringRecord } from '@/lib/storage'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'

const STORAGE_KEY = 'hermes.desktop.chat-max-width.v1'

export const CHAT_MAX_WIDTHS = ['off', '44rem', '56rem', '72rem'] as const
export type ChatMaxWidth = (typeof CHAT_MAX_WIDTHS)[number]

const normalizeChatMaxWidth = (value: unknown): ChatMaxWidth => CHAT_MAX_WIDTHS.find(width => width === value) ?? 'off'

const profileWidth = (profile: string): ChatMaxWidth => normalizeChatMaxWidth(storedStringRecord(STORAGE_KEY)[profile])

const activeProfile = () => normalizeProfileKey($activeGatewayProfile.get())

export const $chatMaxWidth = atom<ChatMaxWidth>(profileWidth(activeProfile()))

function applyChatMaxWidth(width: ChatMaxWidth): void {
  if (typeof document !== 'undefined') {
    document.documentElement.style.setProperty('--composer-width', width === 'off' ? '100%' : width)
  }
}

export function setChatMaxWidth(width: ChatMaxWidth): void {
  const profile = activeProfile()
  const next = normalizeChatMaxWidth(width)

  persistStringRecord(STORAGE_KEY, { ...storedStringRecord(STORAGE_KEY), [profile]: next })
  $chatMaxWidth.set(next)
  applyChatMaxWidth(next)
}

$activeGatewayProfile.subscribe(profile => {
  const width = profileWidth(normalizeProfileKey(profile))
  $chatMaxWidth.set(width)
  applyChatMaxWidth(width)
})
