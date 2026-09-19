import type { ReactNode } from 'react'

interface DraftWelcomeProfileInput {
  activeProfile: string
  newChatProfile: null | string
  routeProfile: null | string
}

/** Resolve the profile that owns a fresh draft, matching session.create. */
export function resolveDraftWelcomeProfile({
  activeProfile,
  newChatProfile,
  routeProfile
}: DraftWelcomeProfileInput): string {
  return routeProfile || newChatProfile || activeProfile.trim() || 'default'
}

/** The fresh draft's welcome surface, before a session exists. */
export const CHAT_WELCOME_AREA = 'chat.welcome'

/** Props handed to a chat-welcome contribution's `render`. */
export interface ChatWelcomeProps {
  /** The draft's current working directory. */
  cwd: string | null
  /** The Hermes profile that will own the draft. */
  profile: string
}

/** Payload of a `chat.welcome` contribution's `data`. */
export interface ChatWelcomeContribution {
  /** Replaces the core Intro for a fresh draft. */
  render: (props: ChatWelcomeProps) => ReactNode
}
