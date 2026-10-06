import { useStore } from '@nanostores/react'

import { TitleMenuTrigger } from '@/components/ui/title-menu-trigger'
import { NEW_SESSION_TITLE, sessionTitle } from '@/lib/chat-runtime'
import { cn } from '@/lib/utils'
import { $pinnedSessionIds } from '@/store/layout'
import { $profiles } from '@/store/profile'
import { $sessions, sessionMatchesStoredId, sessionPinId } from '@/store/session'
import { isAuxiliaryWindow } from '@/store/windows'

import { titlebarHeaderBaseClass, titlebarHeaderShadowClass, titlebarHeaderTitleClass } from '../shell/titlebar'

import { ConversationSearch } from './conversation-search'
import { ProfileTag } from './profile-tag'
import { SessionActionsMenu } from './sidebar/session-actions-menu'

interface ChatHeaderProps {
  activeSessionId: null | string
  isRoutedSessionView: boolean
  onDeleteSelectedSession: () => void
  onToggleSelectedPin: () => void
  selectedSessionId: null | string
}

export function ChatHeader({
  activeSessionId,
  isRoutedSessionView,
  onDeleteSelectedSession,
  onToggleSelectedPin,
  selectedSessionId
}: ChatHeaderProps) {
  const sessions = useStore($sessions)
  const pinnedSessionIds = useStore($pinnedSessionIds)
  const profiles = useStore($profiles)

  const activeStoredSession =
    ((selectedSessionId || activeSessionId) &&
      sessions.find(session => sessionMatchesStoredId(session, selectedSessionId || activeSessionId || ''))) ||
    null

  const title = activeStoredSession ? sessionTitle(activeStoredSession) : NEW_SESSION_TITLE

  // Which agent/persona owns this chat — glanceable in the header once a
  // second profile exists, so the open session's ownership is never ambiguous
  // (#66003). Single-profile users see the unchanged header.
  const showProfileTag = profiles.length > 1 && Boolean(activeStoredSession)

  // Pins live on the durable lineage-root id, but selectedSessionId is the live
  // (tip) id — resolve through the loaded row so the menu reflects the pin
  // state after auto-compression rotates the id.
  const selectedIsPinned = activeStoredSession
    ? pinnedSessionIds.includes(sessionPinId(activeStoredSession))
    : selectedSessionId
      ? pinnedSessionIds.includes(selectedSessionId)
      : false

  // Secondary windows (new-session scratch, subagent watch, cmd-click pop-out)
  // are compact side panels — they drop the session-actions header + border
  // entirely. A brand-new draft has nothing to pin/delete/rename either.
  if (isAuxiliaryWindow() || (!selectedSessionId && !activeSessionId && !isRoutedSessionView)) {
    return null
  }

  return (
    <header className={cn(titlebarHeaderBaseClass, isRoutedSessionView && titlebarHeaderShadowClass)}>
      <div
        className={cn(titlebarHeaderTitleClass, 'flex items-center')}
        style={{
          maxWidth:
            'calc(100vw - var(--titlebar-content-inset,0px) - var(--titlebar-tools-right) - var(--titlebar-tools-width) - 1.5rem)'
        }}
      >
        {showProfileTag && <ProfileTag className="pointer-events-auto mr-1.5" profile={activeStoredSession?.profile} />}
        <SessionActionsMenu
          align="start"
          onDelete={selectedSessionId ? onDeleteSelectedSession : undefined}
          onPin={selectedSessionId ? onToggleSelectedPin : undefined}
          pinned={selectedIsPinned}
          profile={activeStoredSession?.profile}
          sessionId={selectedSessionId || activeSessionId || ''}
          sideOffset={8}
          title={title}
        >
          <TitleMenuTrigger>{title}</TitleMenuTrigger>
        </SessionActionsMenu>
        <ConversationSearch />
      </div>
    </header>
  )
}
