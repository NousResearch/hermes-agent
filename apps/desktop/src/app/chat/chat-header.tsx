import { useStore } from '@nanostores/react'

import {
  $workspaceOwnerLabels,
  workspaceOwnerTitle,
  workspaceSessionRenameable
} from '@/components/pane-shell/workspace-scope'
import { TitleMenuTrigger } from '@/components/ui/title-menu-trigger'
import { NEW_SESSION_TITLE, sessionTitle } from '@/lib/chat-runtime'
import { cn } from '@/lib/utils'
import { $interfaceMode } from '@/store/interface-mode'
import { $pinnedSessionIds } from '@/store/layout'
import { $profiles, normalizeProfileKey } from '@/store/profile'
import { $sessions, sessionMatchesStoredId, sessionPinId } from '@/store/session'
import { $botChatScopes } from '@/store/session-states'
import { isAuxiliaryWindow } from '@/store/windows'
import type { SessionInfo } from '@/types/hermes'

import { titlebarHeaderBaseClass, titlebarHeaderShadowClass, titlebarHeaderTitleClass } from '../shell/titlebar'

import { useComposerScope } from './composer/scope'
import { ProfileTag } from './profile-tag'
import { SessionActionsMenu } from './sidebar/session-actions-menu'

interface ChatHeaderProps {
  activeSessionId: null | string
  isRoutedSessionView: boolean
  onDeleteSelectedSession: () => void
  onToggleSelectedPin: () => void
  selectedSessionId: null | string
}

/** The loaded row for this surface's chat. Stored ids are only unique per
 *  owner, so the surface's own profile and connection pick the row. */
function useHeaderSession(sessionId: null | string): null | SessionInfo {
  const sessions = useStore($sessions)
  const scope = useComposerScope()

  if (!sessionId) {
    return null
  }

  return (
    sessions.find(
      session =>
        sessionMatchesStoredId(session, sessionId) &&
        (!scope.profile || normalizeProfileKey(session.profile) === normalizeProfileKey(scope.profile)) &&
        (!scope.connectionId || (session.connection_id?.trim() || 'local') === scope.connectionId)
    ) ?? null
  )
}

// Pins live on the durable lineage-root id, but selectedSessionId is the live
// (tip) id — resolve through the loaded row so the menu reflects the pin
// state after auto-compression rotates the id.
function isHeaderSessionPinned(
  pinned: readonly string[],
  session: null | SessionInfo,
  selectedSessionId: null | string
) {
  if (session) {
    return pinned.includes(sessionPinId(session))
  }

  return selectedSessionId ? pinned.includes(selectedSessionId) : false
}

export function ChatHeader({
  activeSessionId,
  isRoutedSessionView,
  onDeleteSelectedSession,
  onToggleSelectedPin,
  selectedSessionId
}: ChatHeaderProps) {
  const simple = useStore($interfaceMode) === 'simple'
  const pinnedSessionIds = useStore($pinnedSessionIds)
  const profiles = useStore($profiles)
  const botScopes = useStore($botChatScopes)
  useStore($workspaceOwnerLabels)
  const botScope = selectedSessionId ? botScopes[selectedSessionId] : undefined

  const activeStoredSession = useHeaderSession(selectedSessionId || activeSessionId)
  const storedTitle = activeStoredSession ? sessionTitle(activeStoredSession) : NEW_SESSION_TITLE

  const title = botScope ? workspaceOwnerTitle(botScope.workspaceTabTitle || storedTitle, botScope) : storedTitle

  // Which agent/persona owns this chat — glanceable in the header once a
  // second profile exists, so the open session's ownership is never ambiguous
  // (#66003). Single-profile users see the unchanged header.
  const showProfileTag = profiles.length > 1 && Boolean(activeStoredSession)

  const selectedIsPinned = isHeaderSessionPinned(pinnedSessionIds, activeStoredSession, selectedSessionId)

  // Secondary windows (new-session scratch, subagent watch, cmd-click pop-out)
  // are compact side panels — they drop the session-actions header + border
  // entirely. A brand-new draft has nothing to pin/delete/rename either. Simple
  // has no header: its tab carries the title and the tab menu the actions.
  if (simple || isAuxiliaryWindow() || (!selectedSessionId && !activeSessionId && !isRoutedSessionView)) {
    return null
  }

  return (
    <header className={cn(titlebarHeaderBaseClass, isRoutedSessionView && titlebarHeaderShadowClass)}>
      <div
        className={cn(titlebarHeaderTitleClass, showProfileTag && 'flex items-center')}
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
          renameable={workspaceSessionRenameable(botScope)}
          sessionId={selectedSessionId || activeSessionId || ''}
          sideOffset={8}
          title={title}
        >
          <TitleMenuTrigger>{title}</TitleMenuTrigger>
        </SessionActionsMenu>
      </div>
    </header>
  )
}
