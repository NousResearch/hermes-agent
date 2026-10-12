import { useStore } from '@nanostores/react'
import { useLocation } from 'react-router'

import { routeSessionId } from '@/app/routes'
import { Codicon } from '@/components/ui/codicon'
import { ProfileGlyph } from '@/components/ui/profile-glyph'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { displayPath, pathLeaf } from '@/lib/display-path'
import { profileColor, resolveProfileColor } from '@/lib/profile-color'
import { useStoreSelector } from '@/lib/use-session-slice'
import {
  $activeGatewayProfile,
  $newChatConnectionId,
  $newChatProfile,
  $newChatRoute,
  $profileColors,
  $profiles,
  normalizeProfileKey,
  profileLabel
} from '@/store/profile'
import { $projectOwnerBySessionId, $projects, $projectTree, projectNameForCwd } from '@/store/projects'
import { $connection, $sessions, sessionMatchesStoredId } from '@/store/session'
import { isSessionOwnerRoute } from '@/store/session-request-router'
import { $botChatScopes, knownOwnerForSession } from '@/store/session-states'

import { useComposerScope } from './composer/scope'
import { useSessionView } from './session-view'
import { ProfileSwitcher } from './sidebar/profile-dropdown-switcher'
import { liveSessionProjectId, NO_PROJECT_ID } from './sidebar/projects/workspace-groups'

interface ChatContextProps {
  cwd?: null | string
  profile?: null | string
  /** The composer's Simple layout is in use (not the HUD or the setup guide). */
  shown: boolean
}

/** Keep conversation context with its surface, including retained tabs. */
export function ChatContext({ cwd, profile, shown }: ChatContextProps) {
  const view = useSessionView()
  const scope = useComposerScope()
  const storedSessionId = useStore(view.$storedId)
  const { pathname } = useLocation()
  // The URL names the primary chat before the store has loaded it; until they
  // agree, the loaded session's owner would describe the wrong conversation.
  const routedSessionId = scope.target === 'main' ? routeSessionId(pathname) : storedSessionId
  const settled = !routedSessionId || routedSessionId === storedSessionId

  return shown && settled ? (
    <SimpleChatContext cwd={cwd || ''} profile={profile || ''} storedSessionId={storedSessionId} />
  ) : null
}

interface SimpleChatContextProps {
  cwd: string
  profile: string
  storedSessionId: null | string
}

/** The owner a draft is about to use: its captured route, else the intended profile. */
function useDraftOwner(storedSessionId: null | string) {
  const profile = useStore($newChatProfile)
  const route = useStore($newChatRoute)
  const connectionId = useStore($newChatConnectionId)

  if (storedSessionId) {
    return { connectionId: undefined, profile: undefined }
  }

  return {
    connectionId: route?.connectionId || connectionId || undefined,
    profile: route?.targetProfile || route?.profile || profile || undefined
  }
}

/** Which profile and connection own this surface's chat. A Bot scope wins, then
 *  the surface's own scope, the stored owner, a draft's intent, and the fallback. */
function useChatOwner(storedSessionId: null | string, fallbackProfile: string) {
  const scope = useComposerScope()
  const connection = useStore($connection)
  const draft = useDraftOwner(storedSessionId)
  const botScope = useStoreSelector($botChatScopes, scopes => (storedSessionId ? scopes[storedSessionId] : undefined))
  const owner = botScope?.ownerRoute || knownOwnerForSession(storedSessionId)
  const route = isSessionOwnerRoute(owner) ? owner : undefined
  const storedProfile = route ? route.targetProfile || route.profile : (owner as string | null | undefined)
  const connectionId = connection?.connectionId || (connection?.mode === 'local' ? 'local' : '')

  const ownerConnectionId =
    botScope?.ownerRoute?.connectionId ||
    scope.connectionId ||
    route?.connectionId ||
    draft.connectionId ||
    connectionId

  return {
    connectionId,
    ownerConnectionId,
    profileKey: normalizeProfileKey(
      botScope?.ownerProfile || scope.profile || storedProfile || draft.profile || fallbackProfile
    ),
    sameConnection: ownerConnectionId === connectionId
  }
}

type ChatOwner = ReturnType<typeof useChatOwner>

/** The project or folder this chat belongs to, from the owner's own records. */
function useChatProject(storedSessionId: null | string, cwd: string, owner: ChatOwner) {
  const { t } = useI18n()
  const activeProfile = useStore($activeGatewayProfile)
  const projectTree = useStore($projectTree)
  const projects = useStore($projects)
  const projectOwners = useStore($projectOwnerBySessionId)

  // Restored tiles can show their history before a runtime publishes its cwd.
  // Match the existing workspace fallback without borrowing another owner’s row.
  const storedSession = useStoreSelector($sessions, sessions =>
    sessions.find(
      session =>
        storedSessionId &&
        sessionMatchesStoredId(session, storedSessionId) &&
        normalizeProfileKey(session.profile) === owner.profileKey &&
        (session.connection_id?.trim() || owner.connectionId) === owner.ownerConnectionId
    )
  )

  const workspaceCwd = cwd || storedSession?.cwd || ''

  // The project tree belongs to the active profile/connection. A background
  // tab on another owner must never borrow its project names for equal paths.
  const ownsProjectCache = owner.sameConnection && owner.profileKey === normalizeProfileKey(activeProfile)

  const storedProjectName = () => {
    const projectId =
      projectOwners.get(storedSession?.id || storedSessionId || '') ??
      (storedSession ? liveSessionProjectId(storedSession, projects, projectOwners) : null)

    return (
      projectId !== NO_PROJECT_ID &&
      (projectTree.find(project => project.id === projectId)?.label ||
        projects.find(project => project.id === projectId)?.name)
    )
  }

  const projectName = ownsProjectCache
    ? storedSessionId
      ? storedProjectName()
      : projectNameForCwd(workspaceCwd)
    : null

  // An owner whose tree is unavailable can still name its working folder,
  // but that folder must not masquerade as confirmed project membership.
  const folderOnly = !ownsProjectCache && Boolean(workspaceCwd)

  return {
    folderOnly,
    label: folderOnly ? pathLeaf(workspaceCwd) : projectName || t.rightSidebar.noProjectTitle,
    named: Boolean(projectName) || folderOnly,
    workspaceCwd
  }
}

export function SimpleChatContext({ cwd, profile, storedSessionId }: SimpleChatContextProps) {
  const { t } = useI18n()
  const scope = useComposerScope()
  const profiles = useStore($profiles)
  const colors = useStore($profileColors)
  const owner = useChatOwner(storedSessionId, profile)
  const project = useChatProject(storedSessionId, cwd, owner)
  const { profileKey, sameConnection } = owner

  const owningProfile = sameConnection
    ? profiles.find(candidate => normalizeProfileKey(candidate.name) === profileKey)
    : undefined

  const profileName = owningProfile ? profileLabel(owningProfile) : profileKey

  return (
    <dl className="flex min-w-0 flex-wrap items-center gap-1.5 text-xs" data-chat-context="">
      <div className="min-w-0 max-w-full">
        <dt className="sr-only">{t.sidebar.filter.profile}</dt>
        {!storedSessionId && scope.target === 'main' ? (
          <dd>
            <ProfileSwitcher placement="composer" />
          </dd>
        ) : (
          <Tip label={t.sidebar.row.ownedByProfile(profileName)}>
            <dd className="flex min-w-0 items-center gap-1.5 px-2 py-1 text-(--ui-text-secondary)">
              <ProfileGlyph
                aria-hidden="true"
                color={sameConnection ? resolveProfileColor(profileKey, colors) : profileColor(profileKey)}
                defaultStyle="initial"
                isDefault={profileKey === 'default'}
                name={profileName}
              />
              <span className="truncate">{profileName}</span>
            </dd>
          </Tip>
        )}
      </div>
      {(project.named || !storedSessionId) && (
        <div className="min-w-0 max-w-full">
          <dt className="sr-only">{project.folderOnly ? t.simpleWorkspace.folder : t.sidebar.filter.project}</dt>
          <dd
            className="flex min-w-0 items-center gap-1.5 px-2 py-1 text-(--ui-text-secondary)"
            title={displayPath(project.workspaceCwd) || project.label}
          >
            {project.named && <Codicon name="folder" size="0.75rem" />}
            <span className="truncate">
              {project.folderOnly ? `${t.simpleWorkspace.folder} · ${project.label}` : project.label}
            </span>
          </dd>
        </div>
      )}
    </dl>
  )
}
