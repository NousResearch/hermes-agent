import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useState } from 'react'

import { type NewSessionPlacement, type NewSessionSplitHandler, startNewSessionDrag } from '@/app/chat/new-session-drag'
import { Codicon } from '@/components/ui/codicon'
import { ProfileGlyph } from '@/components/ui/profile-glyph'
import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { displayPath } from '@/lib/display-path'
import { useStoreSelector } from '@/lib/use-session-slice'
import { setWorkspaceNodeOpen } from '@/store/layout'
import { notifyError } from '@/store/notifications'
import { newSessionInProfile, pinNewChatProfile, selectProfile } from '@/store/profile'
import { switchBranchInRepo } from '@/store/projects'
import { $sessionProfilesUsage } from '@/store/session'
import { $sidebarSessionRankIds } from '@/store/sidebar-sort'

import { SidebarGroupRow, SidebarRowLead, SidebarRowLink, SidebarRowStack } from '../chrome'
import { rankSessions } from '../order'

import { PROJECT_PREVIEW_COUNT, SIDEBAR_GROUP_PAGE, useWorkspaceNodeOpen } from './model'
import type { SidebarSessionGroup } from './workspace-groups'
import {
  WorkspaceAddButton,
  WorkspaceContextMenu,
  WorkspaceHeader,
  WorkspaceMenu,
  WorkspaceShowMoreButton
} from './workspace-header'

interface SidebarWorkspaceGroupProps {
  group: SidebarSessionGroup
  renderRows: (sessions: SessionInfo[]) => React.ReactNode
  onNewSession?: (path: null | string) => void
  onNewSessionSplit?: NewSessionSplitHandler
  // When set (linked worktree rows), shows a remove affordance that runs a real
  // `git worktree remove`.
  onRemove?: () => void
}

export function SidebarWorkspaceGroup({
  group,
  renderRows,
  onNewSession,
  onNewSessionSplit,
  onRemove
}: SidebarWorkspaceGroupProps) {
  const { t } = useI18n()
  const s = t.sidebar
  const isProfileGroup = group.mode === 'profile'
  // Totals for the whole profile, not the loaded page — a selector so a refresh
  // that leaves this profile's spend unchanged doesn't repaint its header.
  const usage = useStoreSelector($sessionProfilesUsage, all => all[group.id])
  const rankIds = useStore($sidebarSessionRankIds)
  // Empty worktree/branch lanes start collapsed — they only show a "No sessions
  // yet" placeholder, so defaulting them open just adds noise. Profile lanes and
  // lanes that already hold sessions default open.
  const defaultOpen = isProfileGroup || group.sessions.length > 0
  const [open, toggleOpen] = useWorkspaceNodeOpen(group.id, defaultOpen)
  const [visibleCount, setVisibleCount] = useState(SIDEBAR_GROUP_PAGE)

  // A lane ranks by whatever the sort key says before it trims itself, so the
  // rows it hides are the ones the sort ranked last.
  const sessions = rankSessions(group.sessions, rankIds)
  // A profile previews the same handful a project does, and clicking its label
  // is how you see the rest. Workspace groups page within what's loaded.
  const visibleSessions = sessions.slice(0, isProfileGroup ? PROJECT_PREVIEW_COUNT : visibleCount)
  const hiddenCount = isProfileGroup ? 0 : sessions.length - visibleSessions.length
  const nextCount = Math.min(SIDEBAR_GROUP_PAGE, hiddenCount)

  // Leading glyph: a home mark for the repo's primary checkout (labeled by its
  // live branch), a branch/kanban mark otherwise.
  const leadingIcon = (
    <Codicon
      className="shrink-0 text-(--ui-text-tertiary)"
      name={group.isKanban ? 'checklist' : group.isHome ? 'home' : 'git-branch'}
      size="0.75rem"
    />
  )

  const prepareWorkspaceTarget = async () => {
    // Reveal the lane the new session targets — an empty worktree/branch lane
    // starts collapsed, so without this the session lands in a folder the user
    // can't see. Stable across the lane's default flipping open once populated.
    setWorkspaceNodeOpen(group.id, true)

    if (isProfileGroup) {
      pinNewChatProfile(group.id)

      return true
    }

    // Main-checkout lanes are branch-labeled views over the same repo root path.
    // Clicking "+" on `main` should open on `main`, not whatever branch the root
    // currently sits on (`test0`, etc.), so explicitly switch first.
    if (group.isMain && group.path && group.label) {
      try {
        await switchBranchInRepo(group.path, group.label)
      } catch (err) {
        notifyError(err, t.statusStack.coding.switchFailed(group.label))

        return false
      }
    }

    return true
  }

  const handleNewSession = async () => {
    if (isProfileGroup) {
      setWorkspaceNodeOpen(group.id, true)
      newSessionInProfile(group.id)

      return
    }

    if (!onNewSession || !(await prepareWorkspaceTarget())) {
      return
    }

    onNewSession(group.path)
  }

  const handleNewSessionSplit = async (placement: NewSessionPlacement) => {
    if (!onNewSessionSplit || !(await prepareWorkspaceTarget())) {
      return
    }

    onNewSessionSplit(placement.dir, {
      anchor: placement.anchor,
      before: placement.before,
      cwd: group.path,
      profile: isProfileGroup ? group.id : placement.profile
    })
  }

  // Profile groups start a fresh session in that profile but keep the
  // all-profiles browse view; workspace groups seed the new session's cwd.
  // Main checkout lanes are branch-targeted.
  const addButton = (onNewSession || isProfileGroup) && (
    <WorkspaceAddButton
      label={s.newSessionIn(group.label)}
      onClick={() => void handleNewSession()}
      onPointerDown={
        onNewSessionSplit
          ? event => {
              // Drag the "+" onto a chat zone: create the session pinned to
              // this lane's cwd (or profile for profile groups), exactly where
              // it's dropped. A sub-threshold release falls through to the
              // onClick above.
              startNewSessionDrag(placement => void handleNewSessionSplit(placement), event, {
                cwd: group.path,
                label: s.newSessionIn(group.label),
                profile: isProfileGroup ? group.id : undefined
              })
            }
          : undefined
      }
    />
  )

  return (
    <SidebarRowStack>
      {isProfileGroup ? (
        // A profile heads its sessions the way a project does, so it takes the
        // project row's shape rather than the tree caption the lanes below use.
        <SidebarGroupRow
          actions={addButton}
          // Clicking a profile scopes the sidebar to it, the way clicking a
          // project enters that project. Capitalized to sit level with the
          // project labels it alternates with (`Home`, and whatever the user
          // named theirs) — profile keys are stored lowercase.
          label={
            <SidebarRowLink
              aria-label={t.profiles.switchToProfile(group.label)}
              labelClassName="capitalize hover:text-foreground hover:underline"
              onClick={() => selectProfile(group.id)}
            >
              {group.label}
            </SidebarRowLink>
          }
          lead={
            <SidebarRowLead>
              {/* Fills the lead cell like a project's icon does: the glyph's own
                  16px would sit 2px proud of the 14px column. */}
              <ProfileGlyph
                className="size-full"
                color={group.color ?? null}
                isDefault={group.id === 'default'}
                name={group.label}
              />
            </SidebarRowLead>
          }
          toggle={{ ariaLabel: s.projects.toggle(group.label, !open), onToggle: toggleOpen, open }}
          totals={{ costUsd: usage?.cost_usd ?? 0, tokens: usage?.tokens ?? 0 }}
        />
      ) : (
        <WorkspaceContextMenu onRemove={onRemove} path={group.path}>
          <WorkspaceHeader
            action={
              (onNewSession || onRemove) && (
                <div className="flex items-center">
                  {addButton}
                  {onRemove && <WorkspaceMenu onRemove={onRemove} path={group.path} />}
                </div>
              )
            }
            icon={leadingIcon}
            label={group.label}
            onToggle={toggleOpen}
            open={open}
            title={group.path ? displayPath(group.path) : undefined}
          />
        </WorkspaceContextMenu>
      )}
      {open && (
        <>
          {visibleSessions.length === 0 ? (
            <div className="min-h-7 pl-2 text-[0.75rem] leading-7 text-(--ui-text-quaternary)">{s.noSessions}</div>
          ) : (
            renderRows(visibleSessions)
          )}
          {hiddenCount > 0 && (
            <WorkspaceShowMoreButton
              count={nextCount}
              label={group.label}
              onClick={() => setVisibleCount(count => count + SIDEBAR_GROUP_PAGE)}
            />
          )}
        </>
      )}
    </SidebarRowStack>
  )
}
