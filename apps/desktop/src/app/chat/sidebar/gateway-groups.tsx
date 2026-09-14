import type { useSensors } from '@dnd-kit/core'
import { arrayMove } from '@dnd-kit/sortable'
import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'
import { useState } from 'react'

import { type NewSessionSplitHandler, startNewSessionDrag } from '@/app/chat/new-session-drag'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Input } from '@/components/ui/input'
import { ProfileGlyph } from '@/components/ui/profile-glyph'
import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { useStoreSelector } from '@/lib/use-session-slice'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { newSessionInAgent, newSessionInProfile } from '@/store/profile'
import { $sessionProfilesUsage } from '@/store/session'
import { $sidebarSessionRankIds } from '@/store/sidebar-sort'

import { SidebarGroupRow, SidebarRowGrab, SidebarRowLink, SidebarRowStack } from './chrome'
import {
  $gatewayGroupAliases,
  $gatewayGroupCollapsed,
  $gatewayGroupOrder,
  renameGatewayGroup,
  reorderGatewayGroups,
  toggleGatewayGroup
} from './gateway-group-preferences'
import { rankSessions } from './order'
import { SIDEBAR_GROUP_PAGE } from './projects/model'
import type { SidebarSessionGroup } from './projects/workspace-groups'
import { WorkspaceAddButton, WorkspaceShowMoreButton } from './projects/workspace-header'
import { ReorderableList, useSortableBindings } from './reorderable-list'

interface GatewayProfileGroupsProps {
  groups: SidebarSessionGroup[]
  renderRows: (sessions: SessionInfo[]) => ReactNode
  sensors?: ReturnType<typeof useSensors>
  onNewSessionSplit?: NewSessionSplitHandler
  nested?: boolean
}

export function GatewayProfileGroups({
  groups,
  renderRows,
  sensors,
  onNewSessionSplit,
  nested = false
}: GatewayProfileGroupsProps) {
  const registry = useStore($connectionsRegistry)
  const order = useStore($gatewayGroupOrder)
  const gatewayProfiles = new Map<string, SidebarSessionGroup[]>()
  const sections: SidebarSessionGroup[] = []

  for (const group of groups) {
    // Unknown legacy ownership stays unassigned; never guess a local gateway.
    if (nested || !group.connectionId) {
      sections.push(nested ? { ...group, label: group.profile! } : group)

      continue
    }

    const id = JSON.stringify(['gateway', group.connectionId])
    const profiles = gatewayProfiles.get(id)

    if (profiles) {
      profiles.push(group)
    } else {
      gatewayProfiles.set(id, [group])
      sections.push({
        id,
        connectionId: group.connectionId,
        label:
          registry?.connections.find(connection => connection.id === group.connectionId)?.label || group.connectionId,
        mode: 'profile',
        path: null,
        sessions: []
      })
    }
  }

  const ordered = [...sections].sort((a, b) => {
    const left = order.indexOf(a.id)
    const right = order.indexOf(b.id)

    return (left < 0 ? Infinity : left) - (right < 0 ? Infinity : right) || 0
  })

  const ids = ordered.map(group => group.id)

  return (
    <ReorderableList ids={ids} onReorder={reorderGatewayGroups} sensors={sensors}>
      {ordered.map((group, index) => (
        <GatewayProfileGroup
          first={index === 0}
          group={group}
          key={group.id}
          last={index === ordered.length - 1}
          onMove={direction => reorderGatewayGroups(arrayMove(ids, index, index + direction))}
          onNewSessionSplit={onNewSessionSplit}
          renderRows={renderRows}
        >
          {gatewayProfiles.has(group.id) && (
            <div className="ml-3 border-l border-border/50 pl-1">
              <GatewayProfileGroups
                groups={gatewayProfiles.get(group.id)!}
                nested
                onNewSessionSplit={onNewSessionSplit}
                renderRows={renderRows}
                sensors={sensors}
              />
            </div>
          )}
        </GatewayProfileGroup>
      ))}
    </ReorderableList>
  )
}

interface GatewayProfileGroupProps {
  group: SidebarSessionGroup
  renderRows: (sessions: SessionInfo[]) => ReactNode
  onMove: (direction: 1 | -1) => void
  onNewSessionSplit?: NewSessionSplitHandler
  first: boolean
  last: boolean
  children?: ReactNode
}

function GatewayProfileGroup({
  group,
  renderRows,
  onMove,
  first,
  last,
  onNewSessionSplit,
  children
}: GatewayProfileGroupProps) {
  const { t } = useI18n()
  const s = t.sidebar
  const copy = s.gatewayGroups
  const aliases = useStore($gatewayGroupAliases)
  const collapsed = useStore($gatewayGroupCollapsed)
  const rankIds = useStore($sidebarSessionRankIds)
  // Legacy totals are keyed only by profile. Never attribute those figures to
  // a registry gateway that happens to expose the same profile name.
  const usage = useStoreSelector($sessionProfilesUsage, all => (group.connectionId ? undefined : all[group.profile!]))
  const [renaming, setRenaming] = useState(false)
  const [draft, setDraft] = useState('')
  const [visibleCount, setVisibleCount] = useState(SIDEBAR_GROUP_PAGE)
  const sortable = useSortableBindings(group.id)
  const label = aliases[group.id] || group.label
  const open = !collapsed.includes(group.id)
  const sessions = rankSessions(group.sessions, rankIds)
  const hiddenCount = Math.max(0, sessions.length - visibleCount)
  const route = group.connectionId ? { connectionId: group.connectionId, profile: group.profile! } : undefined

  const startSession = () => {
    if (!open) {
      toggleGatewayGroup(group.id)
    }

    const profile = group.profile!

    if (group.connectionId) {
      newSessionInAgent({ connectionId: group.connectionId, profile })
    } else {
      newSessionInProfile(profile)
    }
  }

  return (
    <SidebarRowStack
      data-gateway-group={group.profile ? group.id : undefined}
      data-gateway-section={!group.profile ? group.id : undefined}
      ref={sortable.ref}
      style={sortable.style}
    >
      <SidebarGroupRow
        actions={
          <div className="flex items-center">
            {group.profile && (
              <WorkspaceAddButton
                label={s.newSessionIn(label)}
                onClick={startSession}
                onPointerDown={
                  onNewSessionSplit
                    ? event =>
                        startNewSessionDrag(
                          placement => {
                            if (!open) {
                              toggleGatewayGroup(group.id)
                            }

                            onNewSessionSplit(placement.dir, {
                              anchor: placement.anchor,
                              before: placement.before,
                              profile: group.profile,
                              route
                            })
                          },
                          event,
                          { label: s.newSessionIn(label), profile: group.profile, route }
                        )
                    : undefined
                }
              />
            )}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button aria-label={`${copy.actions}: ${label}`} size="icon-xs" variant="ghost">
                  <Codicon name="ellipsis" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem
                  onSelect={() => {
                    setDraft(aliases[group.id] || '')
                    setRenaming(true)
                  }}
                >
                  {copy.rename}
                </DropdownMenuItem>
                <DropdownMenuItem disabled={!aliases[group.id]} onSelect={() => renameGatewayGroup(group.id, '')}>
                  {copy.resetName}
                </DropdownMenuItem>
                <DropdownMenuItem disabled={first} onSelect={() => onMove(-1)}>
                  {copy.moveUp}
                </DropdownMenuItem>
                <DropdownMenuItem disabled={last} onSelect={() => onMove(1)}>
                  {copy.moveDown}
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        }
        label={
          <SidebarRowLink aria-expanded={open} onClick={() => toggleGatewayGroup(group.id)}>
            {label}
          </SidebarRowLink>
        }
        lead={
          <SidebarRowGrab
            ariaLabel={`${copy.reorder}: ${label}`}
            dragging={sortable.dragging}
            dragHandleProps={sortable.dragHandleProps}
          >
            {group.profile ? (
              <ProfileGlyph
                className="size-full"
                color={group.color ?? null}
                isDefault={group.profile === 'default'}
                name={group.profile}
              />
            ) : (
              <Codicon name="remote" />
            )}
          </SidebarRowGrab>
        }
        toggle={{ ariaLabel: s.projects.toggle(label, !open), onToggle: () => toggleGatewayGroup(group.id), open }}
        totals={usage ? { costUsd: usage.cost_usd, tokens: usage.tokens } : undefined}
      />
      {open && (
        <>
          {children}
          {renderRows(sessions.slice(0, visibleCount))}
          {hiddenCount > 0 && (
            <WorkspaceShowMoreButton
              count={Math.min(SIDEBAR_GROUP_PAGE, hiddenCount)}
              label={label}
              onClick={() => setVisibleCount(count => count + SIDEBAR_GROUP_PAGE)}
            />
          )}
        </>
      )}
      <Dialog onOpenChange={setRenaming} open={renaming}>
        <DialogContent>
          <form
            onSubmit={event => {
              event.preventDefault()
              renameGatewayGroup(group.id, draft)
              setRenaming(false)
            }}
          >
            <DialogHeader>
              <DialogTitle>{copy.rename}</DialogTitle>
              <DialogDescription>{copy.aliasHint}</DialogDescription>
            </DialogHeader>
            <Input
              aria-label={copy.aliasLabel}
              autoFocus
              maxLength={120}
              onChange={event => setDraft(event.target.value)}
              placeholder={group.label}
              value={draft}
            />
            <DialogFooter>
              <Button onClick={() => setRenaming(false)} type="button" variant="ghost">
                {t.common.cancel}
              </Button>
              <Button type="submit">{t.common.save}</Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </SidebarRowStack>
  )
}
