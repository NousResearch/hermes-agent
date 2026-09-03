import {
  closestCenter,
  DndContext,
  type DragEndEvent,
  type DragOverEvent,
  type DragStartEvent,
  KeyboardSensor,
  type Modifier,
  PointerSensor,
  useSensor,
  useSensors
} from '@dnd-kit/core'
import {
  arrayMove,
  horizontalListSortingStrategy,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable
} from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import { useStore } from '@nanostores/react'
import { Fragment, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router'

import type { ProfileScope } from '@/api/client'
import { CodeEditor } from '@/components/chat/code-editor'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ColorSwatches } from '@/components/ui/color-swatches'
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from '@/components/ui/context-menu'
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  dropdownMenuSectionLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { Popover, PopoverAnchor, PopoverContent } from '@/components/ui/popover'
import { ProfileGlyph } from '@/components/ui/profile-glyph'
import { Tip, Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import type { DesktopRegistryConnection } from '@/global'
import { getProfileSoul, updateProfileSoul } from '@/hermes'
import { useI18n } from '@/i18n'
import { sortConnectionsForDisplay } from '@/lib/connection-display'
import { triggerHaptic } from '@/lib/haptics'
import { Loader2 } from '@/lib/icons'
import { PROFILE_SWATCHES, profileColorSoft, resolveProfileColor } from '@/lib/profile-color'
import {
  REORDER_DRAG_TRANSITION_CSS,
  REORDER_RAIL_TRANSITION,
  reorderCommitHaptic,
  reorderStepHaptic
} from '@/lib/reorder'
import { cn } from '@/lib/utils'
import {
  $activeConnectionId,
  $connectionsRegistry,
  $hasMultipleConnections,
  selectConnection
} from '@/store/connections'
import { $fleetRoster, refreshFleetRoster } from '@/store/fleet-roster'
import { notify, notifyError } from '@/store/notifications'
import {
  $activeGatewayProfile,
  $profileColors,
  $profileCreateRequest,
  $profileOrder,
  $profiles,
  $profileScope,
  ALL_PROFILES,
  normalizeProfileKey,
  profileLabel,
  refreshActiveProfile,
  selectProfile,
  setProfileColor,
  setProfileOrder,
  setShowAllProfiles,
  sortByProfileOrder
} from '@/store/profile'
import {
  $profileRemoteOverrides,
  openRemoteOverrideDialog,
  refreshProfileRemoteOverrides
} from '@/store/profile-remote-override'
import { runExportProfileFlow, runImportProfileFlow } from '@/store/profile-share'
import type { ProfileInfo } from '@/types/hermes'

import { CreateProfileDialog } from '../../profiles/create-profile-dialog'
import { DeleteProfileDialog } from '../../profiles/delete-profile-dialog'
import { RenameProfileDialog } from '../../profiles/rename-profile-dialog'
import { PROFILES_ROUTE, SETTINGS_ROUTE } from '../../routes'

import { ConnectionGlyph } from './connection-glyph'
import { buildRestGroups, countRestAgents, type FleetAgent, type FleetGroup, fleetRouteKey } from './fleet-rail'
import { ProfileRemoteOverrideDialog } from './profile-remote-override-dialog'
import { useFleetRoster } from './use-fleet-roster'
import { useProfilePrewarm } from './use-profile-prewarm'
import { useProfileRailRefreshOnActive } from './use-profile-rail-refresh-on-active'

const RAIL_GAP = 4 // px — matches gap-1 between squares.

// Past this many profiles the strip of colored squares stops scaling (tiny
// drag targets, endless horizontal scroll), so the rail collapses to a compact
// menu. Drag-reorder and long-press-recolor live only on the squares path.
const PROFILE_DROPDOWN_THRESHOLD = 13

// Neighbors reflow on RAIL_TRANSITION; the dragged square glides between
// snapped cells on the snappier DRAG_TRANSITION. Both come from the SHARED
// reorder primitive (lib/reorder.ts) so every reorder strip feels identical.
const RAIL_TRANSITION = REORDER_RAIL_TRANSITION
const DRAG_TRANSITION = REORDER_DRAG_TRANSITION_CSS

// The rail is a single horizontal strip of fixed cells. Pin drags to the x-axis
// (no cross-axis scrollbar), snap to whole cells so a square steps slot-to-slot
// instead of gliding, and clamp to the occupied strip so it can't float past the
// last profile onto the "+".
const stepThroughCells: Modifier = ({ containerNodeRect, draggingNodeRect, transform }) => {
  if (!draggingNodeRect || !containerNodeRect) {
    return { ...transform, y: 0 }
  }

  const pitch = draggingNodeRect.width + RAIL_GAP
  const minX = containerNodeRect.left - draggingNodeRect.left
  const maxX = containerNodeRect.right - draggingNodeRect.right
  const snapped = Math.round(transform.x / pitch) * pitch

  return { ...transform, x: Math.min(maxX, Math.max(minX, snapped)), y: 0 }
}

// Arc-Spaces-style profile rail at the sidebar foot: a default↔all toggle pinned
// left, the colored named profiles scrolling between, and Manage pinned right.
// The active profile pops in its own color — the "where am I" cue.
//
// With one registered gateway this is the whole story. With several, the rail
// becomes the FLEET rail: the active gateway's profiles stay exactly as they
// are, and every other registered gateway follows on the same strip as an
// at-rest group — a hairline, that gateway's kind glyph, its default home
// square and its named squares, dimmed. Clicking an at-rest square performs
// the same re-home the statusbar switcher does, landing on that exact
// (gateway, profile); the workspace still lives on one gateway at a time, only
// the picker spans the fleet. Groups keep registry order regardless of which
// one is active, so a square never moves under the pointer that clicked it.
export function ProfileRail() {
  const { t } = useI18n()
  const p = t.profiles
  const profiles = useStore($profiles)
  const scope = useStore($profileScope)
  const gatewayProfile = useStore($activeGatewayProfile)
  const order = useStore($profileOrder)
  const colors = useStore($profileColors)
  const remoteOverrides = useStore($profileRemoteOverrides)
  const multipleConnections = useStore($hasMultipleConnections)
  const registry = useStore($connectionsRegistry)
  const activeConnectionId = useStore($activeConnectionId)
  const roster = useStore($fleetRoster)
  const navigate = useNavigate()
  const [createOpen, setCreateOpen] = useState(false)
  const [pendingRename, setPendingRename] = useState<null | ProfileInfo>(null)
  const [pendingDelete, setPendingDelete] = useState<null | ProfileInfo>(null)
  const [pendingSoul, setPendingSoul] = useState<null | string>(null)
  // Fleet-side counterparts: the at-rest square being acted on. Its route is
  // the dialog's scope, so the edit executes on the owning gateway.
  const [pendingRestRename, setPendingRestRename] = useState<null | FleetAgent>(null)
  const [pendingRestDelete, setPendingRestDelete] = useState<null | FleetAgent>(null)
  const [pendingRestSoul, setPendingRestSoul] = useState<null | FleetAgent>(null)
  // Route key of the at-rest square whose switch is dialing (spinner on that
  // square, not in the statusbar — the previous source stays painted).
  const [pendingRoute, setPendingRoute] = useState<null | string>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  useFleetRoster(multipleConnections)

  const connections = registry?.connections

  const restGroups = useMemo(
    () => (multipleConnections ? buildRestGroups({ activeConnectionId, connections: connections ?? [], roster }) : []),
    [activeConnectionId, connections, multipleConnections, roster]
  )

  // Fleet mode needs something to show beside the active gateway. Two
  // registrations of one backend collapse to a single roster source, which
  // keeps the rail on its single-gateway path.
  const fleet = restGroups.length > 0

  // Registry order for the whole strip, active group included — the active
  // gateway keeps its slot instead of jumping to the front on a switch.
  const activeConnection = connections?.find(connection => connection.id === activeConnectionId) ?? null

  const fleetSequence = useMemo(() => {
    const byId = new Map(restGroups.map(group => [group.connectionId, group]))
    const ordered = sortConnectionsForDisplay(connections ?? [])
    const sequence: Array<{ kind: 'active' } | { group: FleetGroup; kind: 'rest' }> = []
    let activePlaced = false

    for (const connection of ordered) {
      if (connection.id === activeConnectionId) {
        sequence.push({ kind: 'active' })
        activePlaced = true
      } else {
        const group = byId.get(connection.id)

        if (group) {
          sequence.push({ group, kind: 'rest' })
        }
      }
    }

    // Legacy primary path publishes no connection id: the active gateway is
    // unknown to the registry, so it leads the strip.
    if (!activePlaced) {
      sequence.unshift({ kind: 'active' })
    }

    return sequence
  }, [activeConnectionId, connections, restGroups])

  // Too many profiles for the square strip → collapse to the select. Declared
  // ahead of the wheel effect, which re-binds when the strip mounts/unmounts.
  // The threshold counts the whole fleet: fourteen squares are fourteen
  // squares wherever they live.
  const condensed = profiles.length + countRestAgents(restGroups) > PROFILE_DROPDOWN_THRESHOLD

  const switchToRest = (agent: FleetAgent) => {
    const key = fleetRouteKey(agent.connectionId, agent.profile)
    triggerHaptic('selection')
    setPendingRoute(key)

    void selectConnection(agent.connectionId, { profile: agent.profile })
      .catch((error: unknown) => notifyError(error, p.switchConnectionFailed(agent.connectionLabel)))
      .finally(() => setPendingRoute(current => (current === key ? null : current)))
  }

  const restScope = (agent: FleetAgent): ProfileScope => ({ connectionId: agent.connectionId, profile: agent.profile })

  // A plain mouse wheel only emits deltaY; map it to horizontal scroll so the
  // rail is navigable without a trackpad. Trackpad x-scroll (deltaX) passes
  // through. Native + non-passive so we can preventDefault and not bleed the
  // gesture into the sessions list above.
  useEffect(() => {
    const el = scrollRef.current

    if (!el) {
      return
    }

    const onWheel = (event: WheelEvent) => {
      if (el.scrollWidth <= el.clientWidth || Math.abs(event.deltaY) <= Math.abs(event.deltaX)) {
        return
      }

      el.scrollLeft += event.deltaY
      event.preventDefault()
    }

    el.addEventListener('wheel', onWheel, { passive: false })

    return () => el.removeEventListener('wheel', onWheel)
    // `condensed` swaps the strip out for the dropdown (ref goes null/back).
  }, [condensed])

  const isAll = scope === ALL_PROFILES
  const activeKey = normalizeProfileKey(gatewayProfile)
  const defaultProfile = profiles.find(profile => profile.is_default)
  const onDefault = !isAll && activeKey === 'default'

  const named = sortByProfileOrder(
    profiles.filter(profile => !profile.is_default),
    order
  )

  const multiProfile = profiles.length > 1

  // distance constraint: a small drag reorders, a tap still selects the profile.
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 4 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
  )

  // Tick a haptic each time the drag crosses into a new cell, and a satisfying
  // confirm on a committed reorder.
  const lastOverRef = useRef<string | null>(null)

  const handleDragStart = ({ active }: DragStartEvent) => {
    lastOverRef.current = String(active.id)
  }

  const handleDragOver = ({ over }: DragOverEvent) => {
    const id = over ? String(over.id) : null

    if (id && id !== lastOverRef.current) {
      lastOverRef.current = id
      reorderStepHaptic()
    }
  }

  const handleDragEnd = ({ active, over }: DragEndEvent) => {
    lastOverRef.current = null

    if (!over || active.id === over.id) {
      return
    }

    const ids = named.map(profile => profile.name)
    const from = ids.indexOf(String(active.id))
    const to = ids.indexOf(String(over.id))

    if (from >= 0 && to >= 0) {
      setProfileOrder(arrayMove(ids, from, to))
      reorderCommitHaptic()
    }
  }

  // Re-pull the running profile + list on mount, and again whenever the window
  // regains focus/visibility -- a profile created, deleted, or renamed by
  // another surface (Manage Profiles, another window, the CLI) leaves this
  // rail's cached $profiles stale until something re-fetches it. See
  // use-profile-rail-refresh-on-active.ts for the extracted (and tested)
  // wiring.
  useProfileRailRefreshOnActive()

  // Which profiles carry a per-profile remote override (connection.json
  // profiles.<name>) — refreshed whenever the profile list changes so the
  // rail's "remote" badge tracks create/rename/override edits.
  const profileNames = profiles.map(profile => profile.name)
  const profileNamesKey = profileNames.join('\u0000')

  useEffect(() => {
    void refreshProfileRemoteOverrides(profileNamesKey ? profileNamesKey.split('\u0000') : [])
  }, [profileNamesKey])

  // Open the create dialog when the `profile.create` hotkey fires (the dialog
  // state lives here, so the global keybind bumps a request atom we watch).
  const createRequest = useStore($profileCreateRequest)
  const lastCreateRef = useRef(createRequest)

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    if (createRequest === lastCreateRef.current) {
      return
    }

    lastCreateRef.current = createRequest
    setCreateOpen(true)
  }, [createRequest])

  // The sortable strip of the active gateway's named profiles (unchanged
  // from the single-gateway rail; fleet mode only decides where it sits).
  const activeStrip = (
    <>
      {multiProfile && (
        <DndContext
          collisionDetection={closestCenter}
          modifiers={[stepThroughCells]}
          onDragEnd={handleDragEnd}
          onDragOver={handleDragOver}
          onDragStart={handleDragStart}
          sensors={sensors}
        >
          <SortableContext items={named.map(profile => profile.name)} strategy={horizontalListSortingStrategy}>
            {/* relative → the strip is the dragged square's offsetParent, so the
              clamp modifier bounds drags to the occupied cells (not the +). */}
            <div className="relative flex items-center gap-1">
              {named.map(profile => (
                <ProfileSquare
                  active={!isAll && normalizeProfileKey(profile.name) === activeKey}
                  color={resolveProfileColor(profile.name, colors)}
                  key={profile.name}
                  label={profileLabel(profile)}
                  // The legacy per-profile remote override predates the
                  // gateway registry; once the rail shows machines directly
                  // it only confuses, so it is offered on single-gateway
                  // setups only.
                  onConnectRemote={multipleConnections ? undefined : () => openRemoteOverrideDialog(profile.name)}
                  onDelete={() => setPendingDelete(profile)}
                  onEditSoul={() => setPendingSoul(profile.name)}
                  onRecolor={color => setProfileColor(profile.name, color)}
                  onRename={() => setPendingRename(profile)}
                  onSelect={() => selectProfile(profile.name)}
                  remoteHost={remoteOverrides[normalizeProfileKey(profile.name)]?.host ?? null}
                />
              ))}
            </div>
          </SortableContext>
        </DndContext>
      )}
    </>
  )

  return (
    // `data-tour` as well as `data-slot`: only the former is identity to the
    // tour collector and the tip catalog, and the rail's one other durable
    // handle is a TRANSLATED aria-label, which stops matching the moment the
    // app isn't in English.
    <div
      aria-label={p.title}
      className="flex min-w-0 items-center gap-0.5"
      data-slot="profile-rail"
      data-tip-region=""
      data-tour="profile-rail"
      role="group"
    >
      {/* Fleet: every gateway carries its own home square inside its group, so
          the pinned pill is purely the "all profiles on this gateway" toggle. */}
      {fleet && (
        <ProfilePill
          active={isAll}
          glyph="layers"
          label={p.fleet.allOnGateway}
          onSelect={() => setShowAllProfiles(true)}
        />
      )}

      {/* One button toggles default ↔ all: home face when scoped to a profile,
          layers face when showing everything. Pinned left like Manage is right.
          Hidden until a second profile exists. */}
      {!fleet &&
        multiProfile &&
        (defaultProfile ? (
          // On default → toggle to all. Anywhere else (all view or a named
          // profile) → return to default. So leaving a profile never lands on all.
          <ProfilePill
            active={isAll || onDefault}
            glyph={isAll ? 'layers' : 'home'}
            label={onDefault ? p.showAllProfiles : p.switchToProfile(profileLabel(defaultProfile))}
            onSelect={() => (onDefault ? setShowAllProfiles(true) : selectProfile(defaultProfile.name))}
          />
        ) : (
          <ProfilePill active={isAll} glyph="layers" label={p.allProfiles} onSelect={() => setShowAllProfiles(true)} />
        ))}

      {/* Single-profile: the active default's home icon next to the create +. */}
      {!fleet && !multiProfile && defaultProfile && (
        <ProfilePill
          active
          glyph="home"
          label={profileLabel(defaultProfile)}
          onSelect={() => selectProfile(defaultProfile.name)}
        />
      )}

      {condensed ? (
        // Condensed path: one compact dropdown instead of N squares. No drag
        // reorder, no long-press recolor, no per-square context menu — Manage
        // covers rename/delete at this scale.
        <div className="flex min-w-0 flex-1 items-center gap-1">
          {/* In fleet mode the pinned pill is an all-profiles action, not the
              default↔all toggle used by the single-gateway rail. The expanded
              fleet strip carries its own default home square, so the condensed
              dropdown must carry that same route or default becomes unreachable
              as soon as the whole fleet crosses the collapse threshold. */}
          <ProfileDropdown
            activeKey={isAll ? null : activeKey}
            colors={colors}
            onCreate={() => setCreateOpen(true)}
            onImport={() => void runImportProfileFlow()}
            onSelect={selectProfile}
            onSelectRest={switchToRest}
            profiles={fleet && defaultProfile ? [defaultProfile, ...named] : named}
            restGroups={restGroups}
          />
        </div>
      ) : (
        <div
          className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
          ref={scrollRef}
        >
          {/* The active gateway's squares. In fleet mode they sit in the
              gateway's registry slot with a home square at their head, so the
              strip keeps one shape whichever gateway is active. */}
          {fleet
            ? fleetSequence.map((entry, index) =>
                entry.kind === 'active' ? (
                  <Fragment key="active">
                    <FleetDivider
                      connection={activeConnection}
                      first={index === 0}
                      label={activeConnection ? p.fleet.gateway(activeConnection.label) : null}
                      reachable
                    />
                    <span
                      aria-label={activeConnection ? p.fleet.gateway(activeConnection.label) : undefined}
                      className="flex shrink-0 items-center gap-1"
                      data-active="true"
                      data-connection-id={activeConnection?.id}
                      data-slot="profile-rail-gateway"
                      role="group"
                    >
                      {defaultProfile && (
                        <ProfilePill
                          active={onDefault}
                          glyph="home"
                          label={profileLabel(defaultProfile)}
                          onSelect={() => selectProfile(defaultProfile.name)}
                        />
                      )}
                      {activeStrip}
                    </span>
                  </Fragment>
                ) : (
                  <FleetRestGroup
                    colors={colors}
                    first={index === 0}
                    group={entry.group}
                    key={entry.group.connectionId}
                    onDelete={setPendingRestDelete}
                    onEditSoul={setPendingRestSoul}
                    onRecolor={(agent, color) => setProfileColor(agent.profile, color)}
                    onRename={setPendingRestRename}
                    onSelect={switchToRest}
                    pendingRoute={pendingRoute}
                  />
                )
              )
            : activeStrip}

          <AddProfileButton label={p.newProfile} onClick={() => setCreateOpen(true)} />
          <ImportProfileButton label={p.importProfile} />
        </div>
      )}

      {/* Always reachable, even with only the default profile: the manage
          overlay is the only place to edit a profile's SOUL.md, and a
          single-profile user must be able to edit the default's persona
          without first creating a throwaway second profile. */}
      <ProfilePill active={false} glyph="ellipsis" label={p.manageProfiles} onSelect={() => navigate(PROFILES_ROUTE)} />

      {/* Multi-gateway discoverability: before a second source exists, a plug
          pinned beside Manage deep-links to the unified Gateways page. Once
          there are several sources, the same action lives in their selector. */}
      {!multipleConnections && (
        <ProfilePill
          active={false}
          glyph="plug"
          label={p.connectGateway}
          onSelect={() => navigate(`${SETTINGS_ROUTE}?tab=gateway`)}
        />
      )}

      {/* Land in the new profile on a fresh chat (selectProfile triggers the
          new-session reset), not stuck on the session you were just in. */}
      <CreateProfileDialog
        onClose={() => setCreateOpen(false)}
        onCreated={async name => {
          await refreshActiveProfile()
          selectProfile(name)
        }}
        open={createOpen}
        profiles={profiles}
      />

      <RenameProfileDialog
        currentName={pendingRename?.name ?? ''}
        isDefault={pendingRename?.is_default ?? false}
        onClose={() => setPendingRename(null)}
        onRenamed={refreshActiveProfile}
        open={pendingRename !== null}
      />

      <DeleteProfileDialog
        onClose={() => setPendingDelete(null)}
        onDeleted={refreshActiveProfile}
        open={pendingDelete !== null}
        profile={pendingDelete}
      />

      <EditSoulDialog onClose={() => setPendingSoul(null)} profileName={pendingSoul} />

      {/* Fleet-side dialogs: scoped to the at-rest square's owning gateway, and
          they refresh the roster (not the active profile list) on success. */}
      <RenameProfileDialog
        currentName={pendingRestRename?.profile ?? ''}
        onClose={() => setPendingRestRename(null)}
        onRenamed={() => refreshFleetRoster({ force: true })}
        open={pendingRestRename !== null}
        scope={pendingRestRename ? restScope(pendingRestRename) : undefined}
      />

      <DeleteProfileDialog
        gatewayLabel={pendingRestDelete?.connectionLabel}
        onClose={() => setPendingRestDelete(null)}
        onDeleted={() => refreshFleetRoster({ force: true })}
        open={pendingRestDelete !== null}
        profile={pendingRestDelete ? { name: pendingRestDelete.profile, path: pendingRestDelete.handle } : null}
        scope={pendingRestDelete ? restScope(pendingRestDelete) : undefined}
      />

      <EditSoulDialog
        gatewayLabel={pendingRestSoul?.connectionLabel}
        onClose={() => setPendingRestSoul(null)}
        profileName={pendingRestSoul?.profile ?? null}
        scope={pendingRestSoul ? restScope(pendingRestSoul) : undefined}
      />

      <ProfileRemoteOverrideDialog profileNames={profileNames} />
    </div>
  )
}

// Right-click → Edit SOUL.md for a sidebar profile — the same in-app markdown
// editor as the memory-graph node edit, so a profile's persona is editable
// without opening the Manage overlay.
function EditSoulDialog({
  gatewayLabel,
  onClose,
  profileName,
  scope
}: {
  gatewayLabel?: string
  onClose: () => void
  profileName: null | string
  scope?: ProfileScope
}) {
  const { t } = useI18n()
  const p = t.profiles
  const [content, setContent] = useState('')
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (!profileName) {
      return
    }

    let cancelled = false
    setLoading(true)
    setContent('')

    getProfileSoul(profileName, scope)
      .then(soul => !cancelled && setContent(soul.content))
      .catch(err => !cancelled && notifyError(err, p.failedLoadSoul))
      .finally(() => !cancelled && setLoading(false))

    return () => void (cancelled = true)
  }, [p, profileName, scope])

  const save = async () => {
    if (!profileName) {
      return
    }

    setSaving(true)

    try {
      await updateProfileSoul(profileName, content, scope)
      notify({ kind: 'success', title: p.soulSaved, message: profileName })
      onClose()
    } catch (err) {
      notifyError(err, p.failedSaveSoul)
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog onOpenChange={open => !open && !saving && onClose()} open={profileName !== null}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>
            {gatewayLabel && profileName ? p.fleet.onGateway(profileName, gatewayLabel) : profileName} · SOUL.md
          </DialogTitle>
        </DialogHeader>
        <div className="h-80">
          {!loading && profileName && (
            <CodeEditor
              filePath="SOUL.md"
              framed
              initialValue={content}
              key={profileName}
              onCancel={() => !saving && onClose()}
              onChange={setContent}
              onSave={() => void save()}
            />
          )}
        </div>
        <DialogFooter>
          <Button disabled={saving} onClick={onClose} type="button" variant="ghost">
            {t.common.cancel}
          </Button>
          <Button disabled={saving || loading} onClick={() => void save()}>
            {saving ? p.saving : p.saveSoul}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

// The "+" create button, shared by both rail render paths.
function AddProfileButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <Tip label={label}>
      <button
        aria-label={label}
        className="grid size-5 shrink-0 place-items-center rounded-[3px] text-(--ui-text-tertiary) opacity-55 transition hover:bg-(--ui-control-hover-background) hover:text-foreground hover:opacity-100"
        onClick={onClick}
        type="button"
      >
        <Codicon name="add" size="0.75rem" />
      </button>
    </Tip>
  )
}

// Import-archive door beside the "+": adopt a shared profile bundle (theme,
// skills, layout) as a new profile. Same chrome as AddProfileButton; the whole
// flow (picker → import → apply overlay → switch) lives in the store.
function ImportProfileButton({ label }: { label: string }) {
  return (
    <Tip label={label}>
      <button
        aria-label={label}
        className="grid size-5 shrink-0 place-items-center rounded-[3px] text-(--ui-text-tertiary) opacity-55 transition hover:bg-(--ui-control-hover-background) hover:text-foreground hover:opacity-100"
        onClick={() => void runImportProfileFlow()}
        type="button"
      >
        <Codicon name="cloud-download" size="0.75rem" />
      </button>
    </Tip>
  )
}

// The condensed rail: every named profile in one compact menu. The trigger
// shows the active profile (tinted initial + name); on default/all scope it
// falls back to the placeholder since the left toggle pill carries that state.
function ProfileDropdown({
  activeKey,
  colors,
  onCreate,
  onImport,
  onSelect,
  onSelectRest,
  profiles,
  restGroups
}: {
  activeKey: null | string
  colors: Record<string, string>
  onCreate: () => void
  onImport: () => void
  onSelect: (name: string) => void
  onSelectRest: (agent: FleetAgent) => void
  profiles: ProfileInfo[]
  // Fleet: the other gateways' agents, each under its own section header.
  restGroups: readonly FleetGroup[]
}) {
  const { t } = useI18n()
  const p = t.profiles

  const value = activeKey ? (profiles.find(profile => normalizeProfileKey(profile.name) === activeKey)?.name ?? '') : ''
  const activeProfile = profiles.find(profile => profile.name === value)

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          aria-label={p.title}
          className="min-w-0 flex-1 justify-between overflow-hidden px-1 text-(--ui-text-secondary) data-[state=open]:bg-(--ui-control-active-background) data-[state=open]:text-foreground"
          data-slot="profile-dropdown"
          size="xs"
          type="button"
          variant="ghost"
        >
          <span className="flex min-w-0 flex-1 items-center gap-1 overflow-hidden">
            {activeProfile ? (
              <>
                <ProfileGlyph
                  aria-hidden="true"
                  color={resolveProfileColor(activeProfile.name, colors)}
                  isDefault={activeProfile.is_default}
                  name={activeProfile.name}
                />
                <span className="truncate">{profileLabel(activeProfile)}</span>
              </>
            ) : (
              <span className="truncate">{p.title}</span>
            )}
          </span>
          <Codicon aria-hidden="true" className="shrink-0 opacity-60" name="chevron-down" size="0.875rem" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="min-w-48 max-w-72" collisionPadding={8} side="top">
        <DropdownMenuItem onSelect={onCreate}>
          <Codicon aria-hidden="true" name="add" size="0.875rem" />
          <span className="truncate">{p.newProfile}</span>
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={onImport}>
          <Codicon aria-hidden="true" name="cloud-download" size="0.875rem" />
          <span className="truncate">{p.importProfile}</span>
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuRadioGroup onValueChange={name => name && onSelect(name)} value={value}>
          {profiles.map(profile => (
            <ProfileDropdownItem
              color={resolveProfileColor(profile.name, colors)}
              isDefault={profile.is_default}
              key={profile.name}
              label={profileLabel(profile)}
              name={profile.name}
            />
          ))}
        </DropdownMenuRadioGroup>
        {restGroups.map(group => (
          <div data-connection-id={group.connectionId} data-slot="profile-dropdown-gateway" key={group.connectionId}>
            <DropdownMenuSeparator />
            <DropdownMenuLabel className={cn(dropdownMenuSectionLabel, 'flex items-center gap-1.5')}>
              <ConnectionGlyph connection={group} />
              <span className="truncate">{group.label}</span>
              {!group.reachable && <span aria-hidden="true" className="size-1.5 shrink-0 rounded-full bg-amber-500" />}
            </DropdownMenuLabel>
            {[group.defaultAgent, ...group.named].map(agent => (
              <DropdownMenuItem
                aria-label={p.fleet.onGateway(agent.profile, group.label)}
                className="min-w-0"
                key={agent.profile}
                onSelect={() => onSelectRest(agent)}
              >
                <span className="flex min-w-0 items-center gap-1.5">
                  <ProfileGlyph
                    aria-hidden="true"
                    color={resolveProfileColor(agent.profile, colors)}
                    isDefault={agent.isDefault}
                    name={agent.profile}
                  />
                  <span className="truncate">{agent.profile}</span>
                </span>
              </DropdownMenuItem>
            ))}
          </div>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

// One dropdown row per profile — its own component so each row can own a
// hover-intent prewarm timer (see useProfilePrewarm).
function ProfileDropdownItem({
  color,
  isDefault,
  label,
  name
}: {
  color: null | string
  isDefault: boolean
  label: string
  name: string
}) {
  const { cancelPrewarm, startPrewarm } = useProfilePrewarm(name)

  return (
    <DropdownMenuRadioItem
      className="min-w-0"
      onPointerEnter={startPrewarm}
      onPointerLeave={cancelPrewarm}
      value={name}
    >
      <span className="flex min-w-0 items-center gap-1.5">
        <ProfileGlyph aria-hidden="true" color={color} isDefault={isDefault} name={name} />
        <span className="truncate">{label}</span>
      </span>
    </DropdownMenuRadioItem>
  )
}

interface ProfilePillProps {
  active: boolean
  // home / All / Manage are glyph action buttons (navigation, not identity).
  glyph: string
  label: string
  onSelect: () => void
  // Fleet at-rest: dimmed until hovered, like the at-rest squares beside it.
  muted?: boolean
  pending?: boolean
  slot?: string
  connectionId?: string
}

function ProfilePill({
  active,
  connectionId,
  glyph,
  label,
  muted = false,
  onSelect,
  pending = false,
  slot
}: ProfilePillProps) {
  return (
    <Tip label={label}>
      <Button
        aria-busy={pending || undefined}
        aria-label={label}
        aria-pressed={active}
        className={cn(
          'bg-transparent text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground',
          active && 'bg-(--ui-control-active-background) text-foreground',
          muted && 'opacity-40 hover:opacity-100'
        )}
        data-connection-id={connectionId}
        data-slot={slot}
        onClick={onSelect}
        size="icon-xs"
        type="button"
        variant="ghost"
      >
        {pending ? (
          <Loader2 aria-hidden="true" className="size-3 animate-spin" />
        ) : (
          <Codicon name={glyph} size="0.875rem" />
        )}
      </Button>
    </Tip>
  )
}

// The gateway marker that heads every group on the fleet rail: its kind glyph
// (device / network / terminal / cloud — the same glyph the statusbar readout
// uses), an amber dot when the roster last found it unreachable, and a hairline
// separating it from the previous group. The first group gets no hairline.
function FleetDivider({
  connection,
  first,
  label,
  reachable
}: {
  connection: null | Pick<FleetGroup, 'connectionId' | 'kind'> | Pick<DesktopRegistryConnection, 'id' | 'kind'>
  first: boolean
  label: null | string
  reachable: boolean
}) {
  if (!connection) {
    return null
  }

  const connectionId = 'connectionId' in connection ? connection.connectionId : connection.id

  const marker = (
    <span
      aria-hidden="true"
      className={cn('flex h-5 shrink-0 items-center gap-0.5', first ? 'mr-0.5' : 'mx-0.5')}
      data-connection-id={connectionId}
      data-reachable={reachable}
      data-slot="profile-rail-divider"
    >
      {!first && <span className="h-3 w-px bg-(--ui-stroke-tertiary)" />}
      <ConnectionGlyph connection={connection} />
      {!reachable && <span className="size-1.5 rounded-full bg-amber-500" data-slot="profile-rail-unreachable" />}
    </span>
  )

  return label ? <Tip label={label}>{marker}</Tip> : marker
}

// One at-rest gateway on the fleet rail: hairline + kind glyph (amber dot when
// the roster last found it unreachable — never hidden, a sleeping box is still
// yours), then its home square and named squares, dimmed. Clicking any of
// them re-homes onto that exact (gateway, profile).
function FleetRestGroup({
  colors,
  first,
  group,
  onDelete,
  onEditSoul,
  onRecolor,
  onRename,
  onSelect,
  pendingRoute
}: {
  colors: Record<string, string>
  first: boolean
  group: FleetGroup
  onDelete: (agent: FleetAgent) => void
  onEditSoul: (agent: FleetAgent) => void
  onRecolor: (agent: FleetAgent, color: null | string) => void
  onRename: (agent: FleetAgent) => void
  onSelect: (agent: FleetAgent) => void
  pendingRoute: null | string
}) {
  const { t } = useI18n()
  const p = t.profiles
  const dividerLabel = group.reachable ? p.fleet.gateway(group.label) : p.fleet.gatewayUnreachable(group.label)
  const defaultKey = fleetRouteKey(group.connectionId, group.defaultAgent.profile)

  return (
    <>
      <FleetDivider connection={group} first={first} label={dividerLabel} reachable={group.reachable} />
      <span
        aria-label={p.fleet.gateway(group.label)}
        className="flex shrink-0 items-center gap-1"
        data-active="false"
        data-connection-id={group.connectionId}
        data-reachable={group.reachable}
        data-slot="profile-rail-gateway"
        role="group"
      >
        <ProfilePill
          active={false}
          connectionId={group.connectionId}
          glyph="home"
          label={p.fleet.onGateway(group.defaultAgent.profile, group.label)}
          muted
          onSelect={() => onSelect(group.defaultAgent)}
          pending={pendingRoute === defaultKey}
          slot="profile-rail-rest-home"
        />
        {group.named.map(agent => (
          <RestSquare
            agent={agent}
            color={resolveProfileColor(agent.profile, colors)}
            key={agent.profile}
            onDelete={() => onDelete(agent)}
            onEditSoul={() => onEditSoul(agent)}
            onRecolor={color => onRecolor(agent, color)}
            onRename={() => onRename(agent)}
            onSelect={() => onSelect(agent)}
            pending={pendingRoute === fleetRouteKey(agent.connectionId, agent.profile)}
          />
        ))}
      </span>
    </>
  )
}

// An at-rest square: the same tile as ProfileSquare, minus drag-reorder and
// hold-to-recolor (the strip it lives in is not sortable across machines).
// Tooltip and accessible name carry the gateway so two same-named profiles on
// different machines never read alike; the right-click actions run against
// the square's owning gateway.
function RestSquare({
  agent,
  color,
  onDelete,
  onEditSoul,
  onRecolor,
  onRename,
  onSelect,
  pending
}: {
  agent: FleetAgent
  color: null | string
  onDelete: () => void
  onEditSoul: () => void
  onRecolor: (color: null | string) => void
  onRename: () => void
  onSelect: () => void
  pending: boolean
}) {
  const { t } = useI18n()
  const p = t.profiles
  const hue = color ?? 'var(--ui-text-quaternary)'
  const [pickerOpen, setPickerOpen] = useState(false)
  const label = p.fleet.onGateway(agent.profile, agent.connectionLabel)

  const pickColor = (next: null | string) => {
    onRecolor(next)
    setPickerOpen(false)
    triggerHaptic('selection')
  }

  return (
    <Popover onOpenChange={setPickerOpen} open={pickerOpen}>
      <ContextMenu>
        <TooltipProvider delayDuration={0}>
          <Tooltip>
            <PopoverAnchor asChild>
              <ContextMenuTrigger asChild>
                <TooltipTrigger asChild>
                  <button
                    aria-busy={pending || undefined}
                    aria-label={label}
                    className="relative grid size-5 shrink-0 select-none place-items-center rounded-[3px] text-[0.5625rem] font-semibold uppercase leading-none opacity-35 transition-opacity hover:opacity-100 aria-busy:opacity-100"
                    data-connection-id={agent.connectionId}
                    data-profile={agent.profile}
                    data-slot="profile-rail-rest-square"
                    onClick={onSelect}
                    style={{ backgroundColor: profileColorSoft(hue, 22), color: color ?? undefined }}
                    type="button"
                  >
                    {pending ? (
                      <Loader2 aria-hidden="true" className="size-3 animate-spin" />
                    ) : (
                      agent.profile.replace(/[^a-z0-9]/gi, '').charAt(0) || '?'
                    )}
                  </button>
                </TooltipTrigger>
              </ContextMenuTrigger>
            </PopoverAnchor>
            <TooltipContent>{label}</TooltipContent>
          </Tooltip>
        </TooltipProvider>

        <ContextMenuContent
          aria-label={p.actions}
          className="w-44"
          collisionPadding={{ bottom: 44, left: 8, right: 8, top: 8 }}
          onCloseAutoFocus={event => event.preventDefault()}
        >
          <ContextMenuItem onSelect={onSelect}>
            <Codicon name="arrow-right" size="0.875rem" />
            <span className="truncate">{p.fleet.switchTo(agent.profile, agent.connectionLabel)}</span>
          </ContextMenuItem>
          <ContextMenuItem onSelect={() => setPickerOpen(true)}>
            <Codicon name="symbol-color" size="0.875rem" />
            <span>{p.color}</span>
          </ContextMenuItem>
          <ContextMenuItem onSelect={onRename}>
            <Codicon name="text-size" size="0.875rem" />
            <span>{p.renameMenu}</span>
          </ContextMenuItem>
          <ContextMenuItem onSelect={onEditSoul}>
            <Codicon name="edit" size="0.875rem" />
            <span>{p.editSoul}</span>
          </ContextMenuItem>
          <ContextMenuItem
            className="text-destructive focus:text-destructive"
            onSelect={onDelete}
            variant="destructive"
          >
            <Codicon name="trash" size="0.875rem" />
            <span>{t.common.delete}</span>
          </ContextMenuItem>
        </ContextMenuContent>
      </ContextMenu>

      <PopoverContent
        aria-label={p.colorFor}
        className="w-auto p-2"
        collisionPadding={{ bottom: 44, left: 8, right: 8, top: 8 }}
        side="top"
      >
        <ColorSwatches
          clearIcon="sync"
          clearLabel={p.autoColor}
          onChange={pickColor}
          swatches={PROFILE_SWATCHES}
          swatchLabel={p.setColor}
          value={color}
        />
      </PopoverContent>
    </Popover>
  )
}

interface ProfileSquareProps {
  active: boolean
  color: null | string
  label: string
  onSelect: () => void
  onRecolor: (color: null | string) => void
  onRename: () => void
  onEditSoul: () => void
  // Absent on multi-gateway setups: the legacy per-profile remote override
  // is superseded by the fleet rail there.
  onConnectRemote?: () => void
  onDelete: () => void
  // hostname[:port] of this profile's remote override, or null when the
  // profile runs locally. Drives the "remote" badge on the square.
  remoteHost: null | string
}

// Hold this long without moving (a drag would have started first) to open the
// color picker — the "hard press" gesture, distinct from tap-to-select.
const LONG_PRESS_MS = 450

// A profile *is* its colored square — no icon-button chrome. Soft profile-tint
// fill + the initial in the full color; the active one pops to full opacity with
// a color ring. These pack tightly so the rail reads as a strip of profiles,
// drag-sort to reorder (a tap below the drag threshold still selects), and
// right-click to rename/delete. The button carries both the tooltip and
// context-menu triggers via nested asChild Slots, so a single element keeps the
// dnd listeners, hover tip, and right-click menu.
function ProfileSquare({
  active,
  color,
  label,
  onConnectRemote,
  onDelete,
  onEditSoul,
  onRecolor,
  onRename,
  onSelect,
  remoteHost
}: ProfileSquareProps) {
  const { t } = useI18n()
  const p = t.profiles
  const hue = color ?? 'var(--ui-text-quaternary)'
  const [pickerOpen, setPickerOpen] = useState(false)
  const pressTimer = useRef<null | number>(null)
  const suppressClick = useRef(false)
  // Hovering a square telegraphs the switch — start that profile's backend
  // spawn now so a cold click doesn't pay the full boot.
  const { cancelPrewarm, startPrewarm } = useProfilePrewarm(label)

  const { attributes, isDragging, listeners, setNodeRef, transform, transition } = useSortable({
    id: label,
    transition: RAIL_TRANSITION
  })

  const clearPress = () => {
    if (pressTimer.current != null) {
      clearTimeout(pressTimer.current)
      pressTimer.current = null
    }
  }

  // A real drag (movement past the dnd threshold) cancels the pending hold, so a
  // reorder never doubles as a color pick. Also tidy up on unmount.
  useEffect(() => {
    if (isDragging) {
      clearPress()
    }
  }, [isDragging])
  useEffect(() => clearPress, [])

  const base = CSS.Transform.toString(transform)
  const ring = active ? `inset 0 0 0 1.5px ${hue}` : ''
  const lift = isDragging ? '0 6px 16px -4px rgb(0 0 0 / 0.4)' : ''

  const pickColor = (next: null | string) => {
    onRecolor(next)
    setPickerOpen(false)
    triggerHaptic('selection')
  }

  return (
    <Popover onOpenChange={setPickerOpen} open={pickerOpen}>
      <ContextMenu>
        <TooltipProvider delayDuration={0}>
          <Tooltip>
            <PopoverAnchor asChild>
              <ContextMenuTrigger asChild>
                <TooltipTrigger asChild>
                  <button
                    className={cn(
                      'relative grid size-5 shrink-0 cursor-grab touch-none select-none place-items-center rounded-[3px] text-[0.5625rem] font-semibold uppercase leading-none transition-opacity hover:opacity-100',
                      active ? 'opacity-100' : 'opacity-55',
                      isDragging && 'z-10 cursor-grabbing opacity-100'
                    )}
                    ref={setNodeRef}
                    style={{
                      backgroundColor: profileColorSoft(hue, active ? 30 : 22),
                      boxShadow: [ring, lift].filter(Boolean).join(', ') || undefined,
                      color: color ?? undefined,
                      // Glide the dragged square between snapped cells with a little
                      // overshoot (no scale — the overflow-x strip would clip it).
                      transform: base,
                      transition: isDragging ? DRAG_TRANSITION : transition
                    }}
                    type="button"
                    {...attributes}
                    {...listeners}
                    aria-label={remoteHost ? `${label} — ${p.remoteOverride.badge(remoteHost)}` : label}
                    aria-pressed={active}
                    // Hold-to-recolor rides alongside the dnd pointer listener (call
                    // it first so drag tracking still arms), then a timer opens the
                    // picker and flags the trailing click so it doesn't also select.
                    onClick={() => {
                      if (suppressClick.current) {
                        suppressClick.current = false

                        return
                      }

                      onSelect()
                    }}
                    onPointerCancel={clearPress}
                    onPointerDown={event => {
                      listeners?.onPointerDown?.(event)

                      if (event.button !== 0) {
                        return
                      }

                      suppressClick.current = false
                      clearPress()
                      pressTimer.current = window.setTimeout(() => {
                        suppressClick.current = true
                        triggerHaptic('success')
                        setPickerOpen(true)
                      }, LONG_PRESS_MS)
                    }}
                    onPointerEnter={startPrewarm}
                    onPointerLeave={() => {
                      clearPress()
                      cancelPrewarm()
                    }}
                    onPointerUp={clearPress}
                  >
                    {label.replace(/[^a-z0-9]/gi, '').charAt(0) || '?'}
                    {/* The "remote" badge: a tiny globe pinned to the corner of an
                        overridden profile's square, so which profiles leave this
                        machine is visible at a glance (#91349). */}
                    {remoteHost && (
                      <span
                        aria-hidden="true"
                        className="absolute -right-0.5 -top-0.5 grid size-2 place-items-center rounded-full bg-(--ui-panel-background)"
                        data-slot="profile-remote-badge"
                      >
                        <Codicon name="globe" size="0.5rem" />
                      </span>
                    )}
                  </button>
                </TooltipTrigger>
              </ContextMenuTrigger>
            </PopoverAnchor>
            <TooltipContent>{remoteHost ? `${label} · ${p.remoteOverride.badge(remoteHost)}` : label}</TooltipContent>
          </Tooltip>
        </TooltipProvider>

        {/* The rail sits at the very bottom, so pad off the chrome (esp. the
            statusbar) — Radix then flips the menu up instead of squishing it. */}
        <ContextMenuContent
          aria-label={p.actions}
          className="w-40"
          collisionPadding={{ bottom: 44, left: 8, right: 8, top: 8 }}
          // Menu close refocuses the trigger — which doubles as the popover
          // anchor — so the picker reads it as focus-outside and dies on open.
          // Suppress the refocus and the picker survives.
          onCloseAutoFocus={event => event.preventDefault()}
        >
          <ContextMenuItem onSelect={() => setPickerOpen(true)}>
            <Codicon name="symbol-color" size="0.875rem" />
            <span>{p.color}</span>
          </ContextMenuItem>
          <ContextMenuItem onSelect={onRename}>
            <Codicon name="text-size" size="0.875rem" />
            <span>{p.renameMenu}</span>
          </ContextMenuItem>
          <ContextMenuItem onSelect={onEditSoul}>
            <Codicon name="edit" size="0.875rem" />
            <span>{p.editSoul}</span>
          </ContextMenuItem>
          <ContextMenuItem onSelect={() => void runExportProfileFlow(label)}>
            <Codicon name="package" size="0.875rem" />
            <span>{p.exportMenu}</span>
          </ContextMenuItem>
          {onConnectRemote && (
            <ContextMenuItem onSelect={onConnectRemote}>
              <Codicon name="globe" size="0.875rem" />
              <span>{remoteHost ? p.remoteOverride.badge(remoteHost) : p.remoteOverride.menuItem}</span>
            </ContextMenuItem>
          )}
          <ContextMenuItem
            className="text-destructive focus:text-destructive"
            onSelect={onDelete}
            variant="destructive"
          >
            <Codicon name="trash" size="0.875rem" />
            <span>{t.common.delete}</span>
          </ContextMenuItem>
        </ContextMenuContent>
      </ContextMenu>

      <PopoverContent
        aria-label={p.colorFor}
        className="w-auto p-2"
        collisionPadding={{ bottom: 44, left: 8, right: 8, top: 8 }}
        side="top"
      >
        <ColorSwatches
          clearIcon="sync"
          clearLabel={p.autoColor}
          onChange={pickColor}
          swatches={PROFILE_SWATCHES}
          swatchLabel={p.setColor}
          value={color}
        />
      </PopoverContent>
    </Popover>
  )
}
