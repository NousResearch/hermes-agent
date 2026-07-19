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
import { type CSSProperties, useEffect, useId, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { CodeEditor } from '@/components/chat/code-editor'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ColorSwatches } from '@/components/ui/color-swatches'
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from '@/components/ui/context-menu'
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Popover, PopoverAnchor, PopoverContent } from '@/components/ui/popover'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tip, Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { getProfileSoul, updateProfileSoul } from '@/hermes'
import { type Translations, useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { PROFILE_SWATCHES, profileColorSoft, resolveProfileColor } from '@/lib/profile-color'
import {
  REORDER_DRAG_TRANSITION_CSS,
  REORDER_RAIL_TRANSITION,
  reorderCommitHaptic,
  reorderStepHaptic
} from '@/lib/reorder'
import { cn } from '@/lib/utils'
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
  refreshActiveProfile,
  selectProfile,
  setProfileColor,
  setProfileOrder,
  setShowAllProfiles,
  sortByProfileOrder
} from '@/store/profile'
import {
  $attentionSessionIds,
  $sessions,
  $unreadFinishedSessionIds,
  getAttentionSessionScopeKeys,
  getUnreadSessionScopeKeys
} from '@/store/session'
import { $sessionActivityKeys } from '@/store/session-activity'
import type { ProfileInfo } from '@/types/hermes'

import { CreateProfileDialog } from '../../profiles/create-profile-dialog'
import { DeleteProfileDialog } from '../../profiles/delete-profile-dialog'
import { RenameProfileDialog } from '../../profiles/rename-profile-dialog'
import { PROFILES_ROUTE } from '../../routes'

import { deriveProfileActivityByProfile, type ProfileActivity, profileActivityPriority } from './profile-activity'
import { useProfilePrewarm } from './use-profile-prewarm'

const RAIL_GAP = 4 // px — matches gap-1 between squares.

// Past this many profiles the strip of colored squares stops scaling (tiny
// drag targets, endless horizontal scroll), so the rail collapses to a compact
// select. Drag-reorder and long-press-recolor live only on the squares path.
const PROFILE_DROPDOWN_THRESHOLD = 13

// Neighbors reflow on RAIL_TRANSITION; the dragged square glides between
// snapped cells on the snappier DRAG_TRANSITION. Both come from the SHARED
// reorder primitive (lib/reorder.ts) so every reorder strip feels identical.
const RAIL_TRANSITION = REORDER_RAIL_TRANSITION
const DRAG_TRANSITION = REORDER_DRAG_TRANSITION_CSS

function profileActivityText(p: Translations['profiles'], activity: ProfileActivity): null | string {
  if (activity === 'needs-input') {
    return p.activityNeedsInput
  }

  if (activity === 'working') {
    return p.activityRunning
  }

  return activity === 'unread' ? p.activityUnread : null
}

function profileActivityLabel(
  p: Translations['profiles'],
  profileName: string,
  activity: ProfileActivity
): null | string {
  const activityText = profileActivityText(p, activity)

  return activityText ? p.profileActivity(profileName, activityText) : null
}

function ProfileActivityBorder({ activity, hue }: { activity: ProfileActivity; hue: string }) {
  if (activity === 'idle') {
    return null
  }

  const signalHue = activity === 'needs-input' ? 'var(--color-amber-500)' : hue
  const duration = activity === 'needs-input' ? '1.7s' : activity === 'unread' ? '3.27s' : '2.23s'

  const style = {
    '--arc-c0': `color-mix(in srgb, ${signalHue} 0%, transparent)`,
    '--arc-c1': signalHue,
    '--arc-c2': `color-mix(in srgb, ${signalHue} 55%, var(--ui-text-primary))`,
    '--arc-duration': duration,
    '--arc-inset': '0px',
    '--arc-width': '0.09375rem'
  } as CSSProperties

  return (
    <>
      <span
        aria-hidden="true"
        className={cn('arc-border', activity === 'unread' && 'arc-reverse')}
        data-profile-activity-border={activity}
        style={style}
      />
      {activity !== 'working' && (
        <span
          aria-hidden="true"
          className={cn(
            'pointer-events-none absolute right-0.5 top-0.5 z-2 size-1.5',
            activity === 'needs-input' ? 'quest-glow rounded-full' : 'rotate-45 rounded-[1px]'
          )}
          data-profile-activity-pip={activity}
          style={{ backgroundColor: signalHue }}
        />
      )}
      {activity === 'working' && (
        <span
          aria-hidden="true"
          className="pointer-events-none absolute right-0.5 top-0.5 z-2 hidden size-1.5 rounded-full border bg-transparent motion-reduce:block"
          data-profile-activity-static="working"
          style={{ borderColor: signalHue }}
        />
      )}
    </>
  )
}

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
// The active profile pops in its own color — the "where am I" cue. Single-
// profile users see the "+" (create their first profile) and the Manage
// overflow (edit the default profile's SOUL.md); the colored named squares
// and the default↔all toggle only appear once a second profile exists.
export function ProfileRail() {
  const { t } = useI18n()
  const p = t.profiles
  const profiles = useStore($profiles)
  const scope = useStore($profileScope)
  const gatewayProfile = useStore($activeGatewayProfile)
  const order = useStore($profileOrder)
  const colors = useStore($profileColors)
  const sessions = useStore($sessions)
  const workingSessionIds = useStore($sessionActivityKeys)
  useStore($attentionSessionIds)
  useStore($unreadFinishedSessionIds)

  const activityByProfile = deriveProfileActivityByProfile({
    attentionSessionIds: getAttentionSessionScopeKeys(),
    sessions,
    unreadSessionIds: getUnreadSessionScopeKeys(),
    workingSessionIds
  })

  const activityForProfile = (name: null | string | undefined): ProfileActivity =>
    activityByProfile[normalizeProfileKey(name)] ?? 'idle'

  const navigate = useNavigate()

  const [createOpen, setCreateOpen] = useState(false)
  const [pendingRename, setPendingRename] = useState<null | ProfileInfo>(null)
  const [pendingDelete, setPendingDelete] = useState<null | ProfileInfo>(null)
  const [pendingSoul, setPendingSoul] = useState<null | string>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  // Too many profiles for the square strip → collapse to the select. Declared
  // ahead of the wheel effect, which re-binds when the strip mounts/unmounts.
  const condensed = profiles.length > PROFILE_DROPDOWN_THRESHOLD

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

  // Re-pull the running profile + list on mount so a profile created elsewhere
  // shows up; cheap and best-effort.
  useEffect(() => {
    void refreshActiveProfile()
  }, [])

  // Open the create dialog when the `profile.create` hotkey fires (the dialog
  // state lives here, so the global keybind bumps a request atom we watch).
  const createRequest = useStore($profileCreateRequest)
  const lastCreateRef = useRef(createRequest)

  useEffect(() => {
    if (createRequest === lastCreateRef.current) {
      return
    }

    lastCreateRef.current = createRequest
    setCreateOpen(true)
  }, [createRequest])

  return (
    <div aria-label={p.title} className="flex items-center gap-0.5" data-slot="profile-rail" role="group">
      {/* One button toggles default ↔ all: home face when scoped to a profile,
          layers face when showing everything. Pinned left like Manage is right.
          Hidden until a second profile exists. */}
      {multiProfile &&
        (defaultProfile ? (
          // On default → toggle to all. Anywhere else (all view or a named
          // profile) → return to default. So leaving a profile never lands on all.
          <ProfilePill
            active={isAll || onDefault}
            activity={activityForProfile(defaultProfile.name)}
            activityProfile={defaultProfile.name}
            glyph={isAll ? 'layers' : 'home'}
            label={onDefault ? p.showAllProfiles : p.switchToProfile(defaultProfile.name)}
            onSelect={() => (onDefault ? setShowAllProfiles(true) : selectProfile(defaultProfile.name))}
          />
        ) : (
          <ProfilePill active={isAll} glyph="layers" label={p.allProfiles} onSelect={() => setShowAllProfiles(true)} />
        ))}

      {/* Single-profile: the active default's home icon next to the create +. */}
      {!multiProfile && defaultProfile && (
        <ProfilePill
          active
          activity={activityForProfile(defaultProfile.name)}
          activityProfile={defaultProfile.name}
          glyph="home"
          label={defaultProfile.name}
          onSelect={() => selectProfile(defaultProfile.name)}
        />
      )}

      {condensed ? (
        // Condensed path: one compact dropdown instead of N squares. No drag
        // reorder, no long-press recolor, no per-square context menu — Manage
        // covers rename/delete at this scale.
        <div className="flex min-w-0 flex-1 items-center gap-1">
          <ProfileDropdown
            activeKey={isAll ? null : activeKey}
            activityByProfile={activityByProfile}
            colors={colors}
            onSelect={selectProfile}
            profiles={named}
          />
          <AddProfileButton label={p.newProfile} onClick={() => setCreateOpen(true)} />
        </div>
      ) : (
        <div
          className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
          ref={scrollRef}
        >
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
                      activity={activityForProfile(profile.name)}
                      color={resolveProfileColor(profile.name, colors)}
                      key={profile.name}
                      label={profile.name}
                      onDelete={() => setPendingDelete(profile)}
                      onEditSoul={() => setPendingSoul(profile.name)}
                      onRecolor={color => setProfileColor(profile.name, color)}
                      onRename={() => setPendingRename(profile)}
                      onSelect={() => selectProfile(profile.name)}
                    />
                  ))}
                </div>
              </SortableContext>
            </DndContext>
          )}

          <AddProfileButton label={p.newProfile} onClick={() => setCreateOpen(true)} />
        </div>
      )}

      {/* Always reachable, even with only the default profile: the manage
          overlay is the only place to edit a profile's SOUL.md, and a
          single-profile user must be able to edit the default's persona
          without first creating a throwaway second profile. */}
      <ProfilePill active={false} glyph="ellipsis" label={p.manageProfiles} onSelect={() => navigate(PROFILES_ROUTE)} />

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
    </div>
  )
}

// Right-click → Edit SOUL.md for a sidebar profile — the same in-app markdown
// editor as the memory-graph node edit, so a profile's persona is editable
// without opening the Manage overlay.
function EditSoulDialog({ onClose, profileName }: { onClose: () => void; profileName: null | string }) {
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

    getProfileSoul(profileName)
      .then(soul => !cancelled && setContent(soul.content))
      .catch(err => !cancelled && notifyError(err, p.failedLoadSoul))
      .finally(() => !cancelled && setLoading(false))

    return () => void (cancelled = true)
  }, [p, profileName])

  const save = async () => {
    if (!profileName) {
      return
    }

    setSaving(true)

    try {
      await updateProfileSoul(profileName, content)
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
          <DialogTitle>{profileName} · SOUL.md</DialogTitle>
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

// The condensed rail: every named profile in one compact select. The trigger
// shows the active profile (tinted initial + name); on default/all scope it
// falls back to the placeholder since the left toggle pill carries that state.
function ProfileDropdown({
  activeKey,
  activityByProfile,
  colors,
  onSelect,
  profiles
}: {
  activeKey: null | string
  activityByProfile: Record<string, ProfileActivity>
  colors: Record<string, string>
  onSelect: (name: string) => void
  profiles: ProfileInfo[]
}) {
  const { t } = useI18n()
  const p = t.profiles
  const activityDescriptionId = useId()

  const activeProfile = activeKey
    ? profiles.find(profile => normalizeProfileKey(profile.name) === activeKey)
    : undefined

  const value = activeProfile?.name ?? ''
  const activeActivity = activeProfile ? (activityByProfile[normalizeProfileKey(activeProfile.name)] ?? 'idle') : 'idle'

  const strongestSignal = profiles.reduce<{ activity: ProfileActivity; profile: ProfileInfo } | null>(
    (best, profile) => {
      const activity = activityByProfile[normalizeProfileKey(profile.name)] ?? 'idle'

      if (activity === 'idle') {
        return best
      }

      if (!best || profileActivityPriority(activity) > profileActivityPriority(best.activity)) {
        return { activity, profile }
      }

      return best
    },
    activeProfile && activeActivity !== 'idle' ? { activity: activeActivity, profile: activeProfile } : null
  )

  const triggerActivity = strongestSignal?.activity ?? 'idle'
  const triggerSignalProfile = strongestSignal?.profile

  const triggerHue = triggerSignalProfile
    ? (resolveProfileColor(triggerSignalProfile.name, colors) ?? 'var(--ui-accent)')
    : 'var(--ui-accent)'

  const activityDescription = triggerSignalProfile
    ? (profileActivityLabel(p, triggerSignalProfile.name, triggerActivity) ?? triggerSignalProfile.name)
    : null

  return (
    <Select onValueChange={name => name && onSelect(name)} value={value}>
      <SelectTrigger
        aria-describedby={activityDescription ? activityDescriptionId : undefined}
        aria-label={p.title}
        className="relative min-w-0 flex-1 overflow-visible"
        data-profile-activity={triggerActivity === 'idle' ? undefined : triggerActivity}
        size="xs"
      >
        <SelectValue placeholder={p.title} />
        <ProfileActivityBorder activity={triggerActivity} hue={triggerHue} />
      </SelectTrigger>
      {activityDescription && (
        <span className="sr-only" id={activityDescriptionId}>
          {activityDescription}
        </span>
      )}
      <SelectContent collisionPadding={{ bottom: 44, left: 8, right: 8, top: 8 }} side="top">
        {profiles.map(profile => {
          const color = resolveProfileColor(profile.name, colors)
          const activity = activityByProfile[normalizeProfileKey(profile.name)] ?? 'idle'
          const activityLabel = profileActivityLabel(p, profile.name, activity)

          return (
            <ProfileDropdownItem
              activity={activity}
              activityLabel={activityLabel}
              color={color}
              key={profile.name}
              name={profile.name}
            />
          )
        })}
      </SelectContent>
    </Select>
  )
}

// One dropdown row per profile — its own component so each row can own a
// hover-intent prewarm timer (see useProfilePrewarm).
function ProfileDropdownItem({
  activity,
  activityLabel,
  color,
  name
}: {
  activity: ProfileActivity
  activityLabel: null | string
  color: null | string
  name: string
}) {
  const hue = color ?? 'var(--ui-text-quaternary)'
  const hasActivity = activity !== 'idle'
  const { cancelPrewarm, startPrewarm } = useProfilePrewarm(name)

  return (
    <SelectItem
      aria-label={activityLabel ?? name}
      onPointerEnter={startPrewarm}
      onPointerLeave={cancelPrewarm}
      value={name}
    >
      <span className="flex min-w-0 items-center gap-1.5">
        <span
          aria-hidden="true"
          className="relative grid size-4 shrink-0 place-items-center overflow-visible rounded-[3px] text-[0.5rem] font-semibold uppercase leading-none"
          data-profile-activity={hasActivity ? activity : undefined}
          style={{ backgroundColor: profileColorSoft(hue, hasActivity ? 34 : 22), color: color ?? undefined }}
        >
          {name.replace(/[^a-z0-9]/gi, '').charAt(0) || '?'}
          <ProfileActivityBorder activity={activity} hue={hue} />
        </span>
        <span className={cn('truncate', hasActivity && 'font-medium text-foreground')}>{name}</span>
      </span>
    </SelectItem>
  )
}

interface ProfilePillProps {
  active: boolean
  activity?: ProfileActivity
  activityProfile?: string
  // home / All / Manage are glyph action buttons (navigation, not identity).
  glyph: string
  label: string
  onSelect: () => void
}

function ProfilePill({ active, activity = 'idle', activityProfile, glyph, label, onSelect }: ProfilePillProps) {
  const { t } = useI18n()
  const activityText = activityProfile ? profileActivityText(t.profiles, activity) : null
  const accessibleLabel = activityText ? `${label} · ${activityText}` : label
  const hasActivity = activity !== 'idle'

  return (
    <Tip label={accessibleLabel}>
      <Button
        aria-label={accessibleLabel}
        aria-pressed={active}
        className={cn(
          'relative overflow-visible bg-transparent text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground',
          (active || hasActivity) && 'bg-(--ui-control-active-background) text-foreground'
        )}
        data-profile-activity={hasActivity ? activity : undefined}
        onClick={onSelect}
        size="icon-xs"
        type="button"
        variant="ghost"
      >
        <Codicon name={glyph} size="0.875rem" />
        <ProfileActivityBorder activity={activity} hue="var(--ui-accent)" />
      </Button>
    </Tip>
  )
}

interface ProfileSquareProps {
  active: boolean
  activity: ProfileActivity
  color: null | string
  label: string
  onSelect: () => void
  onRecolor: (color: null | string) => void
  onRename: () => void
  onEditSoul: () => void
  onDelete: () => void
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
  activity,
  color,
  label,
  onDelete,
  onEditSoul,
  onRecolor,
  onRename,
  onSelect
}: ProfileSquareProps) {
  const { t } = useI18n()
  const p = t.profiles
  const hue = color ?? 'var(--ui-text-quaternary)'
  const hasActivity = activity !== 'idle'
  const activityLabel = profileActivityLabel(p, label, activity)
  const accessibleLabel = activityLabel ?? label
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
                      'relative grid size-5 shrink-0 cursor-grab touch-none select-none place-items-center overflow-visible rounded-[3px] text-[0.5625rem] font-semibold uppercase leading-none transition-opacity hover:opacity-100',
                      active || hasActivity ? 'opacity-100' : 'opacity-55',
                      isDragging && 'z-10 cursor-grabbing opacity-100'
                    )}
                    data-profile-activity={hasActivity ? activity : undefined}
                    ref={setNodeRef}
                    style={{
                      backgroundColor: profileColorSoft(hue, hasActivity ? 34 : active ? 30 : 22),
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
                    aria-label={accessibleLabel}
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
                    <ProfileActivityBorder activity={activity} hue={hue} />
                  </button>
                </TooltipTrigger>
              </ContextMenuTrigger>
            </PopoverAnchor>
            <TooltipContent>{accessibleLabel}</TooltipContent>
          </Tooltip>
        </TooltipProvider>

        {/* The rail sits at the very bottom, so pad off the chrome (esp. the
            statusbar) — Radix then flips the menu up instead of squishing it. */}
        <ContextMenuContent
          aria-label={p.actionsFor(label)}
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
        aria-label={p.colorFor(label)}
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
