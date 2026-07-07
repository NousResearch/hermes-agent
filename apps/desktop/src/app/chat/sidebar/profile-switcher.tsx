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
import { useEffect, useRef, useState } from 'react'
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
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { PROFILE_SWATCHES, profileColorSoft, resolveProfileColor } from '@/lib/profile-color'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import {
  $activeGatewayProfile,
  $profileActivity,
  $profileColors,
  $profileCreateRequest,
  $profileOrder,
  $profiles,
  $profileScope,
  ALL_PROFILES,
  neediestSessionId,
  normalizeProfileKey,
  type ProfileActivity,
  refreshActiveProfile,
  selectProfile,
  selectProfileSession,
  setProfileColor,
  setProfileOrder,
  setShowAllProfiles,
  sortByProfileOrder
} from '@/store/profile'
import type { ProfileInfo } from '@/types/hermes'

import { CreateProfileDialog } from '../../profiles/create-profile-dialog'
import { DeleteProfileDialog } from '../../profiles/delete-profile-dialog'
import { RenameProfileDialog } from '../../profiles/rename-profile-dialog'
import { PROFILES_ROUTE } from '../../routes'

const RAIL_GAP = 4 // px — matches gap-1 between squares.

// Past this many profiles the strip of colored squares stops scaling (tiny
// drag targets, endless horizontal scroll), so the rail collapses to a compact
// select. Drag-reorder and long-press-recolor live only on the squares path.
const PROFILE_DROPDOWN_THRESHOLD = 13

// easeOutBack — a little overshoot so squares spring into their new slot rather
// than sliding in flat. Neighbors reflow on RAIL_TRANSITION; the dragged square
// glides between snapped cells on the snappier DRAG_TRANSITION.
const SPRING = 'cubic-bezier(0.34, 1.56, 0.64, 1)'
const RAIL_TRANSITION = { duration: 300, easing: SPRING }
const DRAG_TRANSITION = `transform 200ms ${SPRING}`

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
  const activityByProfile = useStore($profileActivity)
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

  // Default rows are keyed "default" by the aggregator/derivation whatever the
  // profile's directory name; named rows key by their (normalized) name.
  const defaultActivity = defaultProfile
    ? (activityByProfile[normalizeProfileKey(defaultProfile.name)] ?? activityByProfile.default)
    : undefined

  // Badge click: jump into the profile AND straight to the session that needs
  // you (attention first, else the most recent working one) — composing with
  // the per-profile last-session restore for plain square clicks.
  const openNeediest = (name: string, activity: ProfileActivity | undefined) => {
    const target = neediestSessionId(activity)

    if (target) {
      selectProfileSession(name, target)
    } else {
      selectProfile(name)
    }
  }

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
      triggerHaptic('selection')
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
      triggerHaptic('success')
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
    <div aria-label="Profiles" className="flex items-center gap-0.5" data-slot="profile-rail" role="tablist">
      {/* One button toggles default ↔ all: home face when scoped to a profile,
          layers face when showing everything. Pinned left like Manage is right.
          Hidden until a second profile exists. */}
      {multiProfile &&
        (defaultProfile ? (
          // On default → toggle to all. Anywhere else (all view or a named
          // profile) → return to default. So leaving a profile never lands on all.
          // While away from default, its activity badge rides the pill's corner.
          <span className="relative shrink-0">
            <ProfilePill
              active={isAll || onDefault}
              glyph={isAll ? 'layers' : 'home'}
              label={onDefault ? p.showAllProfiles : p.switchToProfile(defaultProfile.name)}
              onSelect={() => (onDefault ? setShowAllProfiles(true) : selectProfile(defaultProfile.name))}
            />
            {!onDefault && defaultActivity && (
              <ActivityBadge
                activity={defaultActivity}
                label={defaultProfile.name}
                onOpen={() => openNeediest(defaultProfile.name, defaultActivity)}
              />
            )}
          </span>
        ) : (
          <ProfilePill active={isAll} glyph="layers" label={p.allProfiles} onSelect={() => setShowAllProfiles(true)} />
        ))}

      {/* Single-profile: the active default's home icon next to the create +. */}
      {!multiProfile && defaultProfile && (
        <ProfilePill
          active
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
                      activity={activityByProfile[normalizeProfileKey(profile.name)]}
                      color={resolveProfileColor(profile.name, colors)}
                      key={profile.name}
                      label={profile.name}
                      onDelete={() => setPendingDelete(profile)}
                      onEditSoul={() => setPendingSoul(profile.name)}
                      onOpenActivity={() =>
                        openNeediest(profile.name, activityByProfile[normalizeProfileKey(profile.name)])
                      }
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

  const value = activeKey ? (profiles.find(profile => normalizeProfileKey(profile.name) === activeKey)?.name ?? '') : ''

  return (
    <Select onValueChange={name => name && onSelect(name)} value={value}>
      <SelectTrigger aria-label={p.title} className="min-w-0 flex-1" size="xs">
        <SelectValue placeholder={p.title} />
      </SelectTrigger>
      <SelectContent collisionPadding={{ bottom: 44, left: 8, right: 8, top: 8 }} side="top">
        {profiles.map(profile => {
          const color = resolveProfileColor(profile.name, colors)
          const hue = color ?? 'var(--ui-text-quaternary)'
          const activity = activityByProfile[normalizeProfileKey(profile.name)]
          const attention = activity ? activity.attention.length : 0
          const working = activity ? activity.working.length : 0

          return (
            <SelectItem key={profile.name} value={profile.name}>
              <span className="flex min-w-0 items-center gap-1.5">
                <span
                  aria-hidden="true"
                  className="grid size-4 shrink-0 place-items-center rounded-[3px] text-[0.5rem] font-semibold uppercase leading-none"
                  style={{ backgroundColor: profileColorSoft(hue, 22), color: color ?? undefined }}
                >
                  {profile.name.replace(/[^a-z0-9]/gi, '').charAt(0) || '?'}
                </span>
                <span className="truncate">{profile.name}</span>
                {/* Condensed rail: a plain dot stands in for the corner badge
                    (amber = needs input beats accent = working). */}
                {(attention > 0 || working > 0) && (
                  <span
                    aria-hidden="true"
                    className={cn(
                      'ml-auto size-1.5 shrink-0 rounded-full',
                      attention > 0 ? 'bg-amber-500' : 'bg-(--ui-accent)'
                    )}
                  />
                )}
              </span>
            </SelectItem>
          )
        })}
      </SelectContent>
    </Select>
  )
}

// How many activity rows a rail tooltip lists before truncating — the badge
// count still carries the full number.
const ACTIVITY_TOOLTIP_LIMIT = 6

// Working/attention badge pinned to a rail square's (or the default pill's)
// corner. Amber + count = sessions blocked on you; accent ping = a turn is
// running somewhere in that profile ("needs input" wins, like the sidebar row
// dot). A separate sibling button — a <button> can't nest one — that stops
// pointer-down so a badge click never arms a drag or selects the square under
// it; clicking jumps straight to the neediest session via onOpen.
function ActivityBadge({ activity, label, onOpen }: { activity: ProfileActivity; label: string; onOpen: () => void }) {
  const { t } = useI18n()
  const p = t.profiles
  const attention = activity.attention.length
  const working = activity.working.length

  if (attention === 0 && working === 0) {
    return null
  }

  const description = attention > 0 ? p.attentionBadge(attention) : p.workingBadge(working)

  return (
    <button
      aria-label={`${label}: ${description}`}
      className={cn(
        'absolute -right-1 -top-1 z-10 grid place-items-center rounded-full leading-none',
        attention > 0
          ? 'h-3 min-w-3 bg-amber-500 px-[0.1875rem] text-[0.5rem] font-semibold text-black'
          : "size-2 bg-(--ui-accent) before:absolute before:inset-0 before:animate-ping before:rounded-full before:bg-(--ui-accent) before:opacity-70 before:content-['']"
      )}
      onClick={event => {
        event.stopPropagation()
        onOpen()
      }}
      onPointerDown={event => event.stopPropagation()}
      title={description}
      type="button"
    >
      {attention > 0 ? (attention > 9 ? '9+' : attention) : null}
    </button>
  )
}

// Square tooltip body: the plain profile name, or — when the profile has live
// activity — the name plus each active session's title and state.
function ActivityTooltip({ activity, label }: { activity: ProfileActivity | undefined; label: string }) {
  const { t } = useI18n()
  const r = t.sidebar.row
  const rows = activity ? [...activity.attention, ...activity.working].slice(0, ACTIVITY_TOOLTIP_LIMIT) : []

  if (rows.length === 0) {
    return <>{label}</>
  }

  return (
    <div className="flex max-w-64 flex-col gap-1">
      <span className="font-semibold">{label}</span>
      {rows.map(row => (
        <span className="flex items-center gap-1.5" key={row.id}>
          <span
            aria-hidden="true"
            className={cn('size-1.5 shrink-0 rounded-full', row.needsInput ? 'bg-amber-500' : 'bg-(--ui-accent)')}
          />
          <span className="min-w-0 flex-1 truncate">{row.title?.trim() || r.untitledPlaceholder}</span>
          <span className="shrink-0 text-(--ui-text-tertiary)">{row.needsInput ? r.needsInput : r.sessionRunning}</span>
        </span>
      ))}
    </div>
  )
}

interface ProfilePillProps {
  active: boolean
  // home / All / Manage are glyph action buttons (navigation, not identity).
  glyph: string
  label: string
  onSelect: () => void
}

function ProfilePill({ active, glyph, label, onSelect }: ProfilePillProps) {
  return (
    <Tip label={label}>
      <Button
        aria-label={label}
        aria-pressed={active}
        className={cn(
          'bg-transparent text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground',
          active && 'bg-(--ui-control-active-background) text-foreground'
        )}
        onClick={onSelect}
        size="icon-xs"
        type="button"
        variant="ghost"
      >
        <Codicon name={glyph} size="0.875rem" />
      </Button>
    </Tip>
  )
}

interface ProfileSquareProps {
  active: boolean
  activity?: ProfileActivity
  color: null | string
  label: string
  onSelect: () => void
  onOpenActivity: () => void
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
  onOpenActivity,
  onRecolor,
  onRename,
  onSelect
}: ProfileSquareProps) {
  const { t } = useI18n()
  const p = t.profiles
  const hue = color ?? 'var(--ui-text-quaternary)'
  const [pickerOpen, setPickerOpen] = useState(false)
  const pressTimer = useRef<null | number>(null)
  const suppressClick = useRef(false)

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
    // The sortable node is this wrapper (not the button) so the corner
    // activity badge — a sibling button; buttons can't nest — rides along with
    // drags. The drag listeners stay on the square itself.
    <div
      className={cn('relative shrink-0', isDragging && 'z-10')}
      ref={setNodeRef}
      style={{
        // Glide the dragged square between snapped cells with a little
        // overshoot (no scale — the overflow-x strip would clip it).
        transform: base,
        transition: isDragging ? DRAG_TRANSITION : transition
      }}
    >
      <Popover onOpenChange={setPickerOpen} open={pickerOpen}>
        <ContextMenu>
          <TooltipProvider delayDuration={0}>
            <Tooltip>
              <PopoverAnchor asChild>
                <ContextMenuTrigger asChild>
                  <TooltipTrigger asChild>
                    <button
                      className={cn(
                        'grid size-5 shrink-0 cursor-grab touch-none select-none place-items-center rounded-[3px] text-[0.5625rem] font-semibold uppercase leading-none transition-opacity hover:opacity-100',
                        active ? 'opacity-100' : 'opacity-55',
                        isDragging && 'cursor-grabbing opacity-100'
                      )}
                      style={{
                        backgroundColor: profileColorSoft(hue, active ? 30 : 22),
                        boxShadow: [ring, lift].filter(Boolean).join(', ') || undefined,
                        color: color ?? undefined
                      }}
                      type="button"
                      {...attributes}
                      {...listeners}
                      aria-label={label}
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
                      onPointerLeave={clearPress}
                      onPointerUp={clearPress}
                    >
                      {label.replace(/[^a-z0-9]/gi, '').charAt(0) || '?'}
                    </button>
                  </TooltipTrigger>
                </ContextMenuTrigger>
              </PopoverAnchor>
              <TooltipContent>
                <ActivityTooltip activity={activity} label={label} />
              </TooltipContent>
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

      {activity && !isDragging && <ActivityBadge activity={activity} label={label} onOpen={onOpenActivity} />}
    </div>
  )
}
