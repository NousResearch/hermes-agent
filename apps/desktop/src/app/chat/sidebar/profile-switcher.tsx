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
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
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
  $profileColors,
  $profileCreateRequest,
  $profileIcons,
  $profileOrder,
  $profiles,
  $profileScope,
  ALL_PROFILES,
  normalizeProfileKey,
  refreshActiveProfile,
  selectProfile,
  setProfileColor,
  setProfileIcon,
  setProfileOrder,
  setShowAllProfiles,
  sortByProfileOrder
} from '@/store/profile'
import {
  $attentionSessionIds,
  $cronSessions,
  $messagingSessions,
  $sessions,
  sessionPinId
} from '@/store/session'
import type { ProfileInfo } from '@/types/hermes'

import { CreateProfileDialog } from '../../profiles/create-profile-dialog'
import { DeleteProfileDialog } from '../../profiles/delete-profile-dialog'
import { RenameProfileDialog } from '../../profiles/rename-profile-dialog'
import { PROFILES_ROUTE } from '../../routes'

import { ProfileIconDialog } from './profile-icon-dialog'

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

const PROFILE_RAIL_SCROLL_FUZZ = 1

interface ProfileRailScrollState {
  canScrollLeft: boolean
  canScrollRight: boolean
  hasOverflow: boolean
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
  const order = useStore($profileOrder)
  const colors = useStore($profileColors)
  const icons = useStore($profileIcons)
  const sessions = useStore($sessions)
  const cronSessions = useStore($cronSessions)
  const messagingSessions = useStore($messagingSessions)
  const attentionSessionIds = useStore($attentionSessionIds)
  const navigate = useNavigate()

  const [createOpen, setCreateOpen] = useState(false)
  const [pendingRename, setPendingRename] = useState<null | ProfileInfo>(null)
  const [pendingDelete, setPendingDelete] = useState<null | ProfileInfo>(null)
  const [pendingSoul, setPendingSoul] = useState<null | string>(null)
  const [scrollState, setScrollState] = useState<ProfileRailScrollState>({
    canScrollLeft: false,
    canScrollRight: false,
    hasOverflow: false
  })
  const scrollRef = useRef<HTMLDivElement>(null)

  // Too many profiles for the square strip → collapse to the select. Declared
  // ahead of the wheel effect, which re-binds when the strip mounts/unmounts.
  const condensed = profiles.length > PROFILE_DROPDOWN_THRESHOLD

  const updateScrollState = useCallback(() => {
    const el = scrollRef.current

    if (!el) {
      setScrollState({ canScrollLeft: false, canScrollRight: false, hasOverflow: false })

      return
    }

    const maxScrollLeft = Math.max(0, el.scrollWidth - el.clientWidth)

    const next = {
      canScrollLeft: el.scrollLeft > PROFILE_RAIL_SCROLL_FUZZ,
      canScrollRight: el.scrollLeft < maxScrollLeft - PROFILE_RAIL_SCROLL_FUZZ,
      hasOverflow: maxScrollLeft > PROFILE_RAIL_SCROLL_FUZZ
    }

    setScrollState(current =>
      current.canScrollLeft === next.canScrollLeft &&
      current.canScrollRight === next.canScrollRight &&
      current.hasOverflow === next.hasOverflow
        ? current
        : next
    )
  }, [])

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
  const activeKey = normalizeProfileKey(scope)
  const defaultProfile = profiles.find(profile => profile.is_default)
  const onDefault = !isAll && activeKey === 'default'

  const named = sortByProfileOrder(
    profiles.filter(profile => !profile.is_default),
    order
  )

  const multiProfile = profiles.length > 1
  const namedProfileSignature = named.map(profile => profile.name).join('\0')

  useEffect(() => {
    const el = scrollRef.current

    if (!el) {
      updateScrollState()

      return
    }

    const onScroll = () => updateScrollState()
    const frame = window.requestAnimationFrame(updateScrollState)

    const resizeObserver =
      typeof ResizeObserver === 'undefined'
        ? null
        : new ResizeObserver(() => {
            updateScrollState()
          })

    el.addEventListener('scroll', onScroll, { passive: true })
    resizeObserver?.observe(el)

    return () => {
      window.cancelAnimationFrame(frame)
      el.removeEventListener('scroll', onScroll)
      resizeObserver?.disconnect()
    }
  }, [multiProfile, namedProfileSignature, updateScrollState])

  const scrollProfiles = (direction: 1 | -1) => {
    const el = scrollRef.current

    if (!el) {
      return
    }

    const maxScrollLeft = Math.max(0, el.scrollWidth - el.clientWidth)
    const step = Math.max(24, Math.floor(el.clientWidth * 0.75))
    el.scrollLeft = Math.min(maxScrollLeft, Math.max(0, el.scrollLeft + direction * step))
    updateScrollState()
  }

  const attentionCounts = useMemo(() => {
    const ids = new Set(attentionSessionIds)
    const counts = new Map<string, number>()

    for (const session of [...sessions, ...cronSessions, ...messagingSessions]) {
      if (!ids.has(session.id) && !ids.has(sessionPinId(session))) {
        continue
      }

      const profile = normalizeProfileKey(session.profile)
      counts.set(profile, (counts.get(profile) ?? 0) + 1)
    }

    return counts
  }, [attentionSessionIds, cronSessions, messagingSessions, sessions])

  const totalAttentionCount = [...attentionCounts.values()].reduce((sum, count) => sum + count, 0)

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
          <ProfilePill
            active={isAll || onDefault}
            badgeCount={isAll ? totalAttentionCount : (attentionCounts.get('default') ?? 0)}
            glyph={isAll ? 'layers' : 'home'}
            label={onDefault ? p.showAllProfiles : p.switchToProfile(defaultProfile.name)}
            onSelect={() => (onDefault ? setShowAllProfiles(true) : selectProfile(defaultProfile.name))}
          />
        ) : (
          <ProfilePill
            active={isAll}
            badgeCount={totalAttentionCount}
            glyph="layers"
            label={p.allProfiles}
            onSelect={() => setShowAllProfiles(true)}
          />
        ))}

      {/* Single-profile: the active default's home icon next to the create +. */}
      {!multiProfile && defaultProfile && (
        <ProfilePill
          active
          badgeCount={attentionCounts.get('default') ?? 0}
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
            colors={colors}
            icons={icons}
            onSelect={selectProfile}
            profiles={named}
          />
          <AddProfileButton label={p.newProfile} onClick={() => setCreateOpen(true)} />
        </div>
      ) : (
        <div className="flex min-w-0 flex-1 items-center gap-0.5">
          {scrollState.hasOverflow && (
            <ProfileRailScrollButton
              direction={-1}
              disabled={!scrollState.canScrollLeft}
              onClick={() => scrollProfiles(-1)}
            />
          )}

          <div
            aria-label="Profile list"
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
                        badgeCount={attentionCounts.get(normalizeProfileKey(profile.name)) ?? 0}
                        color={resolveProfileColor(profile.name, colors)}
                        icon={icons[normalizeProfileKey(profile.name)] ?? null}
                        key={profile.name}
                        label={profile.name}
                        onDelete={() => setPendingDelete(profile)}
                        onEditSoul={() => setPendingSoul(profile.name)}
                        onRecolor={color => setProfileColor(profile.name, color)}
                        onRename={() => setPendingRename(profile)}
                        onSelect={() => selectProfile(profile.name)}
                        onSetIcon={icon => setProfileIcon(profile.name, icon)}
                      />
                    ))}
                  </div>
                </SortableContext>
              </DndContext>
            )}

            <AddProfileButton label={p.newProfile} onClick={() => setCreateOpen(true)} />
          </div>

          {scrollState.hasOverflow && (
            <ProfileRailScrollButton
              direction={1}
              disabled={!scrollState.canScrollRight}
              onClick={() => scrollProfiles(1)}
            />
          )}
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

function ProfileRailScrollButton({
  direction,
  disabled,
  onClick
}: {
  direction: 1 | -1
  disabled: boolean
  onClick: () => void
}) {
  const label = direction === -1 ? 'Scroll profiles left' : 'Scroll profiles right'

  return (
    <Tip label={label}>
      <Button
        aria-label={label}
        className={cn(
          'shrink-0 bg-transparent text-(--ui-text-tertiary) opacity-70 hover:bg-(--ui-control-hover-background) hover:text-foreground hover:opacity-100',
          disabled && 'opacity-25 hover:bg-transparent hover:text-(--ui-text-tertiary)'
        )}
        disabled={disabled}
        onClick={onClick}
        size="icon-xs"
        type="button"
        variant="ghost"
      >
        <Codicon name={direction === -1 ? 'chevron-left' : 'chevron-right'} size="0.75rem" />
      </Button>
    </Tip>
  )
}

// The condensed rail: every named profile in one compact select. The trigger
// shows the active profile (tinted initial + name); on default/all scope it
// falls back to the placeholder since the left toggle pill carries that state.
function ProfileDropdown({
  activeKey,
  colors,
  icons,
  onSelect,
  profiles
}: {
  activeKey: null | string
  colors: Record<string, string>
  icons: Record<string, string>
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
          const key = normalizeProfileKey(profile.name)
          const color = resolveProfileColor(profile.name, colors)
          const hue = color ?? 'var(--ui-text-quaternary)'
          const icon = icons[key]

          return (
            <SelectItem key={profile.name} value={profile.name}>
              <span className="flex min-w-0 items-center gap-1.5">
                <span
                  aria-hidden="true"
                  className="grid size-4 shrink-0 place-items-center rounded-[3px] text-[0.5rem] font-semibold uppercase leading-none"
                  style={{ backgroundColor: profileColorSoft(hue, 22), color: color ?? undefined }}
                >
                  {icon || profile.name.replace(/[^a-z0-9]/gi, '').charAt(0) || '?'}
                </span>
                <span className="truncate">{profile.name}</span>
              </span>
            </SelectItem>
          )
        })}
      </SelectContent>
    </Select>
  )
}

interface ProfilePillProps {
  active: boolean
  badgeCount?: number
  // home / All / Manage are glyph action buttons (navigation, not identity).
  glyph: string
  label: string
  onSelect: () => void
}

function ProfilePill({ active, badgeCount = 0, glyph, label, onSelect }: ProfilePillProps) {
  const ariaLabel = waitingBadgeAriaLabel(label, badgeCount)

  return (
    <Tip label={label}>
      <Button
        aria-label={ariaLabel}
        aria-pressed={active}
        className={cn(
          'relative bg-transparent text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground',
          active && 'bg-(--ui-control-active-background) text-foreground'
        )}
        onClick={onSelect}
        size="icon-xs"
        type="button"
        variant="ghost"
      >
        <Codicon name={glyph} size="0.875rem" />
        <ProfileAttentionBadge count={badgeCount} />
      </Button>
    </Tip>
  )
}

interface ProfileSquareProps {
  active: boolean
  badgeCount?: number
  color: null | string
  icon: null | string
  label: string
  onSelect: () => void
  onRecolor: (color: null | string) => void
  onSetIcon: (icon: null | string) => void
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
  badgeCount = 0,
  color,
  icon,
  label,
  onDelete,
  onEditSoul,
  onRecolor,
  onRename,
  onSelect,
  onSetIcon
}: ProfileSquareProps) {
  const { t } = useI18n()
  const p = t.profiles
  const hue = color ?? 'var(--ui-text-quaternary)'
  const [pickerOpen, setPickerOpen] = useState(false)
  const [iconOpen, setIconOpen] = useState(false)
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
  const ariaLabel = waitingBadgeAriaLabel(label, badgeCount)

  const pickColor = (next: null | string) => {
    onRecolor(next)
    setPickerOpen(false)
    triggerHaptic('selection')
  }

  return (
    <>
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
                      aria-label={ariaLabel}
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
                      {icon || label.replace(/[^a-z0-9]/gi, '').charAt(0) || '?'}
                      <ProfileAttentionBadge count={badgeCount} />
                    </button>
                  </TooltipTrigger>
                </ContextMenuTrigger>
              </PopoverAnchor>
              <TooltipContent>
                <span data-bidi-plaintext="">{label}</span>
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
          <ContextMenuItem onSelect={() => setIconOpen(true)}>
            <Codicon name="symbol-misc" size="0.875rem" />
            <span>{p.icon}</span>
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
    <ProfileIconDialog
      hasIcon={Boolean(icon)}
      label={label}
      onClear={() => {
        onSetIcon(null)
        setIconOpen(false)
      }}
      onOpenChange={setIconOpen}
      onSelect={emoji => onSetIcon(emoji)}
      open={iconOpen}
    />
    </>
  )
}

function waitingBadgeAriaLabel(label: string, count: number): string {
  if (count <= 0) {
    return label
  }

  return `${label}, ${count} waiting for your reply`
}

function ProfileAttentionBadge({ count }: { count: number }) {
  if (count <= 0) {
    return null
  }

  return (
    <span
      aria-hidden="true"
      className="absolute -right-1 -top-1 grid h-3.5 min-w-3.5 place-items-center rounded-full bg-amber-400 px-0.5 text-[0.5rem] font-bold leading-none text-black shadow-[0_0_0_1px_var(--ui-sidebar-surface-background)]"
    >
      {count > 99 ? '99+' : count}
    </span>
  )
}
