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

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ColorSwatches } from '@/components/ui/color-swatches'
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from '@/components/ui/context-menu'
import { Popover, PopoverAnchor, PopoverContent } from '@/components/ui/popover'
import { Tip, Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { PROFILE_SWATCHES, profileColorSoft, resolveProfileColor } from '@/lib/profile-color'
import { cn } from '@/lib/utils'
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
import type { ProfileInfo } from '@/types/hermes'

import { CreateProfileDialog } from '../../profiles/create-profile-dialog'
import { DeleteProfileDialog } from '../../profiles/delete-profile-dialog'
import { RenameProfileDialog } from '../../profiles/rename-profile-dialog'
import { PROFILES_ROUTE } from '../../routes'

const RAIL_GAP = 4 // px — matches gap-1 between squares.

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
  const navigate = useNavigate()

  const [createOpen, setCreateOpen] = useState(false)
  const [pendingRename, setPendingRename] = useState<null | ProfileInfo>(null)
  const [pendingDelete, setPendingDelete] = useState<null | ProfileInfo>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

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
  }, [])

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
          glyph="home"
          label={defaultProfile.name}
          onSelect={() => selectProfile(defaultProfile.name)}
        />
      )}

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
                    color={resolveProfileColor(profile.name, colors)}
                    key={profile.name}
                    label={profile.name}
                    onDelete={() => setPendingDelete(profile)}
                    onRecolor={color => setProfileColor(profile.name, color)}
                    onRename={() => setPendingRename(profile)}
                    onSelect={() => selectProfile(profile.name)}
                  />
                ))}
              </div>
            </SortableContext>
          </DndContext>
        )}

        <Tip label={p.newProfile}>
          <button
            aria-label={p.newProfile}
            className="grid size-5 shrink-0 place-items-center rounded-[3px] text-(--ui-text-tertiary) opacity-55 transition hover:bg-(--ui-control-hover-background) hover:text-foreground hover:opacity-100"
            onClick={() => setCreateOpen(true)}
            type="button"
          >
            <Codicon name="add" size="0.75rem" />
          </button>
        </Tip>
      </div>

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
  color: null | string
  label: string
  onSelect: () => void
  onRecolor: (color: null | string) => void
  onRename: () => void
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
function ProfileSquare({ active, color, label, onDelete, onRecolor, onRename, onSelect }: ProfileSquareProps) {
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
            <TooltipContent>{label}</TooltipContent>
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
            <Codicon name="edit" size="0.875rem" />
            <span>{p.rename}</span>
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
