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
import { Tip } from '@/components/ui/tooltip'
import { triggerHaptic } from '@/lib/haptics'
import { profileColor, profileColorSoft } from '@/lib/profile-color'
import { cn } from '@/lib/utils'
import {
  $activeGatewayProfile,
  $profileOrder,
  $profiles,
  $profileScope,
  ALL_PROFILES,
  normalizeProfileKey,
  refreshActiveProfile,
  selectProfile,
  setProfileOrder,
  setShowAllProfiles,
  sortByProfileOrder
} from '@/store/profile'

import { CreateProfileDialog } from '../../profiles/create-profile-dialog'
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
// profile users see only the "+" (create their first profile); everything else
// appears once a second profile exists.
export function ProfileRail() {
  const profiles = useStore($profiles)
  const scope = useStore($profileScope)
  const gatewayProfile = useStore($activeGatewayProfile)
  const order = useStore($profileOrder)
  const navigate = useNavigate()

  const [createOpen, setCreateOpen] = useState(false)
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

  const named = sortByProfileOrder(profiles.filter(profile => !profile.is_default), order)
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

  return (
    <div aria-label="Profiles" className="flex items-center gap-0.5" role="tablist">
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
            label={onDefault ? 'Show all profiles' : `Switch to ${defaultProfile.name}`}
            onSelect={() => (onDefault ? setShowAllProfiles(true) : selectProfile(defaultProfile.name))}
          />
        ) : (
          <ProfilePill active={isAll} glyph="layers" label="All profiles" onSelect={() => setShowAllProfiles(true)} />
        ))}

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
                    color={profileColor(profile.name)}
                    key={profile.name}
                    label={profile.name}
                    onSelect={() => selectProfile(profile.name)}
                  />
                ))}
              </div>
            </SortableContext>
          </DndContext>
        )}

        <Tip label="New profile">
          <button
            aria-label="New profile"
            className="grid size-5 shrink-0 place-items-center rounded-[3px] text-(--ui-text-tertiary) opacity-55 transition hover:bg-(--ui-control-hover-background) hover:text-foreground hover:opacity-100"
            onClick={() => setCreateOpen(true)}
            type="button"
          >
            <Codicon name="add" size="0.75rem" />
          </button>
        </Tip>
      </div>

      {multiProfile && (
        <ProfilePill active={false} glyph="ellipsis" label="Manage profiles…" onSelect={() => navigate(PROFILES_ROUTE)} />
      )}

      <CreateProfileDialog onClose={() => setCreateOpen(false)} onCreated={refreshActiveProfile} open={createOpen} />
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
}

// A profile *is* its colored square — no icon-button chrome. Soft profile-tint
// fill + the initial in the full color; the active one pops to full opacity with
// a color ring. These pack tightly so the rail reads as a strip of profiles,
// and drag-sort to reorder (a tap below the drag threshold still selects).
function ProfileSquare({ active, color, label, onSelect }: ProfileSquareProps) {
  const hue = color ?? 'var(--ui-text-quaternary)'

  const { attributes, isDragging, listeners, setNodeRef, transform, transition } = useSortable({
    id: label,
    transition: RAIL_TRANSITION
  })

  const base = CSS.Transform.toString(transform)
  const ring = active ? `inset 0 0 0 1.5px ${hue}` : ''
  const lift = isDragging ? '0 6px 16px -4px rgb(0 0 0 / 0.4)' : ''

  return (
    <Tip label={label}>
      <button
        className={cn(
          'grid size-5 shrink-0 cursor-grab touch-none select-none place-items-center rounded-[3px] text-[0.5625rem] font-semibold uppercase leading-none transition-opacity hover:opacity-100',
          active ? 'opacity-100' : 'opacity-55',
          isDragging && 'z-10 cursor-grabbing opacity-100'
        )}
        onClick={onSelect}
        ref={setNodeRef}
        style={{
          backgroundColor: profileColorSoft(hue, active ? 30 : 22),
          boxShadow: [ring, lift].filter(Boolean).join(', ') || undefined,
          color: color ?? undefined,
          // Glide the dragged square between snapped cells with a little overshoot
          // (no scale — the overflow-x strip would clip it).
          transform: base,
          transition: isDragging ? DRAG_TRANSITION : transition
        }}
        type="button"
        {...attributes}
        {...listeners}
        aria-label={label}
        aria-pressed={active}
      >
        {label.replace(/[^a-z0-9]/gi, '').charAt(0) || '?'}
      </button>
    </Tip>
  )
}
