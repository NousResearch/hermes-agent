import { useStore } from '@nanostores/react'
import { useVirtualizer } from '@tanstack/react-virtual'
import { type ReactNode, type RefObject, useEffect, useRef, useState } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { PetThumb } from '@/components/pet/pet-thumb'
import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { SearchField } from '@/components/ui/search-field'
import { Skeleton } from '@/components/ui/skeleton'
import { Slider } from '@/components/ui/slider'
import { Tip } from '@/components/ui/tooltip'
import { useMediaQuery } from '@/hooks/use-media-query'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Download, Loader2, PawPrint, Pencil, Trash2 } from '@/lib/icons'
import { isSubmitEnter } from '@/lib/ime'
import { selectableCardClass } from '@/lib/selectable-card'
import { cn } from '@/lib/utils'
import { $petInfo, $petRoam, setPetRoam } from '@/store/pet'
import {
  $petBusy,
  $petGallery,
  $petGalleryError,
  $petGalleryStatus,
  adoptPet,
  exportPet as exportPetAction,
  type GalleryPet,
  loadPetGallery,
  loadPetThumb,
  PET_SCALE_DEFAULT,
  PET_SCALE_MAX,
  PET_SCALE_MIN,
  rankedGalleryPets,
  removePet as removePetAction,
  renamePet as renamePetAction,
  setPetEnabled,
  setPetScale
} from '@/store/pet-gallery'
import { $gatewayState } from '@/store/session'

import { ListRow, SectionHeading, ToggleRow } from './primitives'

// A pet tile is one fixed-size card: the 40px (size-10) thumb + 2×8px vertical
// padding; name and slug truncate to one line each beside the thumb, so the
// row height never varies and no per-row measurement is needed.
export const PET_ROW_ESTIMATE_PX = 56
const PET_GRID_OVERSCAN_ROWS = 4

/**
 * Appearance opt-in for the floating petdex mascot. A thin view over the shared
 * `pet-gallery` store — it subscribes to the atoms and calls the store actions,
 * so the gallery is fetched once + cached and adopt/toggle/remove patch local
 * state instead of re-pulling the network gallery. The floating mascot polls
 * `pet.info`, so picking a pet here lights it up within a couple seconds.
 */
export function PetSettings() {
  const { t } = useI18n()
  const copy = t.settings.appearance.pet
  const { requestGateway } = useGatewayRequest()
  const gatewayState = useStore($gatewayState)
  const gallery = useStore($petGallery)
  const status = useStore($petGalleryStatus)
  const error = useStore($petGalleryError)
  const busySlug = useStore($petBusy)
  const petInfo = useStore($petInfo)
  const roam = useStore($petRoam)
  const [query, setQuery] = useState('')
  const [confirmDelete, setConfirmDelete] = useState<GalleryPet | null>(null)
  const [renameTarget, setRenameTarget] = useState<GalleryPet | null>(null)
  const [renameValue, setRenameValue] = useState('')
  const scale = petInfo.scale ?? PET_SCALE_DEFAULT
  // The grid's fixed-height scroll area — shared with the virtualizer so the
  // page owns one scroller, not two nested ones.
  const gridScrollRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (gatewayState !== 'open') {
      return
    }

    void loadPetGallery(requestGateway)
  }, [gatewayState, requestGateway])

  const enabled = gallery?.enabled ?? false
  const active = gallery?.active ?? ''
  const pets = gallery?.pets ?? []
  const staleBackend = status === 'stale'

  const selectPet = (slug: string) => {
    void adoptPet(requestGateway, slug, copy.adoptFailed(slug)).then(ok => ok && triggerHaptic('crisp'))
  }

  const removePet = (slug: string) => {
    void removePetAction(requestGateway, slug, copy.uninstallFailed(slug)).then(ok => ok && triggerHaptic('crisp'))
  }

  const exportPet = (slug: string) => {
    void exportPetAction(requestGateway, slug, copy.exportFailed(slug)).then(ok => ok && triggerHaptic('crisp'))
  }

  const saveRename = () => {
    if (!renameTarget || !renameValue.trim()) {
      return
    }

    // Optimistic: the rename paints instantly, so close now and let the RPC
    // settle in the background (it rolls back + surfaces an error on failure).
    const { slug } = renameTarget
    setRenameTarget(null)
    triggerHaptic('crisp')
    void renamePetAction(requestGateway, slug, renameValue, copy.renameFailed(slug))
  }

  const toggle = (on: boolean) => {
    void setPetEnabled(requestGateway, on, {
      noneAvailable: copy.noneAvailable,
      fallback: on ? copy.turnOnFailed : copy.turnOffFailed
    }).then(ok => ok && triggerHaptic('crisp'))
  }

  // The petdex catalog is thousands of entries, so rank for the picker and
  // virtualize: only the rows in view mount (each card carries its own
  // IntersectionObserver'd thumbnail), and the whole catalog stays reachable.
  const sorted = rankedGalleryPets(gallery, query)

  // Column count mirrors the grid's `sm:grid-cols-2 xl:grid-cols-3` breakpoints
  // (Tailwind sm = 640px, xl = 1280px) so a virtual row paints exactly what the
  // plain grid would.
  const wide = useMediaQuery('(min-width: 1280px)')
  const medium = useMediaQuery('(min-width: 640px)')
  const columns = wide ? 3 : medium ? 2 : 1

  return (
    <div>
      <SectionHeading icon={PawPrint} page title={copy.title} />
      <p className="max-w-2xl text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
        {copy.intro}
      </p>

      {staleBackend && (
        <p className="mt-2 rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) px-3 py-2 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
          {copy.restartHint}
        </p>
      )}

      <div className="mt-2">
        <ToggleRow
          below={
            <>
              <SearchField
                containerClassName="mt-3 w-full"
                inputClassName="flex-1"
                onChange={setQuery}
                placeholder={copy.searchPlaceholder}
                value={query}
              />
              {/* Fixed-height scroll area so filtering never grows/shrinks the
                  page (no layout thrash); the grid scrolls inside it. */}
              <div className="mt-3 h-72 overflow-y-auto pr-1" ref={gridScrollRef}>
                {status === 'loading' && pets.length === 0 ? (
                  // First load keeps the grid's shape rather than flashing the
                  // "unreachable" copy before the gallery has even arrived.
                  <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
                    {Array.from({ length: 6 }, (_, i) => (
                      <div className="flex items-center gap-2.5 px-2.5 py-2" key={i}>
                        <Skeleton className="size-10 shrink-0 rounded-md" />
                        <div className="min-w-0 flex-1 space-y-1.5">
                          <Skeleton className="h-3.5 w-24 max-w-full" />
                          <Skeleton className="h-3 w-16 max-w-full" />
                        </div>
                      </div>
                    ))}
                  </div>
                ) : pets.length === 0 ? (
                  <p className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                    {copy.unreachable}
                  </p>
                ) : sorted.length === 0 ? (
                  <p className="wrap-anywhere text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                    {copy.noMatch(query)}
                  </p>
                ) : (
                  <VirtualPetGrid
                    active={active}
                    busySlug={busySlug}
                    columns={columns}
                    copy={copy}
                    enabled={enabled}
                    exportPet={exportPet}
                    onConfirmDelete={setConfirmDelete}
                    onRename={pet => {
                      setRenameValue(pet.displayName)
                      setRenameTarget(pet)
                    }}
                    pets={sorted}
                    removePet={removePet}
                    requestGateway={requestGateway}
                    scrollRef={gridScrollRef}
                    selectPet={selectPet}
                  />
                )}
              </div>
              {/* Always-present status line so its appearance never shifts layout. */}
              <p className="mt-2 min-h-4 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                {error ? <span className="text-(--ui-red)">{error}</span> : copy.count(sorted.length)}
              </p>
            </>
          }
          checked={enabled}
          description={copy.chooseDesc}
          label={copy.chooseTitle}
          onChange={toggle}
          wide
        />

        {enabled && (
          <ListRow
            action={
              <div className="flex items-center gap-3">
                <Slider
                  aria-label={copy.scaleTitle}
                  max={PET_SCALE_MAX}
                  min={PET_SCALE_MIN}
                  onChange={event => {
                    triggerHaptic('selection')
                    setPetScale(requestGateway, Number(event.target.value))
                  }}
                  step={0.05}
                  value={scale}
                />
                <span className="w-9 text-right text-[length:var(--conversation-caption-font-size)] tabular-nums text-(--ui-text-tertiary)">
                  {`${Math.round(scale * 100)}%`}
                </span>
              </div>
            }
            description={copy.scaleDesc}
            title={copy.scaleTitle}
          />
        )}

        {enabled && (
          <ToggleRow checked={roam} description={copy.roamDesc} label={copy.roamTitle} onChange={setPetRoam} />
        )}
      </div>

      <ConfirmDialog
        confirmLabel={copy.deleteConfirm}
        description={copy.deleteBody}
        destructive
        onClose={() => setConfirmDelete(null)}
        onConfirm={async () => {
          if (confirmDelete) {
            const ok = await removePetAction(
              requestGateway,
              confirmDelete.slug,
              copy.uninstallFailed(confirmDelete.slug)
            )

            if (!ok) {
              throw new Error(copy.uninstallFailed(confirmDelete.slug))
            }

            triggerHaptic('crisp')
          }
        }}
        open={confirmDelete !== null}
        title={confirmDelete ? copy.deleteTitle(confirmDelete.displayName) : ''}
      />

      <Dialog onOpenChange={open => !open && setRenameTarget(null)} open={renameTarget !== null}>
        <DialogContent className="max-w-sm">
          <DialogHeader>
            <DialogTitle>{copy.renameTitle}</DialogTitle>
          </DialogHeader>
          <Input
            autoFocus
            onChange={event => setRenameValue(event.target.value)}
            onKeyDown={event => {
              if (isSubmitEnter(event)) {
                event.preventDefault()
                saveRename()
              }
            }}
            placeholder={copy.renamePlaceholder}
            value={renameValue}
          />
          <DialogFooter>
            <Button onClick={() => setRenameTarget(null)} type="button" variant="ghost">
              {t.common.cancel}
            </Button>
            <Button disabled={!renameValue.trim()} onClick={saveRename}>
              {copy.renameSave}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

/** One rendered row of the virtualized pet grid: `columns` cards laid out like
 *  the plain `grid gap-2 sm:grid-cols-2 xl:grid-cols-3` the page used to paint. */
function VirtualPetGrid({
  active,
  busySlug,
  columns,
  copy,
  enabled,
  exportPet,
  pets,
  removePet,
  requestGateway,
  scrollRef,
  selectPet,
  onConfirmDelete,
  onRename
}: {
  active: string
  busySlug: string | null
  columns: number
  copy: ReturnType<typeof useI18n>['t']['settings']['appearance']['pet']
  enabled: boolean
  exportPet: (slug: string) => void
  pets: GalleryPet[]
  removePet: (slug: string) => void
  requestGateway: ReturnType<typeof useGatewayRequest>['requestGateway']
  /** The page's fixed-height scroll area (`h-72 overflow-y-auto`) — the grid
   *  rides inside it rather than owning a second nested scroller. */
  scrollRef: RefObject<HTMLDivElement | null>
  selectPet: (slug: string) => void
  onConfirmDelete: (pet: GalleryPet) => void
  onRename: (pet: GalleryPet) => void
}) {
  const virtualizer = useVirtualizer({
    count: Math.ceil(pets.length / columns),
    estimateSize: () => PET_ROW_ESTIMATE_PX,
    getItemKey: index => pets[index * columns]?.slug ?? String(index),
    getScrollElement: () => scrollRef.current,
    // jsdom-friendly default; the real rect takes over on first observe.
    initialRect: { height: 288, width: 600 },
    overscan: PET_GRID_OVERSCAN_ROWS
  })

  const rows = virtualizer.getVirtualItems()

  return (
    <div className="relative" style={{ height: `${virtualizer.getTotalSize()}px` }}>
      {rows.map(virtualRow => {
        const rowPets = pets.slice(virtualRow.index * columns, (virtualRow.index + 1) * columns)

        return (
          <div
            className="absolute left-0 top-0 grid w-full gap-2"
            data-index={virtualRow.index}
            key={virtualRow.key}
            style={{
              gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
              transform: `translateY(${virtualRow.start}px)`
            }}
          >
            {rowPets.map(pet => (
              <PetCard
                active={active}
                busy={busySlug === pet.slug}
                copy={copy}
                enabled={enabled}
                exportPet={exportPet}
                key={pet.slug}
                onConfirmDelete={onConfirmDelete}
                onRename={onRename}
                pet={pet}
                removePet={removePet}
                requestGateway={requestGateway}
                selectPet={selectPet}
              />
            ))}
          </div>
        )
      })}
    </div>
  )
}

/** A single selectable pet tile (thumbnail, name, slug, hover actions). */
function PetCard({
  active,
  busy,
  copy,
  enabled,
  pet,
  requestGateway,
  exportPet,
  removePet,
  selectPet,
  onConfirmDelete,
  onRename
}: {
  active: string
  busy: boolean
  copy: ReturnType<typeof useI18n>['t']['settings']['appearance']['pet']
  enabled: boolean
  pet: GalleryPet
  requestGateway: ReturnType<typeof useGatewayRequest>['requestGateway']
  exportPet: (slug: string) => void
  removePet: (slug: string) => void
  selectPet: (slug: string) => void
  onConfirmDelete: (pet: GalleryPet) => void
  onRename: (pet: GalleryPet) => void
}) {
  const isActive = enabled && active === pet.slug

  return (
    <div className="group relative">
      <button
        className={cn(
          'flex w-full items-center gap-2.5 px-2.5 py-2 text-left disabled:opacity-50',
          selectableCardClass({ active: isActive, prominent: pet.installed })
        )}
        disabled={busy}
        onClick={() => selectPet(pet.slug)}
        type="button"
      >
        <PetThumb
          alt={pet.displayName}
          load={(slug, url) => loadPetThumb(requestGateway, slug, url)}
          slug={pet.slug}
          url={pet.spritesheetUrl}
        />
        <span className="min-w-0 flex-1">
          <span className="flex items-center gap-1.5">
            <span className="truncate text-[length:var(--conversation-text-font-size)] font-medium">
              {pet.displayName}
            </span>
            {pet.generated && (
              <span className="shrink-0 rounded-full bg-primary/15 px-1.5 py-px text-[0.625rem] font-medium text-primary">
                {copy.generatedTag}
              </span>
            )}
          </span>
          <span className="block truncate text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
            {pet.slug}
            {pet.installed ? ` · ${copy.installedTag}` : ''}
          </span>
        </span>
        {busy && <Loader2 className="size-4 shrink-0 animate-spin text-(--ui-text-tertiary)" />}
      </button>
      {!busy && (pet.installed || pet.generated) && (
        <div className="absolute right-1.5 top-1.5 flex gap-1 opacity-0 transition focus-within:opacity-100 group-hover:opacity-100">
          {pet.generated && (
            <PetAction icon={<Pencil className="size-3.5" />} label={copy.rename(pet.displayName)} onClick={() => onRename(pet)} />
          )}
          {pet.generated && (
            <PetAction
              icon={<Download className="size-3.5" />}
              label={copy.exportPet(pet.displayName)}
              onClick={() => exportPet(pet.slug)}
            />
          )}
          {pet.installed && (
            // Generated pets have no remote source — deletion is
            // permanent, so confirm; petdex pets just uninstall.
            <PetAction
              danger
              icon={<Trash2 className="size-3.5" />}
              label={pet.generated ? copy.delete(pet.displayName) : copy.uninstall(pet.displayName)}
              onClick={() => (pet.generated ? onConfirmDelete(pet) : removePet(pet.slug))}
            />
          )}
        </div>
      )}
    </div>
  )
}

/** A single hover-revealed icon action on a pet card (rename / export / delete). */
function PetAction({
  danger,
  icon,
  label,
  onClick
}: {
  danger?: boolean
  icon: ReactNode
  label: string
  onClick: () => void
}) {
  return (
    <Tip label={label}>
      <button
        aria-label={label}
        className={cn(
          'grid size-6 place-items-center rounded-md bg-(--ui-bg-elevated)/80 text-(--ui-text-tertiary) backdrop-blur-sm transition',
          danger ? 'hover:text-(--ui-red)' : 'hover:text-foreground'
        )}
        onClick={onClick}
        type="button"
      >
        {icon}
      </button>
    </Tip>
  )
}
