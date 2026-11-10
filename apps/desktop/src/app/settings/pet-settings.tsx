import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect, useState } from 'react'

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

/**
 * Appearance opt-in for the floating petdex mascot. A thin view over the shared
 * `pet-gallery` store — it subscribes to the atoms and calls the store actions,
 * so the gallery is fetched once + cached and adopt/toggle/remove patch local
 * state instead of re-pulling the network gallery. The floating mascot polls
 * `pet.info`, so picking a pet here lights it up within a couple seconds.
 *
 * `page` enables the Figma Appearance → Pet chrome (426:17109).
 */
export function PetSettings({ page = false }: { page?: boolean } = {}) {
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

  // The petdex catalog is thousands of entries, so rank + cap how many render.
  const RENDER_CAP = 60
  const sorted = rankedGalleryPets(gallery, query)
  const shown = sorted.slice(0, RENDER_CAP)
  const thumbSize = page ? 72 : 40

  const galleryBody = (
    <>
      <SearchField
        containerClassName={cn('mt-3 w-full', page && 'appearance-pet-search')}
        inputClassName="flex-1"
        onChange={setQuery}
        placeholder={copy.searchPlaceholder}
        value={query}
        variant={page ? 'box' : 'underline'}
      />
      {/* Overview keeps a fixed scroll pane so filtering never thrash-layouts
          the whole Appearance page. The dedicated Pet page expands instead —
          no inset scrollbar beside the grid (Figma 426:17109). */}
      <div className={cn('mt-3', page ? 'pb-1' : 'h-72 overflow-y-auto pr-1')}>
        {status === 'loading' && pets.length === 0 ? (
          // First load keeps the grid's shape rather than flashing the
          // "unreachable" copy before the gallery has even arrived.
          <div
            className={cn(
              'grid gap-3',
              page ? 'grid-cols-2 sm:grid-cols-3 xl:grid-cols-4' : 'sm:grid-cols-2 xl:grid-cols-3'
            )}
          >
            {Array.from({ length: page ? 8 : 6 }, (_, i) =>
              page ? (
                <div className="flex flex-col items-center gap-2 px-3 py-3" key={i}>
                  <Skeleton className="size-[72px] shrink-0 rounded-2xl" />
                  <Skeleton className="h-3.5 w-20" />
                  <Skeleton className="h-3 w-14" />
                </div>
              ) : (
                <div className="flex items-center gap-2.5 px-2.5 py-2" key={i}>
                  <Skeleton className="size-10 shrink-0 rounded-md" />
                  <div className="min-w-0 flex-1 space-y-1.5">
                    <Skeleton className="h-3.5 w-24 max-w-full" />
                    <Skeleton className="h-3 w-16 max-w-full" />
                  </div>
                </div>
              )
            )}
          </div>
        ) : pets.length === 0 ? (
          <p className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
            {copy.unreachable}
          </p>
        ) : shown.length === 0 ? (
          <p className="wrap-anywhere text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
            {copy.noMatch(query)}
          </p>
        ) : (
          <div
            className={cn(
              'grid gap-3',
              page ? 'grid-cols-2 sm:grid-cols-3 xl:grid-cols-4' : 'sm:grid-cols-2 xl:grid-cols-3'
            )}
          >
            {shown.map(pet => {
              const isActive = enabled && active === pet.slug
              const isBusy = busySlug === pet.slug

              return (
                <div className="group relative" key={pet.slug}>
                  <button
                    className={cn(
                      page
                        ? 'flex w-full flex-col items-center px-3 py-3 text-center disabled:opacity-50'
                        : 'flex w-full items-center gap-2.5 px-2.5 py-2 text-left disabled:opacity-50',
                      page
                        ? undefined
                        : selectableCardClass({ active: isActive, prominent: pet.installed })
                    )}
                    data-active={page && isActive ? 'true' : undefined}
                    data-appearance-pet-item={page ? '' : undefined}
                    disabled={isBusy}
                    onClick={() => void selectPet(pet.slug)}
                    type="button"
                  >
                    <PetThumb
                      alt={pet.displayName}
                      load={(slug, url) => loadPetThumb(requestGateway, slug, url)}
                      size={thumbSize}
                      slug={pet.slug}
                      url={pet.spritesheetUrl}
                    />
                    <span className={cn('min-w-0', page ? 'mt-2 w-full' : 'flex-1')}>
                      <span className={cn('flex items-center gap-1.5', page && 'justify-center')}>
                        <span
                          className={cn(
                            'truncate font-medium',
                            page
                              ? 'text-[14px] font-semibold text-[#0f172a] dark:text-foreground'
                              : 'text-[length:var(--conversation-text-font-size)]'
                          )}
                        >
                          {pet.displayName}
                        </span>
                        {pet.generated && (
                          <span className="shrink-0 rounded-full bg-primary/15 px-1.5 py-px text-[0.625rem] font-medium text-primary">
                            {copy.generatedTag}
                          </span>
                        )}
                      </span>
                      <span
                        className={cn(
                          'block truncate text-(--ui-text-tertiary)',
                          page
                            ? 'mt-0.5 text-[12px] text-[#94a3b8]'
                            : 'text-[length:var(--conversation-caption-font-size)]'
                        )}
                      >
                        {pet.slug}
                        {!page && pet.installed ? ` · ${copy.installedTag}` : ''}
                      </span>
                    </span>
                    {isBusy && <Loader2 className="size-4 shrink-0 animate-spin text-(--ui-text-tertiary)" />}
                  </button>
                  {!isBusy && (pet.installed || pet.generated) && (
                    <div className="absolute right-1.5 top-1.5 flex gap-1 opacity-0 transition focus-within:opacity-100 group-hover:opacity-100">
                      {pet.generated && (
                        <PetAction
                          icon={<Pencil className="size-3.5" />}
                          label={copy.rename(pet.displayName)}
                          onClick={() => {
                            setRenameValue(pet.displayName)
                            setRenameTarget(pet)
                          }}
                        />
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
                          onClick={() => (pet.generated ? setConfirmDelete(pet) : removePet(pet.slug))}
                        />
                      )}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>
      {/* Always-present status line so its appearance never shifts layout. */}
      <p
        className={cn(
          'mt-2 min-h-4 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)',
          page && 'mb-3'
        )}
      >
        {error ? (
          <span className="text-(--ui-red)">{error}</span>
        ) : sorted.length > RENDER_CAP ? (
          copy.countCapped(RENDER_CAP, sorted.length)
        ) : (
          copy.count(sorted.length)
        )}
      </p>
    </>
  )

  const extras = enabled && (
    <>
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
      <ToggleRow checked={roam} description={copy.roamDesc} label={copy.roamTitle} onChange={setPetRoam} />
    </>
  )

  const dialogs = (
    <>
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
    </>
  )

  if (page) {
    return (
      <section className="grid gap-3.5" data-appearance-pet="">
        <header className="space-y-1.5">
          <h1 className="text-[32px] font-extrabold leading-none tracking-tight text-[#0f172a] dark:text-foreground">
            {copy.title}
          </h1>
          <p className="max-w-3xl text-[14px] leading-snug text-[#475569] dark:text-(--ui-text-tertiary)">
            {copy.intro}
          </p>
        </header>

        {staleBackend && (
          <p className="rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) px-3 py-2 text-[13px] leading-snug text-(--ui-text-tertiary)">
            {copy.restartHint}
          </p>
        )}

        <div data-appearance-pet-card="">
          <div className="scroll-mt-6" data-chat-settings-row="">
            <ToggleRow
              below={galleryBody}
              checked={enabled}
              description={copy.chooseDesc}
              label={copy.chooseTitle}
              onChange={toggle}
              wide
            />
          </div>
          {enabled && (
            <>
              <div className="scroll-mt-6" data-chat-settings-row="">
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
                      <span className="w-9 text-right text-[13px] tabular-nums text-[#64748b]">
                        {`${Math.round(scale * 100)}%`}
                      </span>
                    </div>
                  }
                  description={copy.scaleDesc}
                  title={copy.scaleTitle}
                />
              </div>
              <div className="scroll-mt-6" data-chat-settings-row="">
                <ToggleRow checked={roam} description={copy.roamDesc} label={copy.roamTitle} onChange={setPetRoam} />
              </div>
            </>
          )}
        </div>

        {dialogs}
      </section>
    )
  }

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
          below={galleryBody}
          checked={enabled}
          description={copy.chooseDesc}
          label={copy.chooseTitle}
          onChange={toggle}
          wide
        />
        {extras}
      </div>

      {dialogs}
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
