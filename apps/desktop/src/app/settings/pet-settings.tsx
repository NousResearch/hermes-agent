import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { PetThumb } from '@/components/pet/pet-thumb'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Loader2, PawPrint, Trash2 } from '@/lib/icons'
import { selectableCardClass } from '@/lib/selectable-card'
import { cn } from '@/lib/utils'
import { $petInfo } from '@/store/pet'
import {
  $petBusy,
  $petGallery,
  $petGalleryError,
  $petGalleryStatus,
  adoptPet,
  loadPetGallery,
  loadPetThumb,
  PET_SCALE_DEFAULT,
  PET_SCALE_MAX,
  PET_SCALE_MIN,
  rankedGalleryPets,
  removePet as removePetAction,
  setPetEnabled,
  setPetScale
} from '@/store/pet-gallery'
import { $gatewayState } from '@/store/session'

import { ListRow, SectionHeading } from './primitives'

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
  const [query, setQuery] = useState('')
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

  return (
    <div>
      <SectionHeading icon={PawPrint} title={copy.title} />
      <p className="max-w-2xl text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
        {copy.intro}
      </p>

      {staleBackend && (
        <p className="mt-2 rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) px-3 py-2 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
          {copy.restartHint}
        </p>
      )}

      <div className="mt-2">
        <ListRow
          below={
            <>
              <input
                className="mt-3 w-full rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) px-3 py-1.5 text-[length:var(--conversation-caption-font-size)] outline-none placeholder:text-(--ui-text-tertiary) focus:border-(--ui-stroke-secondary)"
                onChange={event => setQuery(event.target.value)}
                placeholder={copy.searchPlaceholder}
                spellCheck={false}
                value={query}
              />
              {/* Fixed-height scroll area so filtering never grows/shrinks the
                  page (no layout thrash); the grid scrolls inside it. */}
              <div className="mt-3 h-72 overflow-y-auto pr-1">
                {pets.length === 0 ? (
                  <p className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                    {copy.unreachable}
                  </p>
                ) : shown.length === 0 ? (
                  <p className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                    {copy.noMatch(query)}
                  </p>
                ) : (
                  <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
                    {shown.map(pet => {
                      const isActive = enabled && active === pet.slug
                      const isBusy = busySlug === pet.slug

                      return (
                        <div className="group relative" key={pet.slug}>
                          <button
                            className={cn(
                              'flex w-full items-center gap-2.5 px-2.5 py-2 text-left disabled:opacity-50',
                              selectableCardClass({ active: isActive, prominent: pet.installed })
                            )}
                            disabled={isBusy}
                            onClick={() => void selectPet(pet.slug)}
                            type="button"
                          >
                            <PetThumb
                              alt={pet.displayName}
                              load={(slug, url) => loadPetThumb(requestGateway, slug, url)}
                              slug={pet.slug}
                              url={pet.spritesheetUrl}
                            />
                            <span className="min-w-0 flex-1">
                              <span className="block truncate text-[length:var(--conversation-text-font-size)] font-medium">
                                {pet.displayName}
                              </span>
                              <span className="block truncate text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                                {pet.slug}
                                {pet.installed ? ` · ${copy.installedTag}` : ''}
                              </span>
                            </span>
                            {isBusy && <Loader2 className="size-4 shrink-0 animate-spin text-(--ui-text-tertiary)" />}
                          </button>
                          {pet.installed && !isBusy && (
                            <button
                              aria-label={copy.uninstall(pet.displayName)}
                              className="absolute right-1.5 top-1.5 grid size-6 place-items-center rounded-md bg-(--ui-bg-elevated)/80 text-(--ui-text-tertiary) opacity-0 backdrop-blur-sm transition hover:text-(--ui-red) focus-visible:opacity-100 group-hover:opacity-100"
                              onClick={() => void removePet(pet.slug)}
                              title={copy.uninstall(pet.displayName)}
                              type="button"
                            >
                              <Trash2 className="size-3.5" />
                            </button>
                          )}
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
              {/* Always-present status line so its appearance never shifts layout. */}
              <p className="mt-2 min-h-4 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                {error ? (
                  <span className="text-(--ui-red)">{error}</span>
                ) : sorted.length > RENDER_CAP ? (
                  copy.countCapped(RENDER_CAP, sorted.length)
                ) : (
                  copy.count(sorted.length)
                )}
              </p>
            </>
          }
          description={copy.chooseDesc}
          title={
            <div className="flex items-center justify-between gap-3">
              <span>{copy.chooseTitle}</span>
              <SegmentedControl
                onChange={id => void toggle(id === 'on')}
                options={[
                  { id: 'off', label: copy.off },
                  { id: 'on', label: copy.on }
                ]}
                value={enabled ? 'on' : 'off'}
              />
            </div>
          }
          wide
        />

        {enabled && (
          <ListRow
            action={
              <div className="flex items-center gap-3">
                <input
                  aria-label={copy.scaleTitle}
                  className="h-1 w-40 cursor-pointer appearance-none rounded-full bg-(--ui-stroke-tertiary)"
                  max={PET_SCALE_MAX}
                  min={PET_SCALE_MIN}
                  onChange={event => {
                    triggerHaptic('selection')
                    setPetScale(requestGateway, Number(event.target.value))
                  }}
                  step={0.05}
                  style={{ accentColor: 'var(--dt-primary)' }}
                  type="range"
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
      </div>
    </div>
  )
}
