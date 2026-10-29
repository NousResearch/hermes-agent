import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect, useState } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { CompanionThumb } from '@/components/companion/companion-thumb'
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
import { $companionInfo, $companionRoam, setCompanionRoam } from '@/store/companion'
import {
  $companionBusy,
  $companionGallery,
  $companionGalleryError,
  $companionGalleryStatus,
  adoptCompanion,
  COMPANION_SCALE_DEFAULT,
  COMPANION_SCALE_MAX,
  COMPANION_SCALE_MIN,
  exportCompanion as exportCompanionAction,
  type GalleryCompanion,
  loadCompanionGallery,
  loadCompanionThumb,
  rankedGalleryCompanions,
  removeCompanion as removeCompanionAction,
  renameCompanion as renameCompanionAction,
  setCompanionEnabled,
  setCompanionScale
} from '@/store/companion-gallery'
import { $gatewayState } from '@/store/session'

import { ListRow, SectionHeading, ToggleRow } from './primitives'

/**
 * Appearance opt-in for the floating companiondex mascot. A thin view over the shared
 * `companion-gallery` store — it subscribes to the atoms and calls the store actions,
 * so the gallery is fetched once + cached and adopt/toggle/remove patch local
 * state instead of re-pulling the network gallery. The floating mascot polls
 * `companion.info`, so picking a companion here lights it up within a couple seconds.
 */
export function CompanionSettings() {
  const { t } = useI18n()
  const copy = t.settings.appearance.companion
  const { requestGateway } = useGatewayRequest()
  const gatewayState = useStore($gatewayState)
  const gallery = useStore($companionGallery)
  const status = useStore($companionGalleryStatus)
  const error = useStore($companionGalleryError)
  const busySlug = useStore($companionBusy)
  const companionInfo = useStore($companionInfo)
  const roam = useStore($companionRoam)
  const [query, setQuery] = useState('')
  const [confirmDelete, setConfirmDelete] = useState<GalleryCompanion | null>(null)
  const [renameTarget, setRenameTarget] = useState<GalleryCompanion | null>(null)
  const [renameValue, setRenameValue] = useState('')
  const scale = companionInfo.scale ?? COMPANION_SCALE_DEFAULT

  useEffect(() => {
    if (gatewayState !== 'open') {
      return
    }

    void loadCompanionGallery(requestGateway)
  }, [gatewayState, requestGateway])

  const enabled = gallery?.enabled ?? false
  const active = gallery?.active ?? ''
  const companions = gallery?.companions ?? []
  const staleBackend = status === 'stale'

  const selectCompanion = (slug: string) => {
    void adoptCompanion(requestGateway, slug, copy.adoptFailed(slug)).then(ok => ok && triggerHaptic('crisp'))
  }

  const removeCompanion = (slug: string) => {
    void removeCompanionAction(requestGateway, slug, copy.uninstallFailed(slug)).then(ok => ok && triggerHaptic('crisp'))
  }

  const exportCompanion = (slug: string) => {
    void exportCompanionAction(requestGateway, slug, copy.exportFailed(slug)).then(ok => ok && triggerHaptic('crisp'))
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
    void renameCompanionAction(requestGateway, slug, renameValue, copy.renameFailed(slug))
  }

  const toggle = (on: boolean) => {
    void setCompanionEnabled(requestGateway, on, {
      noneAvailable: copy.noneAvailable,
      fallback: on ? copy.turnOnFailed : copy.turnOffFailed
    }).then(ok => ok && triggerHaptic('crisp'))
  }

  // The companiondex catalog is thousands of entries, so rank + cap how many render.
  const RENDER_CAP = 60
  const sorted = rankedGalleryCompanions(gallery, query)
  const shown = sorted.slice(0, RENDER_CAP)

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
              <div className="mt-3 h-72 overflow-y-auto pr-1">
                {status === 'loading' && companions.length === 0 ? (
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
                ) : companions.length === 0 ? (
                  <p className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                    {copy.unreachable}
                  </p>
                ) : shown.length === 0 ? (
                  <p className="wrap-anywhere text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                    {copy.noMatch(query)}
                  </p>
                ) : (
                  <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
                    {shown.map(companion => {
                      const isActive = enabled && active === companion.slug
                      const isBusy = busySlug === companion.slug

                      return (
                        <div className="group relative" key={companion.slug}>
                          <button
                            className={cn(
                              'flex w-full items-center gap-2.5 px-2.5 py-2 text-left disabled:opacity-50',
                              selectableCardClass({ active: isActive, prominent: companion.installed })
                            )}
                            disabled={isBusy}
                            onClick={() => void selectCompanion(companion.slug)}
                            type="button"
                          >
                            <CompanionThumb
                              alt={companion.displayName}
                              load={(slug, url) => loadCompanionThumb(requestGateway, slug, url)}
                              slug={companion.slug}
                              url={companion.spritesheetUrl}
                            />
                            <span className="min-w-0 flex-1">
                              <span className="flex items-center gap-1.5">
                                <span className="truncate text-[length:var(--conversation-text-font-size)] font-medium">
                                  {companion.displayName}
                                </span>
                                {companion.generated && (
                                  <span className="shrink-0 rounded-full bg-primary/15 px-1.5 py-px text-[0.625rem] font-medium text-primary">
                                    {copy.generatedTag}
                                  </span>
                                )}
                              </span>
                              <span className="block truncate text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                                {companion.slug}
                                {companion.installed ? ` · ${copy.installedTag}` : ''}
                              </span>
                            </span>
                            {isBusy && <Loader2 className="size-4 shrink-0 animate-spin text-(--ui-text-tertiary)" />}
                          </button>
                          {!isBusy && (companion.installed || companion.generated) && (
                            <div className="absolute right-1.5 top-1.5 flex gap-1 opacity-0 transition focus-within:opacity-100 group-hover:opacity-100">
                              {companion.generated && (
                                <CompanionAction
                                  icon={<Pencil className="size-3.5" />}
                                  label={copy.rename(companion.displayName)}
                                  onClick={() => {
                                    setRenameValue(companion.displayName)
                                    setRenameTarget(companion)
                                  }}
                                />
                              )}
                              {companion.generated && (
                                <CompanionAction
                                  icon={<Download className="size-3.5" />}
                                  label={copy.exportCompanion(companion.displayName)}
                                  onClick={() => exportCompanion(companion.slug)}
                                />
                              )}
                              {companion.installed && (
                                // Generated companions have no remote source — deletion is
                                // permanent, so confirm; companiondex companions just uninstall.
                                <CompanionAction
                                  danger
                                  icon={<Trash2 className="size-3.5" />}
                                  label={companion.generated ? copy.delete(companion.displayName) : copy.uninstall(companion.displayName)}
                                  onClick={() => (companion.generated ? setConfirmDelete(companion) : removeCompanion(companion.slug))}
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
                  max={COMPANION_SCALE_MAX}
                  min={COMPANION_SCALE_MIN}
                  onChange={event => {
                    triggerHaptic('selection')
                    setCompanionScale(requestGateway, Number(event.target.value))
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
          <ToggleRow checked={roam} description={copy.roamDesc} label={copy.roamTitle} onChange={setCompanionRoam} />
        )}
      </div>

      <ConfirmDialog
        confirmLabel={copy.deleteConfirm}
        description={copy.deleteBody}
        destructive
        onClose={() => setConfirmDelete(null)}
        onConfirm={async () => {
          if (confirmDelete) {
            const ok = await removeCompanionAction(
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

/** A single hover-revealed icon action on a companion card (rename / export / delete). */
function CompanionAction({
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
