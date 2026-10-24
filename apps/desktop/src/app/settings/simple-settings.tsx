import { useStore } from '@nanostores/react'
import { lazy, Suspense } from 'react'

import { PageLoader } from '@/components/page-loader'
import { ActionsMenu, renderActionItem } from '@/components/ui/actions-menu'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { Download, RefreshCw, Upload } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $activeConnectionId } from '@/store/connections'
import { $settingsScopeProfile } from '@/store/settings-scope'

import { PAGE_INSET_X } from '../layout-constants'
import { OverlayIconButton } from '../overlays/overlay-chrome'

import { SettingsProfileScope } from './profile-scope'

const CapabilitiesView = lazy(async () => ({ default: (await import('../capabilities')).CapabilitiesView }))

/** Capabilities as a Settings page, pinned to the profile and device Settings is editing. */
export function CapabilitiesSettings() {
  const { t } = useI18n()
  const scopeProfile = useStore($settingsScopeProfile)
  const activeConnectionId = useStore($activeConnectionId)

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <SettingsProfileScope className={cn('pb-3', PAGE_INSET_X)} />
      <Suspense fallback={<PageLoader label={t.skills.loading} />}>
        <CapabilitiesView
          embedded
          fixedConnection={activeConnectionId ?? undefined}
          fixedProfile={scopeProfile}
          key={`${activeConnectionId}:${scopeProfile}`}
        />
      </Suspense>
    </div>
  )
}

interface SettingsNavFooterProps {
  onExport: () => void
  onImport: () => void
  onReset: () => void
  simple: boolean
}

/** Config export, import and reset: icon buttons in Advanced, one labeled menu in Simple. */
export function SettingsNavFooter({ onExport, onImport, onReset, simple }: SettingsNavFooterProps) {
  const { t } = useI18n()

  if (simple) {
    return (
      <ActionsMenu
        ariaLabel={t.settings.page.actions}
        items={kit => [
          renderActionItem(kit, { label: t.settings.exportConfig, iconNode: <Download />, onSelect: onExport }),
          renderActionItem(kit, { label: t.settings.importConfig, iconNode: <Upload />, onSelect: onImport }),
          renderActionItem(kit, {
            label: t.settings.resetToDefaults,
            iconNode: <RefreshCw />,
            variant: 'destructive',
            onSelect: onReset
          })
        ]}
      >
        <Button aria-label={t.settings.page.actions} className="min-w-0 max-w-full" size="sm" variant="ghost">
          <Codicon name="ellipsis" />
          <span className="truncate">{t.settings.page.actions}</span>
        </Button>
      </ActionsMenu>
    )
  }

  return (
    <>
      <Tip label={t.settings.exportConfig}>
        <OverlayIconButton onClick={onExport}>
          <Download />
        </OverlayIconButton>
      </Tip>
      <Tip label={t.settings.importConfig}>
        <OverlayIconButton onClick={onImport}>
          <Upload />
        </OverlayIconButton>
      </Tip>
      <Tip label={t.settings.resetToDefaults}>
        <OverlayIconButton className="hover:text-destructive" onClick={onReset}>
          <RefreshCw />
        </OverlayIconButton>
      </Tip>
    </>
  )
}
