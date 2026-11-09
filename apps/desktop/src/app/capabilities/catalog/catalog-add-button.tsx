import { Button } from '@/components/ui/button'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { Check, Loader2, Plus } from '@/lib/icons'

interface CatalogAddButtonProps {
  name: string
  installed: boolean
  installing: boolean
  disabled: boolean
  onInstall: () => void
  /** Hero cards and the detail have room: "+ Install" instead of a bare +. */
  labeled?: boolean
}

/** Install is one-way, so it is a button, not a switch: + installs, then the
 *  owner's on/off switch and ⋯ menu take the slot. The check only shows as a
 *  fallback for installed entries whose owner renders no switch. */
export function CatalogAddButton({ name, installed, installing, disabled, onInstall, labeled }: CatalogAddButtonProps) {
  const { t } = useI18n()

  // The labeled form says "Install" on its face, so its accessible name and
  // tooltip-free look match; the bare + keeps the "Add <name>" tooltip.
  const label = labeled
    ? `${installed ? t.catalog.installed : t.catalog.install} ${name}`
    : `${installed ? t.catalog.added : t.catalog.add} ${name}`

  const button = (
    <Button
      aria-busy={installing}
      aria-label={label}
      disabled={disabled || installing || installed}
      onClick={onInstall}
      size={labeled ? 'xs' : 'icon-xs'}
      variant={labeled ? 'outline' : 'ghost'}
    >
      {installing ? <Loader2 className="animate-spin text-(--ui-text-tertiary)" /> : installed ? <Check /> : <Plus />}
      {labeled && (installing ? t.catalog.installing : installed ? t.catalog.installed : t.catalog.install)}
    </Button>
  )

  return labeled ? button : <Tip label={label}>{button}</Tip>
}
