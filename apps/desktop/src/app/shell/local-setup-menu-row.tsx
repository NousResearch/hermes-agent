import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { SETTINGS_ROUTE } from '@/app/routes'
import { Codicon } from '@/components/ui/codicon'
import { DropdownMenuItem, dropdownMenuRow, DropdownMenuSeparator } from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $localSetupRowFit, acceptLocalSetupOffer, refreshLocalSetupEligibility } from '@/store/local-setup-offer'
import { requestRoute } from '@/store/recovery-requests'

/**
 * "Run locally" at the top of the composer's model menu, for as long as this
 * machine qualifies and nothing is set up. The permanent home of the
 * local-setup offer: the card can be dismissed, this row stays until setup
 * completes. Every menu open re-reads eligibility (a user action, not a poll),
 * so finishing setup retires the row. A menu item, so the keyboard reaches it.
 */
export function LocalSetupMenuRow() {
  const fit = useStore($localSetupRowFit)
  const copy = useI18n().t.shell.modelMenu.localSetup

  useEffect(() => {
    void refreshLocalSetupEligibility()
  }, [])

  if (!fit) {
    return null
  }

  return (
    <>
      <DropdownMenuItem
        className={cn(dropdownMenuRow, 'items-start gap-2 py-1.5')}
        data-slot="model-menu-local-setup"
        onSelect={() => {
          acceptLocalSetupOffer()
          requestRoute(`${SETTINGS_ROUTE}?tab=providers&pview=local`)
        }}
      >
        <Codicon className="mt-0.5 shrink-0 text-(--ui-accent)" name="chip" size="0.8rem" />
        <span className="min-w-0 flex-1 leading-snug">
          <span className="block font-medium text-foreground">{copy.title}</span>
          <span className="block truncate text-(--ui-text-tertiary)">
            {copy.text(fit.model.display_name, fit.model.size_label)}
          </span>
        </span>
        <span className="shrink-0 self-center text-(--ui-accent)">{copy.action}</span>
      </DropdownMenuItem>
      <DropdownMenuSeparator className="mx-0" />
    </>
  )
}
