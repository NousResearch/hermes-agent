import { useStore } from '@nanostores/react'

import { SIDEBAR_NAV_AREA, type SidebarNavContribution } from '@/app/routes'
import { Checkbox } from '@/components/ui/checkbox'
import { useContributions } from '@/contrib/react/use-contributions'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { $sidebarNavHidden, setSidebarNavHidden, SIDEBAR_NAV_IDS } from '@/store/sidebar-nav'

import { ListRow } from './primitives'

// Every built-in row is listed even when the current Interface mode rests it:
// hiding is the user's choice, independent of mode (#119965), so a hide made
// while a row is mode-shadowed must persist for when the mode shows it again.
// The ids are the store's canonical core list — the same list the rendered
// sidebar is pinned to by test — so Settings can never drift from the sidebar.

/**
 * Per-row sidebar visibility: one checkbox per nav row (built-ins first, then
 * plugin-contributed rows), checked = shown. Writes straight through to
 * `$sidebarNavHidden`; the sidebar render filters from the same atom.
 */
export function SidebarNavRowsSetting() {
  const { t } = useI18n()
  const copy = t.settings.appearance
  const hidden = new Set(useStore($sidebarNavHidden))

  // Same area AND the same validity rule the sidebar itself applies, so this
  // list never offers a toggle for a contribution that could not render.
  // Contributed ids stay namespaced (`${pluginId}:${id}`) — the namespace the
  // store and the sidebar's filter already speak.
  const navContributions = useContributions(SIDEBAR_NAV_AREA)

  const rows = [
    ...SIDEBAR_NAV_IDS.map(id => ({ id: id as string, label: t.sidebar.nav[id] })),
    ...navContributions.flatMap(contribution => {
      const data = contribution.data as Partial<SidebarNavContribution> | undefined

      if (!data?.path?.startsWith('/') || !data.label) {
        return []
      }

      return [{ id: contribution.id, label: data.label }]
    })
  ]

  return (
    <ListRow
      below={
        <div className="mt-3 grid gap-2">
          {rows.map(row => (
            // Label wraps the control (custom-endpoints precedent) for the
            // click target; the explicit aria-label is what names the Radix
            // button-role checkbox deterministically (ToggleRow's approach).
            <label
              className="flex cursor-pointer items-center gap-2 text-[length:var(--conversation-text-font-size)] text-foreground"
              key={row.id}
            >
              <Checkbox
                aria-label={row.label}
                checked={!hidden.has(row.id)}
                onCheckedChange={checked => {
                  triggerHaptic('selection')
                  // Honor Radix's explicit target state instead of blind-toggling:
                  // an assistive-tech event already carries the intended state, and
                  // setting it keeps repeated or out-of-order events idempotent.
                  const hiddenIds = $sidebarNavHidden.get()

                  setSidebarNavHidden(
                    checked === true ? hiddenIds.filter(id => id !== row.id) : [...hiddenIds, row.id]
                  )
                }}
              />
              {row.label}
            </label>
          ))}
        </div>
      }
      description={copy.sidebarNavDesc}
      title={copy.sidebarNavTitle}
    />
  )
}
