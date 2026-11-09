import type { ReactNode } from 'react'

import type { DelegationRouting } from '@/api/delegation'
import { Button } from '@/components/ui/button'
import type { ProfileScope } from '@/hermes'
import { useI18n } from '@/i18n'
import type { DelegationRoutingTranslations } from '@/i18n/types_delegation_routing'
import { Info } from '@/lib/icons'

import { chooseDelegationMode, routesOwnSubagentModel, useDelegationRouting } from './routing-data'

/** Why the subagent model does not apply in the saved mode, and what a task that picks no route runs on. */
function inactiveReason(routing: DelegationRouting, r: DelegationRoutingTranslations): string {
  const mode = routing.configured

  if (mode === 'model_map') {
    const route = routing.model_map.default

    return route ? r.inactiveModelMap(route) : r.inactiveModelMapNoDefault
  }

  if (mode === 'profile_routes') {
    const profile = routing.profile_routes.default

    return profile ? r.inactiveProfiles(profile) : r.inactiveProfilesNoDefault
  }

  return r.inactivePlugin(routing.backends.find(b => b.name === mode)?.label || mode)
}

/**
 * The subagent model/provider rows (`delegation.model` / `delegation.provider`) in any mode but Default:
 * shown, but visibly inert, with the one line that says why and the way back.
 */
export function SubagentModelGate({ scope, children }: { scope: ProfileScope; children: ReactNode }) {
  const { t } = useI18n()
  const r = t.settings.model.subagents.routing
  const { data: routing } = useDelegationRouting(scope)

  if (!routing || !routesOwnSubagentModel(routing)) {
    return children
  }

  const reason = inactiveReason(routing, r)

  return (
    <div data-slot="subagent-model-inactive">
      <div className="mt-2 flex items-start gap-2.5 rounded-md bg-(--ui-bg-quaternary) px-3 py-2.5" role="note">
        <Info className="mt-0.5 size-4 shrink-0 text-(--ui-text-tertiary)" />
        <div className="grid min-w-0 flex-1 gap-0.5">
          <span className="text-[length:var(--conversation-text-font-size)] font-medium text-foreground">
            {r.inactiveTitle}
          </span>
          <span className="text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
            {reason}
          </span>
        </div>
        <Button onClick={() => void chooseDelegationMode(scope, 'default', r.modeFailed)} size="xs" variant="secondary">
          {r.useDefault}
        </Button>
      </div>
      <div aria-disabled className="pointer-events-none opacity-45 select-none" inert>
        {children}
      </div>
    </div>
  )
}
