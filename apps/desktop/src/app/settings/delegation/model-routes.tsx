import { useState } from 'react'

import { ActionsMenu, renderActionItem } from '@/components/ui/actions-menu'
import { Button } from '@/components/ui/button'
import type { DelegationModelMap, DelegationModelRoute, ProfileScope } from '@/hermes'
import { useI18n } from '@/i18n'
import { Box, MoreHorizontal, Plus } from '@/lib/icons'
import { confirm } from '@/store/confirm'

import { Pill, SectionHeading } from '../primitives'

import { ModelRouteDialog } from './model-route-dialog'
import { removeRoute, setDefaultRoute, upsertRoute } from './routes'
import { saveModelRoutes } from './routing-data'

interface ModelRoutesEditorProps {
  scope: ProfileScope
  routes: DelegationModelMap
  providerNames: ReadonlyMap<string, string>
}

/** `delegation.backends.model_map`: the named routes the agent picks between, with create / edit / delete. */
export function ModelRoutesEditor({ scope, routes, providerNames }: ModelRoutesEditorProps) {
  const { t } = useI18n()
  const r = t.settings.model.subagents.routing
  // undefined = closed, null = creating, a route = editing it.
  const [editing, setEditing] = useState<DelegationModelRoute | null | undefined>(undefined)
  const save = (next: DelegationModelMap) => saveModelRoutes(scope, next, r.saveFailed)

  const askDelete = async (route: DelegationModelRoute) => {
    const next = removeRoute(routes, route.name)
    const movesDefault = routes.default === route.name && next.default

    const ok = await confirm({
      confirmLabel: r.deleteConfirmAction,
      description: movesDefault
        ? `${r.deleteConfirmBody} ${r.deleteConfirmNextDefault(next.default!)}`
        : r.deleteConfirmBody,
      destructive: true,
      title: r.deleteConfirmTitle(route.name)
    })

    if (ok) {
      await save(next)
    }
  }

  return (
    <section data-slot="delegation-model-routes">
      <SectionHeading
        aside={
          <Button onClick={() => setEditing(null)} size="sm" variant="secondary">
            <Plus />
            {r.addRoute}
          </Button>
        }
        icon={Box}
        title={r.modelRoutesTitle}
      />
      {routes.options.length === 0 ? (
        <p className="py-3 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
          {r.modelRoutesEmpty}
        </p>
      ) : (
        <ul className="grid">
          {routes.options.map(route => (
            <RouteRow
              isDefault={routes.default === route.name}
              key={route.name}
              onDelete={() => void askDelete(route)}
              onEdit={() => setEditing(route)}
              onMakeDefault={() => void save(setDefaultRoute(routes, route.name))}
              providerName={providerNames.get(route.provider ?? '') ?? route.provider}
              route={route}
            />
          ))}
        </ul>
      )}
      {editing !== undefined && (
        <ModelRouteDialog
          onClose={() => setEditing(undefined)}
          onSave={route => save(upsertRoute(routes, editing?.name ?? null, route))}
          route={editing}
          scope={scope}
          taken={routes.options.map(o => o.name).filter(name => name !== editing?.name)}
        />
      )}
    </section>
  )
}

interface RouteRowProps {
  route: DelegationModelRoute
  isDefault: boolean
  providerName?: string
  onEdit: () => void
  onMakeDefault: () => void
  onDelete: () => void
}

function RouteRow({ route, isDefault, providerName, onEdit, onMakeDefault, onDelete }: RouteRowProps) {
  const { t } = useI18n()
  const r = t.settings.model.subagents.routing

  const facts = [
    providerName || (route.base_url ? r.baseUrlRoute(route.base_url) : ''),
    route.model,
    route.reasoning_effort ? `${r.effortLabel.toLowerCase()} ${route.reasoning_effort}` : '',
    route.service_tier ?? ''
  ].filter(Boolean)

  return (
    <li className="group flex items-start gap-3 border-t border-(--ui-stroke-tertiary) py-3 first:border-t-0">
      <button className="grid min-w-0 flex-1 gap-1 text-left" onClick={onEdit} type="button">
        <span className="flex min-w-0 items-center gap-2">
          <span className="truncate font-mono text-[length:var(--conversation-text-font-size)] font-medium text-foreground">
            {route.name}
          </span>
          {isDefault && <Pill tone="primary">{r.defaultBadge}</Pill>}
        </span>
        <span className="text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-secondary)">
          {route.description}
        </span>
        <span className="truncate font-mono text-[0.68rem] text-(--ui-text-tertiary)">{facts.join(' · ')}</span>
      </button>
      <ActionsMenu
        ariaLabel={r.routeActions(route.name)}
        items={kit => (
          <>
            {renderActionItem(kit, { icon: 'edit', label: r.editRoute, onSelect: onEdit })}
            {!isDefault && renderActionItem(kit, { icon: 'star-empty', label: r.makeDefault, onSelect: onMakeDefault })}
            <kit.Separator />
            {renderActionItem(kit, { icon: 'trash', label: r.deleteRoute, onSelect: onDelete, variant: 'destructive' })}
          </>
        )}
      >
        <Button aria-label={r.routeActions(route.name)} size="icon-sm" variant="ghost">
          <MoreHorizontal />
        </Button>
      </ActionsMenu>
    </li>
  )
}
