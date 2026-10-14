import type { ModelOptionsResult } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import type { ReactElement } from 'react'

import type { MenuKit } from '@/components/ui/actions-menu'
import { Codicon } from '@/components/ui/codicon'
import { Skeleton } from '@/components/ui/skeleton'
import { useI18n } from '@/i18n'
import { modelOptionsQueryKey, requestModelOptions } from '@/lib/model-options'
import { cn } from '@/lib/utils'
import { $activeGatewayProfile } from '@/store/profile'
import { $projects, setProjectModelRoute } from '@/store/projects'

// A project's default model route is a stored pin, not a live session model, so
// this lists the authenticated catalog flat — no reasoning / fast submenus, no
// favorites, and no "switch this chat" side effects.
const MAX_MODELS_PER_PROVIDER = 50

/** Key matching the pinned route — provider + model, since a bare model id can
 *  live on more than one provider. */
const routeKey = (provider: null | string, model: null | string): string =>
  model ? `${provider ?? ''}\u0000${model}` : ''

interface ProjectModelPickerProps {
  kit: MenuKit
  /** The explicit project whose default route this edits. */
  projectId: string
}

/** "Default model" submenu body. Mounted only while the submenu is open (Radix
 *  portals the content), so the catalog is fetched on demand rather than for
 *  every project row that renders a menu. */
export function ProjectModelPicker({ kit, projectId }: ProjectModelPickerProps): ReactElement {
  const { t } = useI18n()
  const copy = t.sidebar.projects
  const profile = useStore($activeGatewayProfile)
  const projects = useStore($projects)

  const current = projects.find(proj => proj.id === projectId)
  const active = routeKey(current?.provider ?? null, current?.model ?? null)
  const hasPin = Boolean(current?.model)

  const catalog = useQuery<ModelOptionsResult>({
    enabled: Boolean(profile),
    queryFn: () => requestModelOptions({ profile }),
    queryKey: modelOptionsQueryKey(profile),
    staleTime: 60_000
  })

  const providers = (catalog.data?.providers ?? []).filter(group => (group.models?.length ?? 0) > 0)

  return (
    <div className="flex w-64 flex-col">
      <kit.Item onSelect={() => void setProjectModelRoute(projectId, null)}>
        {hasPin ? null : <Codicon className="text-(--ui-accent)" name="check" size="0.875rem" />}
        <span className={cn(hasPin && 'pl-4')}>{copy.modelProfileDefault}</span>
      </kit.Item>
      <kit.Separator />
      <div className="px-2 py-1 text-(--ui-text-tertiary) text-xs">{copy.modelHint}</div>
      <div className="max-h-72 overflow-y-auto">
        {catalog.isPending ? (
          <div className="flex flex-col gap-1 p-2">
            <Skeleton className="h-5 w-full" />
            <Skeleton className="h-5 w-4/5" />
            <Skeleton className="h-5 w-3/5" />
          </div>
        ) : providers.length === 0 ? (
          <div className="p-2 text-(--ui-text-tertiary) text-xs">{copy.modelLoadFailed}</div>
        ) : (
          providers.map(group => (
            <div key={group.slug}>
              <kit.Label className="px-2 py-1 text-(--ui-text-tertiary) text-xs">{group.name}</kit.Label>
              {(group.models ?? []).slice(0, MAX_MODELS_PER_PROVIDER).map(id => {
                const isActive = routeKey(group.slug, id) === active

                return (
                  <kit.Item
                    key={routeKey(group.slug, id)}
                    onSelect={() => void setProjectModelRoute(projectId, { model: id, provider: group.slug })}
                  >
                    {isActive ? <Codicon className="text-(--ui-accent)" name="check" size="0.875rem" /> : null}
                    <span className={cn('truncate font-mono text-xs', !isActive && 'pl-4')}>{id}</span>
                  </kit.Item>
                )
              })}
            </div>
          ))
        )}
      </div>
    </div>
  )
}

/** The "Default model" submenu, shared by the kebab dropdown and the right-click
 *  menu. Explicit projects only: an auto-discovered repo has no row to pin on. */
export function ProjectModelSubmenu({ kit, projectId }: ProjectModelPickerProps): ReactElement {
  const { t } = useI18n()

  return (
    <kit.Sub>
      <kit.SubTrigger>
        <Codicon name="symbol-method" size="0.875rem" />
        <span>{t.sidebar.projects.menuDefaultModel}</span>
      </kit.SubTrigger>
      <kit.SubContent className="p-1">
        <ProjectModelPicker kit={kit} projectId={projectId} />
      </kit.SubContent>
    </kit.Sub>
  )
}
