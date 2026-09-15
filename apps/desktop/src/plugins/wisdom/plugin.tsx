/**
 * Collective Wisdom — "Team Skills" page. Pure SDK consumer over the plugin's own
 * `/api/plugins/wisdom/*` router (`plugins/wisdom/dashboard/plugin_api.py`): browse the
 * team catalog, see installed versions + pending updates, install/update/remove.
 *
 * Consent is two-step by construction: `/plan` returns the exact version, content hash and
 * Gateway security verdict; the ConfirmDialog shows them; `/install` echoes the hash back and
 * the server refuses if the package changed in between. Ships ON: it renders nothing but an
 * explanatory empty state unless the Nous token carries Wisdom scopes.
 */

import {
  Badge,
  Button,
  Codicon,
  ConfirmDialog,
  EmptyState,
  type HermesPlugin,
  host,
  type PluginContext,
  type RouteContribution,
  ROUTES_AREA,
  SIDEBAR_NAV_AREA,
  type SidebarNavContribution,
  STATUSBAR_AREAS,
  Tip,
  useMutation,
  useQuery,
  useQueryClient
} from '@hermes/plugin-sdk'
import { useState } from 'react'

interface Skill { id: string; slug: string | null; version: number | null; installs: number; description: string | null; security: string | null }
interface Installed { slug: string; version: number; path: string }
interface Update { skill_id: string; slug: string; installed: number; latest: number; required: boolean }
interface Notice { skill_id: string; version: number | null; kind: string; installed: number | null }
interface Overview {
  entitled: boolean
  skills: Skill[]
  status: { installed?: Record<string, Installed>; updates?: Update[]; notices?: Notice[] }
}
interface Plan { skill_id: string; slug: string; version: number; content_hash: string; security: string; author: string | null; explanation: string | null; target: string }

let rest: PluginContext['rest'] | null = null
const KEY = ['wisdom', 'overview'] as const
const fetchOverview = () => rest!<Overview>('/overview')

function PlanDialog({ plan, onClose }: { plan: Plan; onClose: () => void }) {
  const qc = useQueryClient()

  return (
    <ConfirmDialog
      confirmLabel="Install"
      description={
        <span className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
          <span className="text-muted-foreground">security</span><span>{plan.security}</span>
          <span className="text-muted-foreground">hash</span><span className="font-mono break-all">{plan.content_hash}</span>
          {plan.author && (<><span className="text-muted-foreground">publisher</span><span>{plan.author}</span></>)}
          <span className="text-muted-foreground">target</span><span className="font-mono break-all">{plan.target}</span>
        </span>
      }
      onClose={onClose}
      onConfirm={async () => {
        await rest!('/install', { method: 'POST', body: { skill_id: plan.skill_id, version: plan.version, content_hash: plan.content_hash } })
        await qc.invalidateQueries({ queryKey: KEY })
        host.notify({ kind: 'info', message: `Installed ${plan.slug} v${plan.version}` })
      }}
      open
      title={`Install ${plan.slug} v${plan.version}`}
    />
  )
}

function TeamSkillsPage() {
  const qc = useQueryClient()
  const { data, error, isLoading } = useQuery({ queryKey: KEY, queryFn: fetchOverview, refetchInterval: 120_000 })
  const [plan, setPlan] = useState<Plan | null>(null)

  const planFor = useMutation({
    mutationFn: (body: { skill_id: string; version?: number }) => rest!<Plan>('/plan', { method: 'POST', body }),
    onSuccess: setPlan,
    onError: e => host.notifyError(e, 'Could not prepare install')
  })

  const remove = useMutation({
    mutationFn: (skill_id: string) => rest!('/uninstall', { method: 'POST', body: { skill_id } }),
    onSuccess: () => void qc.invalidateQueries({ queryKey: KEY }),
    onError: e => host.notifyError(e, 'Could not remove skill')
  })

  if (isLoading) {return <div className="p-6 text-sm text-muted-foreground">Loading team skills…</div>}

  if (error) {return <EmptyState description={String((error as Error).message)} title="Collective Wisdom unavailable" />}

  if (!data?.entitled) {
    return <EmptyState description="Sign in to Nous with a team that has Collective Wisdom enabled (hermes login) to browse and share skills." title="Collective Wisdom" />
  }

  const installed = data.status.installed ?? {}
  const updates = new Map((data.status.updates ?? []).map(u => [u.skill_id, u]))

  return (
    <div className="flex h-full min-h-0 flex-col gap-4 overflow-y-auto p-6">
      <header className="flex items-baseline justify-between">
        <h1 className="text-lg font-semibold">Team Skills</h1>
        <span className="text-xs text-muted-foreground">{data.skills.length} shared · {Object.keys(installed).length} installed</span>
      </header>
      {data.skills.length === 0 && <EmptyState description="Share a skill with `hermes wisdom share <name> --description …`." title="Nothing shared yet" />}
      <ul className="flex flex-col divide-y divide-(--ui-stroke-secondary)">
        {data.skills.map(s => {
          const local = installed[s.id]
          const upd = updates.get(s.id)

          return (
            <li className="flex items-center gap-3 py-3" key={s.id}>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="truncate font-medium">{s.slug ?? s.id}</span>
                  <Badge variant="outline">v{s.version}</Badge>
                  {s.security && <Badge variant={s.security === 'pass' ? 'success' : 'destructive'}>{s.security}</Badge>}
                  {local && <Badge variant={upd ? 'default' : 'muted'}>{upd ? `update v${local.version} → v${upd.latest}` : `installed v${local.version}`}</Badge>}
                </div>
                {s.description && <p className="truncate text-xs text-muted-foreground">{s.description}</p>}
              </div>
              <span className="text-xs tabular-nums text-muted-foreground">{s.installs} installs</span>
              <span className="grid w-32 grid-cols-[1fr_2rem] items-center justify-items-end gap-1">
              {(!local || upd) && (
                <Button disabled={planFor.isPending} onClick={() => planFor.mutate({ skill_id: s.id })} size="sm">
                  {upd ? 'Update' : 'Install'}
                </Button>
              )}
              {local && !upd && <span />}
              {local ? (
                <Tip label="Remove from this profile">
                  <Button disabled={remove.isPending} onClick={() => remove.mutate(s.id)} size="sm" variant="ghost">
                    <Codicon name="trash" size="0.8rem" />
                  </Button>
                </Tip>
              ) : <span />}
              </span>
            </li>
          )
        })}
      </ul>
      {plan && <PlanDialog onClose={() => setPlan(null)} plan={plan} />}
    </div>
  )
}

/** Sidebar-adjacent pill: pending team updates/notices, one glance from anywhere. */
function WisdomCount() {
  const { data } = useQuery({ queryKey: KEY, queryFn: fetchOverview, refetchInterval: 300_000 })
  const n = (data?.status.updates?.length ?? 0) + (data?.status.notices?.length ?? 0)

  if (!data?.entitled || n === 0) {return null}

  return (
    <Tip label={`${n} team skill update${n === 1 ? '' : 's'} waiting`}>
      <button
        className="inline-flex h-full items-center gap-1 px-1.5 text-[0.6875rem] tabular-nums text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground"
        onClick={() => host.navigate('/wisdom')}
        type="button"
      >
        <Codicon name="organization" size="0.7rem" />
        <span>{n}</span>
      </button>
    </Tip>
  )
}

const plugin: HermesPlugin = {
  id: 'wisdom',
  name: 'Collective Wisdom',
  description: 'Team Skills page: browse, install and update skills your Nous team shares; status-bar count of pending updates.',
  defaultEnabled: true,
  register(ctx) {
    rest = ctx.rest
    ctx.onDispose(() => { rest = null })
    ctx.registerMany([
      { id: 'page', area: ROUTES_AREA, data: { path: '/wisdom' } satisfies RouteContribution, render: () => <TeamSkillsPage /> },
      { id: 'nav', area: SIDEBAR_NAV_AREA, order: 55, data: { codicon: 'organization', label: 'Team Skills', path: '/wisdom' } satisfies SidebarNavContribution },
      { id: 'count', area: STATUSBAR_AREAS.right, order: 81, render: () => <WisdomCount /> }
    ])
  }
}

export default plugin
