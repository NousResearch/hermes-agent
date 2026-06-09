import type * as React from 'react'
import { useCallback, useEffect, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import { createWebhook, deleteWebhook, getWebhooks, setWebhookEnabled } from '@/hermes'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import type { WebhookRoute, WebhooksResponse } from '@/types/hermes'

import { useRefreshHotkey } from '../hooks/use-refresh-hotkey'
import { PAGE_INSET_X } from '../layout-constants'
import { PageSearchShell } from '../page-search-shell'
import { includesQuery } from '../settings/helpers'
import type { SetStatusbarItemGroup } from '../shell/statusbar-controls'

interface WebhooksViewProps extends React.ComponentProps<'section'> {
  setStatusbarItemGroup?: SetStatusbarItemGroup
}

function asList(value: string): string[] {
  return value
    .split(',')
    .map(item => item.trim())
    .filter(Boolean)
}

function CopyButton({ label, value }: { label: string; value: string }) {
  const [copied, setCopied] = useState(false)

  const onCopy = useCallback(() => {
    void navigator.clipboard.writeText(value).then(() => {
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1200)
    })
  }, [value])

  return (
    <Button disabled={!value} onClick={onCopy} size="sm" variant="secondary">
      <Codicon name={copied ? 'check' : 'copy'} size="0.85rem" />
      {copied ? 'Copied' : label}
    </Button>
  )
}

function routeMatches(route: WebhookRoute, query: string) {
  const q = query.trim().toLowerCase()

  if (!q) {
    return true
  }

  return (
    includesQuery(route.name, q) ||
    includesQuery(route.description, q) ||
    includesQuery(route.prompt, q) ||
    route.skills.some(skill => includesQuery(skill, q)) ||
    route.events.some(event => includesQuery(event, q))
  )
}

export function WebhooksView({ className, setStatusbarItemGroup: _setStatusbarItemGroup, ...props }: WebhooksViewProps) {
  const [data, setData] = useState<WebhooksResponse | null>(null)
  const [query, setQuery] = useState('')
  const [refreshing, setRefreshing] = useState(false)
  const [saving, setSaving] = useState<string | null>(null)
  const [createdSecret, setCreatedSecret] = useState<{ name: string; secret: string; url: string } | null>(null)

  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [events, setEvents] = useState('')
  const [prompt, setPrompt] = useState('')
  const [skills, setSkills] = useState('')

  const refreshWebhooks = useCallback(async () => {
    setRefreshing(true)

    try {
      setData(await getWebhooks())
    } catch (err) {
      notifyError(err, 'Failed to load webhooks')
    } finally {
      setRefreshing(false)
    }
  }, [])

  useRefreshHotkey(refreshWebhooks)

  useEffect(() => {
    void refreshWebhooks()
  }, [refreshWebhooks])

  const resetForm = useCallback(() => {
    setName('')
    setDescription('')
    setEvents('')
    setPrompt('')
    setSkills('')
  }, [])

  const handleCreate = useCallback(async () => {
    const endpointName = name.trim()

    if (!endpointName) {
      notifyError(new Error('Endpoint name is required'), 'Missing name')

      return
    }

    setSaving('create')

    try {
      const created = await createWebhook({
        deliver: 'log',
        description: description.trim() || undefined,
        events: asList(events),
        name: endpointName,
        prompt: prompt.trim() || undefined,
        skills: asList(skills)
      })

      setCreatedSecret({ name: created.name, secret: created.secret, url: created.url })
      notify({ message: created.url, title: 'Skill API endpoint created' })
      resetForm()
      await refreshWebhooks()
    } catch (err) {
      notifyError(err, 'Failed to create endpoint')
    } finally {
      setSaving(null)
    }
  }, [description, events, name, prompt, refreshWebhooks, resetForm, skills])

  const handleToggle = useCallback(
    async (route: WebhookRoute, enabled: boolean) => {
      setSaving(route.name)

      try {
        await setWebhookEnabled(route.name, enabled)
        await refreshWebhooks()
      } catch (err) {
        notifyError(err, `Failed to update ${route.name}`)
      } finally {
        setSaving(null)
      }
    },
    [refreshWebhooks]
  )

  const handleDelete = useCallback(
    async (route: WebhookRoute) => {
      if (!window.confirm(`Delete endpoint "${route.name}"?`)) {
        return
      }

      setSaving(route.name)

      try {
        await deleteWebhook(route.name)
        await refreshWebhooks()
        notify({ message: route.name, title: 'Endpoint deleted' })
      } catch (err) {
        notifyError(err, `Failed to delete ${route.name}`)
      } finally {
        setSaving(null)
      }
    },
    [refreshWebhooks]
  )

  if (!data) {
    return <PageLoader label="Loading skill API endpoints…" />
  }

  const visibleRoutes = data.subscriptions.filter(route => routeMatches(route, query))

  return (
    <section className={cn('flex h-full min-h-0 flex-col bg-(--ui-chat-surface-background)', className)} {...props}>
      <PageSearchShell
        onSearchChange={setQuery}
        searchPlaceholder="Search endpoints, skills, prompts…"
        searchTrailingAction={
          <Button disabled={refreshing} onClick={() => void refreshWebhooks()} size="sm" variant="secondary">
            <Codicon className={cn(refreshing && 'animate-spin')} name="refresh" size="0.85rem" />
            Refresh
          </Button>
        }
        searchValue={query}
      >
        <div className={cn('flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto py-4', PAGE_INSET_X)}>
        <div className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-bg-secondary)/70 p-4">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h2 className="text-sm font-semibold text-(--ui-text-primary)">Create endpoint</h2>
              <p className="mt-1 max-w-2xl text-xs text-(--ui-text-tertiary)">
                Create a local webhook endpoint that runs Hermes with a prompt and optional skills. Use it from scripts,
                websites, automations, or other apps.
              </p>
            </div>
            <Badge variant={data.enabled ? 'default' : 'muted'}>{data.enabled ? 'Webhook platform on' : 'Webhook platform off'}</Badge>
          </div>

          {!data.enabled && (
            <div className="mt-3 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-300">
              Enable the Webhook messaging platform before creating endpoints.
            </div>
          )}

          <div className="mt-4 grid gap-3 lg:grid-cols-2">
            <Input onChange={event => setName(event.target.value)} placeholder="Endpoint name, e.g. travel-analyzer" value={name} />
            <Input onChange={event => setDescription(event.target.value)} placeholder="Description" value={description} />
            <Input onChange={event => setEvents(event.target.value)} placeholder="Events, comma-separated (optional)" value={events} />
            <Input onChange={event => setSkills(event.target.value)} placeholder="Skills, comma-separated (optional)" value={skills} />
            <Input className="lg:col-span-2" onChange={event => setPrompt(event.target.value)} placeholder="Prompt/instructions for this endpoint" value={prompt} />
          </div>

          <div className="mt-4 flex flex-wrap items-center gap-2">
            <Button disabled={!data.enabled || saving === 'create'} onClick={() => void handleCreate()} size="sm">
              <Codicon name="add" size="0.85rem" />
              Create endpoint
            </Button>
            <span className="text-xs text-(--ui-text-tertiary)">Base URL: {data.base_url || 'not configured'}</span>
          </div>

          {createdSecret && (
            <div className="mt-4 rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-3 text-xs text-emerald-200">
              <div className="font-semibold">Secret for {createdSecret.name}</div>
              <div className="mt-2 flex flex-wrap items-center gap-2">
                <code className="rounded bg-black/30 px-2 py-1">{createdSecret.url}</code>
                <CopyButton label="Copy URL" value={createdSecret.url} />
                <CopyButton label="Copy Secret" value={createdSecret.secret} />
              </div>
              <p className="mt-2 text-emerald-100/80">Save this secret now. It is only shown once.</p>
            </div>
          )}
        </div>

        {visibleRoutes.length === 0 ? (
          <div className="rounded-xl border border-dashed border-(--ui-stroke-tertiary) bg-(--ui-bg-secondary)/40 p-8 text-center">
            <h3 className="text-sm font-semibold text-(--ui-text-primary)">No endpoints found</h3>
            <p className="mt-1 text-xs text-(--ui-text-tertiary)">
              Create one above to expose a Hermes prompt and skill set as a callable webhook endpoint.
            </p>
          </div>
        ) : (
          <div className="grid gap-3">
            {visibleRoutes.map(route => (
              <article className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-bg-secondary)/60 p-4" key={route.name}>
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <h3 className="truncate text-sm font-semibold text-(--ui-text-primary)">{route.name}</h3>
                      <Badge variant={route.enabled ? 'default' : 'muted'}>{route.enabled ? 'Enabled' : 'Disabled'}</Badge>
                      {route.deliver_only && <Badge variant="muted">Deliver only</Badge>}
                    </div>
                    <p className="mt-1 text-xs text-(--ui-text-tertiary)">{route.description || 'No description'}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <Switch checked={route.enabled} disabled={saving === route.name} onCheckedChange={checked => void handleToggle(route, checked)} />
                    <Button disabled={saving === route.name} onClick={() => void handleDelete(route)} size="sm" variant="destructive">
                      <Codicon name="trash" size="0.85rem" />
                      Delete
                    </Button>
                  </div>
                </div>

                <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
                  <code className="min-w-0 max-w-full truncate rounded border border-(--ui-stroke-quaternary) bg-(--ui-bg-primary)/50 px-2 py-1 text-(--ui-text-secondary)">
                    {route.url}
                  </code>
                  <CopyButton label="Copy URL" value={route.url} />
                </div>

                {route.prompt && <p className="mt-3 rounded-lg bg-(--ui-bg-primary)/40 p-3 text-xs text-(--ui-text-secondary)">{route.prompt}</p>}

                <div className="mt-3 flex flex-wrap gap-2">
                  {route.skills.map(skill => (
                    <Badge key={skill} variant="muted">{skill}</Badge>
                  ))}
                  {route.events.map(event => (
                    <Badge key={event} variant="outline">{event}</Badge>
                  ))}
                  {route.secret_set && <Badge variant="outline">Secret set</Badge>}
                </div>
              </article>
            ))}
          </div>
        )}
        </div>
      </PageSearchShell>
    </section>
  )
}
