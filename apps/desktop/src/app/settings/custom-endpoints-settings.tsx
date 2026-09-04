import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import {
  activateCustomEndpoint,
  deleteCustomEndpoint,
  getCustomEndpoints,
  saveCustomEndpoint,
  validateCustomEndpoint
} from '@/hermes'
import { triggerHaptic } from '@/lib/haptics'
import { Check, Globe, Loader2, Plus, Save, Trash2, Zap } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { confirm } from '@/store/confirm'
import { notify, notifyError } from '@/store/notifications'
import type { CustomEndpoint, CustomEndpointUpdate } from '@/types/hermes'

import { EmptyState, Pill, SectionHeading, SettingsContent, SettingsSkeleton } from './primitives'

interface CustomEndpointsSettingsProps {
  onConfigSaved?: () => void
  onMainModelChanged?: (provider: string, model: string) => void
}

interface EndpointForm {
  apiKey: string
  baseUrl: string
  contextLength: string
  discoverModels: boolean
  id: string
  makeDefault: boolean
  model: string
  name: string
}

const EMPTY_FORM: EndpointForm = {
  apiKey: '',
  baseUrl: '',
  contextLength: '',
  discoverModels: true,
  id: '',
  makeDefault: true,
  model: '',
  name: ''
}

function formFromEndpoint(endpoint: CustomEndpoint): EndpointForm {
  return {
    apiKey: '',
    baseUrl: endpoint.base_url,
    contextLength: endpoint.context_length ? String(endpoint.context_length) : '',
    discoverModels: endpoint.discover_models,
    id: endpoint.id,
    makeDefault: Boolean(endpoint.is_current),
    model: endpoint.model,
    name: endpoint.name
  }
}

function toPayload(form: EndpointForm, models?: string[], replaceModels = false): CustomEndpointUpdate {
  const contextLength = Number.parseInt(form.contextLength, 10)

  return {
    id: form.id.trim() || undefined,
    name: form.name.trim(),
    base_url: form.baseUrl.trim(),
    model: form.model.trim(),
    api_key: form.apiKey.trim() || undefined,
    context_length: Number.isFinite(contextLength) && contextLength > 0 ? contextLength : undefined,
    discover_models: form.discoverModels,
    make_default: form.makeDefault,
    models: models?.length ? models : undefined,
    // Only Save asserts authority over the catalogue (#101764). Test must
    // stay additive — it posts the same payload shape but is only probing.
    replace_models: replaceModels || undefined
  }
}

export function CustomEndpointsSettings({ onConfigSaved, onMainModelChanged }: CustomEndpointsSettingsProps) {
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [activating, setActivating] = useState<string | null>(null)
  const [deleting, setDeleting] = useState<string | null>(null)
  const [endpoints, setEndpoints] = useState<CustomEndpoint[]>([])
  const [form, setForm] = useState<EndpointForm>(EMPTY_FORM)
  const [discoveredModels, setDiscoveredModels] = useState<string[]>([])
  // Which of `discoveredModels` the user wants persisted. An endpoint can
  // advertise 100+ models while a plan covers two, and Save used to store
  // every discovered id, flooding the model picker (#101764).
  const [selectedModels, setSelectedModels] = useState<string[]>([])
  const [modelFilter, setModelFilter] = useState('')

  async function refresh() {
    const data = await getCustomEndpoints()
    setEndpoints(data.endpoints)
  }

  useEffect(() => {
    let cancelled = false

    async function load() {
      try {
        const data = await getCustomEndpoints()

        if (cancelled) {
          return
        }

        setEndpoints(data.endpoints)
        const current = data.endpoints.find(endpoint => endpoint.is_current) ?? data.endpoints[0]

        if (current) {
          setForm(formFromEndpoint(current))
          setDiscoveredModels(current.models)
          setSelectedModels(current.models)
        }
      } catch (err) {
        notifyError(err, 'Could not load custom endpoints')
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    void load()

    return () => { cancelled = true }
  }, [])

  async function handleSave() {
    try {
      setSaving(true)
      // The default model must always survive, even if the user never ticked
      // it in the list (it may have been typed by hand).
      const models = Array.from(new Set([...selectedModels, form.model.trim()].filter(Boolean)))
      const response = await saveCustomEndpoint(toPayload(form, models, true))
      setEndpoints(response.endpoints)
      const saved = response.endpoints.find(endpoint => endpoint.id === response.id)

      if (saved) {
        setForm(formFromEndpoint(saved))
        setDiscoveredModels(saved.models)
        setSelectedModels(saved.models)
      }

      if (saved && saved.is_current) {
        onMainModelChanged?.(saved.id, saved.model)
      }

      triggerHaptic('success')
      onConfigSaved?.()
      notify({ kind: 'success', message: 'Custom endpoint saved.' })
    } catch (err) {
      notifyError(err, 'Save failed')
    } finally {
      setSaving(false)
    }
  }

  async function handleValidate() {
    try {
      setTesting(true)
      const response = await validateCustomEndpoint(toPayload(form))
      setDiscoveredModels(response.models)
      // Keep whatever the user had already chosen that this probe still
      // advertises; a re-Test must not silently re-tick 98 models they
      // deselected. A first probe (nothing chosen yet) selects everything,
      // preserving the pre-#101764 default of "save what you found".
      setSelectedModels(previous => {
        if (!previous.length) {
          return response.models
        }

        const stillOffered = previous.filter(model => response.models.includes(model))

        return stillOffered.length ? stillOffered : response.models
      })

      if (response.ok) {
        if (!form.model && response.models[0]) {
          setForm(current => ({ ...current, model: response.models[0] }))
        }

        notify({
          kind: 'success',
          message: response.models.length
            ? `Endpoint is reachable. Found ${response.models.length} models.`
            : 'Endpoint is reachable.'
        })
      } else {
        notify({
          kind: response.reachable ? 'warning' : 'error',
          message: response.message || 'Endpoint validation failed.'
        })
      }
    } catch (err) {
      notifyError(err, 'Validation failed')
    } finally {
      setTesting(false)
    }
  }

  async function handleActivate(endpoint: CustomEndpoint) {
    try {
      setActivating(endpoint.id)
      const response = await activateCustomEndpoint(endpoint.id)
      await refresh()
      onConfigSaved?.()
      onMainModelChanged?.(response.provider, response.model)
      triggerHaptic('success')
    } catch (err) {
      notifyError(err, 'Activation failed')
    } finally {
      setActivating(null)
    }
  }

  async function handleDelete(endpoint: CustomEndpoint) {
    // This panel is not internationalized at all — keep the literal it had.
    if (!(await confirm({ destructive: true, title: `Delete ${endpoint.name}?` }))) {
      return
    }

    try {
      setDeleting(endpoint.id)
      const response = await deleteCustomEndpoint(endpoint.id)
      setEndpoints(response.endpoints)

      if (form.id === endpoint.id) {
        setForm(EMPTY_FORM)
        setDiscoveredModels([])
        setSelectedModels([])
      }

      onConfigSaved?.()
      triggerHaptic('success')
    } catch (err) {
      notifyError(err, 'Delete failed')
    } finally {
      setDeleting(null)
    }
  }

  if (loading) {
    return <SettingsSkeleton sections={[{ heading: true, rows: 3 }]} />
  }

  const allModelOptions = Array.from(new Set([...discoveredModels, form.model].filter(Boolean)))
  const canSave = form.name.trim() && form.baseUrl.trim() && form.model.trim()
  const normalizedFilter = modelFilter.trim().toLowerCase()

  const visibleModels = normalizedFilter
    ? discoveredModels.filter(model => model.toLowerCase().includes(normalizedFilter))
    : discoveredModels

  function toggleModel(model: string, checked: boolean) {
    setSelectedModels(previous =>
      checked ? Array.from(new Set([...previous, model])) : previous.filter(entry => entry !== model)
    )
  }

  return (
    <SettingsContent>
      <div className="space-y-6">
        <section>
          <SectionHeading icon={Globe} meta={`${endpoints.length}`} title="Custom Endpoints" />
          <div className="divide-y divide-border/40 rounded-md border border-border/50">
            {endpoints.length ? (
              endpoints.map(endpoint => (
                <div className="grid gap-3 p-3 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-center" key={endpoint.id}>
                  <button
                    className="min-w-0 text-left"
                    onClick={() => {
                      setForm(formFromEndpoint(endpoint))
                      setDiscoveredModels(endpoint.models)
                      setSelectedModels(endpoint.models)
                      setModelFilter('')
                    }}
                    type="button"
                  >
                    <div className="flex min-w-0 items-center gap-2">
                      <span className="truncate text-sm font-medium">{endpoint.name}</span>
                      {endpoint.is_current && (
                        <Pill tone="primary">
                          <Check className="size-3" />
                          Active
                        </Pill>
                      )}
                      {endpoint.source === 'direct-config' && <Pill>config.yaml</Pill>}
                    </div>
                    <div className="mt-1 truncate font-mono text-[0.7rem] text-muted-foreground">
                      {endpoint.base_url}
                    </div>
                    <div className="mt-1 flex flex-wrap gap-2 text-xs text-muted-foreground">
                      <span>{endpoint.model}</span>
                      {/* Surface the stored catalogue size so a flooded save
                          is visible without opening the editor (#101764). */}
                      {endpoint.models.length > 1 && <span>{`${endpoint.models.length} models`}</span>}
                      {endpoint.has_api_key && <span>{endpoint.api_key_preview ?? 'API key set'}</span>}
                    </div>
                  </button>
                  <div className="flex items-center gap-2 sm:justify-end">
                    <Button
                      disabled={endpoint.is_current || activating === endpoint.id}
                      onClick={() => void handleActivate(endpoint)}
                      size="sm"
                      variant="outline"
                    >
                      {activating === endpoint.id ? <Loader2 className="animate-spin" /> : <Zap />}
                      Use
                    </Button>
                    {endpoint.source !== 'direct-config' && (
                      <Button
                        className="hover:text-destructive"
                        disabled={deleting === endpoint.id}
                        onClick={() => void handleDelete(endpoint)}
                        size="icon-sm"
                        title="Delete endpoint"
                        variant="ghost"
                      >
                        {deleting === endpoint.id ? <Loader2 className="animate-spin" /> : <Trash2 />}
                      </Button>
                    )}
                  </div>
                </div>
              ))
            ) : (
              <EmptyState description="Add an OpenAI-compatible endpoint below." title="No custom endpoints" />
            )}
          </div>
        </section>

        <section>
          <SectionHeading icon={Plus} title={form.id ? 'Edit Endpoint' : 'Add Endpoint'} />
          <div className="grid gap-3 rounded-md border border-border/50 p-3">
            <div className="grid gap-3 sm:grid-cols-2">
              <label className="grid gap-1.5 text-xs text-muted-foreground">
                Name
                <Input
                  onChange={event => setForm(current => ({ ...current, name: event.target.value }))}
                  placeholder="Axet Proxy"
                  value={form.name}
                />
              </label>
              <label className="grid gap-1.5 text-xs text-muted-foreground">
                Provider ID
                <Input
                  onChange={event => setForm(current => ({ ...current, id: event.target.value }))}
                  placeholder="axet-proxy"
                  value={form.id}
                />
              </label>
            </div>
            <label className="grid gap-1.5 text-xs text-muted-foreground">
              Endpoint URL
              <Input
                onChange={event => setForm(current => ({ ...current, baseUrl: event.target.value }))}
                placeholder="http://127.0.0.1:8081/v1"
                value={form.baseUrl}
              />
            </label>
            <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_12rem]">
              <label className="grid gap-1.5 text-xs text-muted-foreground">
                Default Model
                <Input
                  list="custom-endpoint-models"
                  onChange={event => setForm(current => ({ ...current, model: event.target.value }))}
                  placeholder="gpt-5.4"
                  value={form.model}
                />
                <datalist id="custom-endpoint-models">
                  {allModelOptions.map(model => (
                    <option key={model} value={model} />
                  ))}
                </datalist>
              </label>
              <label className="grid gap-1.5 text-xs text-muted-foreground">
                Context
                <Input
                  inputMode="numeric"
                  onChange={event => setForm(current => ({ ...current, contextLength: event.target.value }))}
                  placeholder="Auto"
                  value={form.contextLength}
                />
              </label>
            </div>
            <label className="grid gap-1.5 text-xs text-muted-foreground">
              API Key
              <Input
                onChange={event => setForm(current => ({ ...current, apiKey: event.target.value }))}
                placeholder={form.id ? 'Leave blank to keep current key' : 'Optional'}
                type="password"
                value={form.apiKey}
              />
            </label>
            <div className="flex flex-wrap items-center gap-4 text-xs text-muted-foreground">
              <label className="flex items-center gap-2">
                <Checkbox
                  checked={form.makeDefault}
                  onCheckedChange={checked => setForm(current => ({ ...current, makeDefault: checked === true }))}
                />
                Use for new chats
              </label>
              <label className="flex items-center gap-2">
                <Checkbox
                  checked={form.discoverModels}
                  onCheckedChange={checked => setForm(current => ({ ...current, discoverModels: checked === true }))}
                />
                Discover models
              </label>
            </div>
            {discoveredModels.length > 0 && (
              <div className="grid gap-2 rounded-md border border-border/50 p-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="text-xs text-muted-foreground">
                    {`Models to keep (${selectedModels.length}/${discoveredModels.length})`}
                  </span>
                  <div className="flex items-center gap-2">
                    <Button
                      onClick={() => setSelectedModels(discoveredModels)}
                      size="sm"
                      type="button"
                      variant="ghost"
                    >
                      Select all
                    </Button>
                    <Button onClick={() => setSelectedModels([])} size="sm" type="button" variant="ghost">
                      Select none
                    </Button>
                  </div>
                </div>
                {discoveredModels.length > 8 && (
                  <Input
                    onChange={event => setModelFilter(event.target.value)}
                    placeholder="Filter models"
                    value={modelFilter}
                  />
                )}
                <div className="max-h-56 overflow-y-auto">
                  {visibleModels.length ? (
                    visibleModels.map(model => (
                      <label className="flex items-center gap-2 py-1 text-xs" key={model}>
                        <Checkbox
                          checked={selectedModels.includes(model)}
                          onCheckedChange={checked => toggleModel(model, checked === true)}
                        />
                        <span className="truncate font-mono">{model}</span>
                        {model === form.model.trim() && <Pill tone="primary">Default</Pill>}
                      </label>
                    ))
                  ) : (
                    <p className="py-1 text-xs text-muted-foreground">No models match that filter.</p>
                  )}
                </div>
                <p className="text-[0.7rem] text-muted-foreground">
                  Only checked models are saved. The default model is always kept.
                </p>
              </div>
            )}
            <div className="flex flex-wrap gap-2">
              <Button
                disabled={testing || !form.baseUrl.trim()}
                onClick={() => void handleValidate()}
                variant="outline"
              >
                {testing ? <Loader2 className="animate-spin" /> : <Zap />}
                Test
              </Button>
              <Button disabled={saving || !canSave} onClick={() => void handleSave()}>
                {saving ? <Loader2 className="animate-spin" /> : <Save />}
                Save
              </Button>
              <Button
                className={cn(!form.id && 'hidden')}
                onClick={() => {
                  setForm(EMPTY_FORM)
                  setDiscoveredModels([])
                  setSelectedModels([])
                  setModelFilter('')
                }}
                type="button"
                variant="ghost"
              >
                New endpoint
              </Button>
            </div>
          </div>
        </section>
      </div>
    </SettingsContent>
  )
}
