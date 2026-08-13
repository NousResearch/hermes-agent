import { useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { Switch } from '@/components/ui/switch'
import { useI18n } from '@/i18n'
import { getEnvVars, revealEnvVar } from '@/hermes'
import { normalizeProviderName, type CustomProviderEntry } from '@/lib/custom-provider-config'
import { Box } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { useRouteEnumParam } from '../hooks/use-route-enum-param'
import { PAGE_INSET_X } from '../layout-constants'

import { redactedValue } from './helpers'
import { ProviderConfigurePanel } from './provider-configure-panel'
import { ProviderFormDialog } from './provider-form-dialog'
import { isConfigurableProvider } from './provider-grouping'
import { ModelAddDialog } from './model-add-dialog'
import { ProviderManagerNav } from './provider-manager-nav'
import { ProviderModelList } from './provider-model-list'
import { useProviderConfig } from './use-provider-config'
import { useProviderModelCatalog } from './use-provider-catalog'

/**
 * Full Provider Manager view: a provider list on the left, the selected
 * provider's models (with active/inactive toggles) on the right. The selected
 * provider is deep-linkable via `?pmprovider=<slug>` so a bookmark or shared
 * link lands on the right provider. Custom providers can be added/edited via
 * a dialog; any provider can be activated/deactivated from the right-pane
 * header.
 */
export function ProviderModelManager() {
  const { t } = useI18n()
  const copy = t.providerManager
  const { providers, isPending, isError } = useProviderModelCatalog()
  const { customProviders, saveCustomProvider, deleteCustomProvider, setEnabled, discoverModels, addModel, testProviderConnection, refreshCatalog, saveBuiltInCredentials } =
    useProviderConfig()
  const [providerSearch, setProviderSearch] = useState('')
  const [formOpen, setFormOpen] = useState(false)
  const [editMode, setEditMode] = useState<'builtin' | 'custom'>('custom')
  const [editing, setEditing] = useState<CustomProviderEntry | null>(null)
  const [redactedApiKey, setRedactedApiKey] = useState<string | undefined>(undefined)
  const [addModelOpen, setAddModelOpen] = useState(false)
  const [discoverState, setDiscoverState] = useState<'idle' | 'working' | 'error' | 'done'>('idle')
  const [discoverMessage, setDiscoverMessage] = useState<string | null>(null)
  const [testState, setTestState] = useState<'idle' | 'working' | 'error' | 'done'>('idle')
  const [testMessage, setTestMessage] = useState<string | null>(null)
  const [configureState, setConfigureState] = useState<'idle' | 'working' | 'error'>('idle')
  const [configureError, setConfigureError] = useState<string | null>(null)

  const providerSlugs = providers.map(provider => provider.slug)
  const [selectedSlug, setSelectedSlug] = useRouteEnumParam('pmprovider', providerSlugs, '')

  const activeSlug = selectedSlug || providers[0]?.slug || null
  const selectedProvider = providers.find(provider => provider.slug === activeSlug) ?? null

  // Provider-list search: case-insensitive substring on name/slug.
  const filteredProviders = useMemo(() => {
    const q = providerSearch.trim().toLowerCase()
    if (!q) {
      return providers
    }
    return providers.filter(
      provider => provider.name.toLowerCase().includes(q) || provider.slug.toLowerCase().includes(q)
    )
  }, [providers, providerSearch])

  const handleAdd = () => {
    setEditing(null)
    // Reset the dialog mode + stale credential hint so a prior "Edit
    // credentials" (builtin) session doesn't leak into the Add dialog.
    setEditMode('custom')
    setRedactedApiKey(undefined)
    setFormOpen(true)
  }

  const handleEdit = () => {
    if (!selectedProvider) {
      return
    }

    if (selectedProvider.is_user_defined) {
      const norm = normalizeProviderName(selectedProvider.slug.replace(/^custom:/, ''))
      const entry = customProviders.find(c => normalizeProviderName(c.name) === norm) ?? null
      setEditing(entry)
      setEditMode('custom')
    } else {
      setEditing(null)
      setEditMode('builtin')
      setRedactedApiKey(undefined)
      // Fetch the redacted API key so the dialog can show the already-saved key
      // (same masked value the API Keys settings page renders). Env vars are
      // matched by the backend `provider`/`provider_label` identity — the SAME
      // one the Keys page groups by — with a key-prefix fallback. `key_env` is
      // only populated for *unconfigured* providers, so it can't be relied on
      // for a provider that already has a key saved.
      const slug = selectedProvider.slug
      void (async () => {
        const vars = await getEnvVars()
        const prefix = `${slug.toUpperCase().replace(/-/g, '_')}_`
        const match = Object.entries(vars).find(
          ([key, info]) =>
            info.category === 'provider' &&
            info.is_set &&
            (info.provider === slug ||
              info.provider_label?.toLowerCase() === slug.toLowerCase() ||
              key.startsWith(prefix))
        )
        if (!match) {
          return
        }
        // Prefer the backend-supplied mask; fall back to revealing the value and
        // redacting it locally (same helper the API Keys page uses) for backends
        // that return a null redacted_value.
        if (match[1].redacted_value) {
          setRedactedApiKey(match[1].redacted_value)
          return
        }
        const revealed = await revealEnvVar(match[0])
        if (revealed.value) {
          setRedactedApiKey(redactedValue(revealed.value))
        }
      })().catch(() => {
        /* non-fatal — dialog still opens without the hint */
      })
    }
    setFormOpen(true)
  }

  const handleDiscover = async () => {
    if (!selectedProvider) {
      return
    }

    setDiscoverState('working')
    setDiscoverMessage(null)

    try {
      if (selectedProvider.is_user_defined) {
        // Custom providers: hit the provider's /models endpoint directly and
        // merge discovered models into its config entry.
        const added = await discoverModels(selectedProvider.slug)
        setDiscoverState('done')
        setDiscoverMessage(added.length === 0 ? copy.discoveryEmpty : copy.discoverySuccess(added.length))
      } else {
        // Built-in providers: ask the backend to re-probe all configured
        // providers and refresh the catalog query.
        await refreshCatalog()
        setDiscoverState('done')
        setDiscoverMessage(copy.listRefreshed)
      }
    } catch (err) {
      setDiscoverState('error')
      setDiscoverMessage(err instanceof Error ? err.message : copy.discoveryFailed)
    }
  }

  const handleAddModel = () => {
    if (!selectedProvider?.is_user_defined) {
      return
    }
    setAddModelOpen(true)
  }

  const handleTestConnection = async () => {
    if (!selectedProvider?.is_user_defined) {
      return
    }

    setTestState('working')
    setTestMessage(null)

    try {
      const result = await testProviderConnection(selectedProvider.slug)
      if (result.ok) {
        setTestState('done')
        setTestMessage(copy.testOk(result.latencyMs ?? 0))
      } else {
        setTestState('error')
        setTestMessage(result.error || copy.testFailed)
      }
    } catch (err) {
      setTestState('error')
      setTestMessage(err instanceof Error ? err.message : copy.testFailed)
    }
  }

  // Configure an unconfigured built-in provider from the inline panel: persist
  // the API key (+ optional base URL override) via setEnvVar, then re-probe the
  // catalog with refresh:true so the backend discovers the provider's models and
  // marks it authenticated. The catalog invalidation flips the provider from
  // `unconfigured` → `configured` (data-driven via classifyProvider), so the nav
  // moves it into the Configured group and the right pane swaps to the model list.
  const handleConfigure = async (apiKey: string, baseUrl?: string) => {
    if (!selectedProvider?.key_env) {
      return
    }

    setConfigureState('working')
    setConfigureError(null)

    try {
      await saveBuiltInCredentials(selectedProvider.key_env, apiKey, baseUrl, selectedProvider.slug)
      await refreshCatalog()
      setConfigureState('idle')
    } catch (err) {
      setConfigureState('error')
      setConfigureError(err instanceof Error ? err.message : copy.discoveryFailed)
    }
  }

  const existingNames = customProviders
    .filter(c => !editing || normalizeProviderName(c.name) !== normalizeProviderName(editing.name))
    .map(c => c.name)

  const enabled = selectedProvider ? selectedProvider.enabled !== false : true
  const existingModelIds = selectedProvider?.models ?? []

  // An unconfigured-but-configurable built-in provider gets the inline configure
  // panel instead of the model list. Once saved, the refreshed catalog flips it
  // to `configured` and this guard becomes false, swapping in the model list.
  const showConfigurePanel = selectedProvider != null && isConfigurableProvider(selectedProvider)

  return (
    <div className={cn('flex h-full min-h-0 flex-col pb-6', PAGE_INSET_X)}>
      <header className="mb-5 shrink-0 border-b border-(--ui-stroke-tertiary) pb-4">
        <div className="flex items-center gap-2 text-[length:var(--conversation-text-font-size)] font-medium">
          <Box className="size-4 text-muted-foreground" />
          {copy.title}
        </div>
        <p className="mt-2 max-w-2xl text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
          {copy.description}
        </p>
      </header>

      <div className="flex min-h-0 flex-1">
        <div className="w-56 shrink-0 border-r border-(--ui-stroke-tertiary)">
          <ProviderManagerNav
            onAdd={handleAdd}
            onProviderSearch={setProviderSearch}
            onSelect={setSelectedSlug}
            providerSearch={providerSearch}
            providers={filteredProviders}
            selectedSlug={activeSlug}
          />
        </div>

        <div className="flex min-w-0 flex-1 flex-col">
        {isPending ? (
          <div className="flex h-full items-center justify-center">
            <GlyphSpinner className="text-sm" />
          </div>
        ) : isError || providers.length === 0 ? (
          <div className="px-3 py-5 text-center text-xs text-muted-foreground">{copy.noProviders}</div>
        ) : selectedProvider ? (
          showConfigurePanel ? (
            <div className="min-h-0 flex-1">
              <ProviderConfigurePanel
                error={configureError}
                onConfigure={(apiKey, baseUrl) => void handleConfigure(apiKey, baseUrl)}
                provider={selectedProvider}
                working={configureState === 'working'}
              />
            </div>
          ) : (
            <>
              <div className="flex items-center justify-between gap-2 px-3 py-1.5">
                <span className="min-w-0 flex-1 truncate text-sm font-medium">{selectedProvider.name}</span>
                <Button onClick={handleEdit} type="button" variant="ghost">
                  {selectedProvider.is_user_defined ? copy.editProvider : copy.editProviderCredentials}
                </Button>
                {selectedProvider.is_user_defined && (
                  <Button
                    disabled={!enabled || testState === 'working'}
                    onClick={() => void handleTestConnection()}
                    type="button"
                    variant="ghost"
                  >
                    {copy.testConnection}
                  </Button>
                )}
                <Switch
                  aria-label={enabled ? copy.disableProvider : copy.enableProvider}
                  checked={enabled}
                  onCheckedChange={value => setEnabled(selectedProvider.slug, value)}
                />
              </div>

              <div className="min-h-0 flex-1">
                <ProviderModelList
                  discoverWorking={discoverState === 'working'}
                  enabled={enabled}
                  onAddModel={handleAddModel}
                  onDiscover={handleDiscover}
                  provider={selectedProvider}
                />
              </div>

              {discoverState !== 'idle' && discoverMessage && (
                <div
                  className={
                    discoverState === 'error'
                      ? 'mx-3 mb-2 rounded border border-destructive/30 bg-destructive/10 px-2 py-1 text-[0.6875rem] text-destructive'
                      : 'mx-3 mb-2 rounded bg-(--ui-bg-tertiary) px-2 py-1 text-[0.6875rem] text-(--ui-text-tertiary)'
                  }
                >
                  {discoverMessage}
                </div>
              )}

              {testState !== 'idle' && testMessage && (
                <div
                  className={
                    testState === 'error'
                      ? 'mx-3 mb-2 rounded border border-destructive/30 bg-destructive/10 px-2 py-1 text-[0.6875rem] text-destructive'
                      : 'mx-3 mb-2 rounded bg-(--ui-bg-tertiary) px-2 py-1 text-[0.6875rem] text-(--ui-text-tertiary)'
                  }
                >
                  {testMessage}
                </div>
              )}
            </>
          )
        ) : (
          <div className="px-3 py-5 text-center text-xs text-muted-foreground">{copy.selectProviderHint}</div>
        )}
        </div>
      </div>

      <ProviderFormDialog
        existingNames={existingNames}
        initial={editing}
        mode={editMode}
        onClose={() => setFormOpen(false)}
        onDelete={deleteCustomProvider}
        onSave={saveCustomProvider}
        onSaveBuiltIn={(apiKey, baseUrl) =>
          selectedProvider?.key_env
            ? saveBuiltInCredentials(selectedProvider.key_env, apiKey, baseUrl, selectedProvider.slug)
            : Promise.resolve()
        }
        open={formOpen}
        providerName={selectedProvider?.name ?? ''}
        redactedApiKey={redactedApiKey}
      />

      <ModelAddDialog
        existingIds={existingModelIds}
        onClose={() => setAddModelOpen(false)}
        onSave={model => {
          if (selectedProvider) {
            void addModel(selectedProvider.slug, model)
          }
        }}
        open={addModelOpen}
      />
    </div>
  )
}
