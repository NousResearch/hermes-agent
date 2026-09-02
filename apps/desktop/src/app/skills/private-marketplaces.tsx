import { useStore } from '@nanostores/react'
import { useEffect, useState, type ReactNode } from 'react'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { Loader2, Package, Plus, RefreshCw, Trash2 } from '@/lib/icons'
import {
  installAgentPlugin,
  loadAgentPlugins,
  updateAgentPlugin,
  type AgentPluginRow,
  type GatewayRequest
} from '@/store/agent-plugins'
import {
  $pluginMarketplaceBusy,
  $pluginMarketplaceScope,
  $pluginMarketplaces,
  $pluginMarketplacesError,
  $pluginMarketplacesStatus,
  addPluginMarketplace,
  loadPluginMarketplaces,
  removePluginMarketplace,
  type MarketplacePlugin,
  type PluginMarketplace
} from '@/store/plugin-marketplaces'
import { notify } from '@/store/notifications'

interface PendingInstall {
  entry: MarketplacePlugin
  marketplace: PluginMarketplace
  profile: null | string
  scopeKey: string
}

export function PrivateMarketplaces({
  installed,
  officialCatalog,
  profile,
  request,
  scopeKey
}: {
  installed: AgentPluginRow[]
  officialCatalog: ReactNode
  profile: null | string
  request: GatewayRequest
  scopeKey: string
}) {
  const { t } = useI18n()
  const copy = t.skills.plugins.marketplaces
  const savedMarketplaces = useStore($pluginMarketplaces)
  const marketplaceScope = useStore($pluginMarketplaceScope)
  const savedStatus = useStore($pluginMarketplacesStatus)
  const savedError = useStore($pluginMarketplacesError)
  const sameScope = marketplaceScope === scopeKey
  const marketplaces = sameScope ? savedMarketplaces : []
  const status = sameScope ? savedStatus : 'loading'
  const error = sameScope ? savedError : null
  const busy = useStore($pluginMarketplaceBusy)
  const [adding, setAdding] = useState(false)
  const [url, setUrl] = useState('')
  const [pendingInstall, setPendingInstall] = useState<PendingInstall | null>(null)
  const [installing, setInstalling] = useState(false)
  const [installError, setInstallError] = useState<string | null>(null)
  const [selectedMarketplace, setSelectedMarketplace] = useState('official')

  useEffect(() => {
    void loadPluginMarketplaces(request, profile, false, scopeKey)
  }, [profile, request, scopeKey])

  useEffect(() => {
    setAdding(false)
    setUrl('')
    setPendingInstall(null)
    setInstalling(false)
    setInstallError(null)
    setSelectedMarketplace('official')
  }, [scopeKey])

  useEffect(() => {
    if (selectedMarketplace !== 'official' && !marketplaces.some(item => item.id === selectedMarketplace)) {
      setSelectedMarketplace('official')
    }
  }, [marketplaces, selectedMarketplace])

  const add = async () => {
    if (!url.trim()) {
      return
    }
    const added = await addPluginMarketplace(request, url.trim(), profile, scopeKey)
    if (added && $pluginMarketplaceScope.get() === scopeKey) {
      setAdding(false)
      setUrl('')
      notify({ kind: 'success', message: copy.added })
    }
  }

  const install = async () => {
    if (!pendingInstall || pendingInstall.scopeKey !== scopeKey) {
      return
    }
    const pending = pendingInstall
    setInstalling(true)
    setInstallError(null)
    const result = await installAgentPlugin(request, {
      identifier: '',
      marketplaceId: pending.marketplace.id,
      marketplacePluginName: pending.entry.name,
      profile: pending.profile
    })
    if ($pluginMarketplaceScope.get() !== scopeKey) {
      return
    }
    setInstalling(false)
    if (!result.ok) {
      setInstallError(result.error || copy.installFailed(pending.entry.display_name))
      return
    }
    await loadAgentPlugins(request, profile, scopeKey)
    if ($pluginMarketplaceScope.get() !== scopeKey) {
      return
    }
    notify({ kind: 'success', message: copy.installedSuccess(pending.entry.display_name) })
    setPendingInstall(null)
  }

  return (
    <section className="max-h-80 overflow-y-auto border-t border-(--ui-stroke-secondary)">
      <div className="flex items-center justify-between gap-2 px-3 py-2">
        <div>
          <div className="text-[length:var(--conversation-caption-font-size)] font-medium text-foreground">
            {copy.title}
          </div>
          <div className="text-[0.65rem] text-(--ui-text-quaternary)">{copy.hint}</div>
        </div>
        <div className="flex items-center gap-1">
          <Button
            aria-label={copy.refresh}
            disabled={status === 'loading'}
            onClick={() => void loadPluginMarketplaces(request, profile, true, scopeKey)}
            size="xs"
            variant="text"
          >
            <RefreshCw className={status === 'loading' ? 'size-3 animate-spin' : 'size-3'} />
          </Button>
          <Button onClick={() => setAdding(true)} size="xs" variant="outline">
            <Plus className="size-3" />
            {copy.add}
          </Button>
        </div>
      </div>

      {error && <p className="px-3 pb-2 text-xs text-destructive">{error}</p>}

      <div
        aria-label={copy.title}
        className="flex gap-1 overflow-x-auto border-t border-(--ui-stroke-tertiary) px-3 py-2"
        role="tablist"
      >
        <Button
          aria-selected={selectedMarketplace === 'official'}
          onClick={() => setSelectedMarketplace('official')}
          role="tab"
          size="xs"
          variant={selectedMarketplace === 'official' ? 'outline' : 'text'}
        >
          {copy.official}
        </Button>
        {marketplaces.map(marketplace => (
          <Button
            aria-selected={selectedMarketplace === marketplace.id}
            key={marketplace.id}
            onClick={() => setSelectedMarketplace(marketplace.id)}
            role="tab"
            size="xs"
            variant={selectedMarketplace === marketplace.id ? 'outline' : 'text'}
          >
            {marketplace.name}
          </Button>
        ))}
      </div>

      {selectedMarketplace === 'official' ? (
        <div role="tabpanel">{officialCatalog}</div>
      ) : marketplaces.length === 0 && status !== 'loading' ? (
        <p className="px-3 pb-3 text-xs text-(--ui-text-tertiary)">{copy.empty}</p>
      ) : (
        marketplaces
          .filter(marketplace => marketplace.id === selectedMarketplace)
          .map(marketplace => (
            <div className="border-t border-(--ui-stroke-tertiary)" key={marketplace.id}>
              <div className="flex items-start justify-between gap-2 px-3 py-2">
                <div className="min-w-0">
                  <div className="flex items-center gap-2 text-xs font-medium text-foreground">
                    <span className="truncate">{marketplace.name}</span>
                    {marketplace.stale && <span className="text-(--ui-text-quaternary)">{copy.stale}</span>}
                  </div>
                  <div className="truncate text-[0.65rem] text-(--ui-text-quaternary)">{marketplace.url}</div>
                  {marketplace.error && <div className="mt-1 text-[0.65rem] text-destructive">{marketplace.error}</div>}
                </div>
                <Button
                  aria-label={copy.remove(marketplace.name)}
                  disabled={busy === marketplace.id}
                  onClick={() => void removePluginMarketplace(request, marketplace.id, profile, scopeKey)}
                  size="xs"
                  variant="text"
                >
                  {busy === marketplace.id ? (
                    <Loader2 className="size-3 animate-spin" />
                  ) : (
                    <Trash2 className="size-3" />
                  )}
                </Button>
              </div>

              {!marketplace.available && marketplace.entries.length === 0 && (
                <p className="px-3 pb-3 text-xs text-(--ui-text-tertiary)">{copy.unavailable}</p>
              )}

              {marketplace.entries.map(entry => {
                const current = installed.find(
                  row => row.marketplace_id === marketplace.id && row.marketplace_plugin_name === entry.name
                )
                const action =
                  current?.update_available && current.key ? (
                    <Button
                      onClick={() =>
                        void updateAgentPlugin(
                          request,
                          current.key!,
                          copy.updateFailed(entry.display_name),
                          profile,
                          scopeKey
                        )
                      }
                      size="xs"
                      variant="outline"
                    >
                      {copy.update}
                    </Button>
                  ) : current ? (
                    <Button disabled size="xs" variant="text">
                      {current.marketplace_available === false ? copy.sourceUnavailable : copy.installed}
                    </Button>
                  ) : !marketplace.available || marketplace.stale ? (
                    <Button disabled size="xs" variant="text">
                      {copy.sourceUnavailable}
                    </Button>
                  ) : entry.compatible ? (
                    <Button
                      onClick={() => {
                        setInstallError(null)
                        setPendingInstall({ entry, marketplace, profile, scopeKey })
                      }}
                      size="xs"
                      variant="outline"
                    >
                      {copy.install}
                    </Button>
                  ) : (
                    <Button disabled size="xs" title={entry.incompatibility_reason} variant="text">
                      {copy.incompatible}
                    </Button>
                  )

                return (
                  <div
                    className="flex items-start gap-3 border-t border-(--ui-stroke-tertiary) px-3 py-2"
                    key={`${marketplace.id}:${entry.name}`}
                  >
                    <Package aria-hidden className="mt-0.5 size-4 shrink-0 text-(--ui-text-tertiary)" />
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2 text-xs font-medium text-foreground">
                        {entry.display_name}
                        {entry.version && <span className="text-(--ui-text-quaternary)">v{entry.version}</span>}
                      </div>
                      {entry.description && (
                        <div className="mt-0.5 text-[0.7rem] text-(--ui-text-tertiary)">{entry.description}</div>
                      )}
                      {!entry.compatible && entry.incompatibility_reason && (
                        <div className="mt-0.5 text-[0.65rem] text-(--ui-text-quaternary)">
                          {entry.incompatibility_reason}
                        </div>
                      )}
                    </div>
                    <div className="shrink-0">{action}</div>
                  </div>
                )
              })}
            </div>
          ))
      )}

      <Dialog onOpenChange={setAdding} open={adding}>
        <DialogContent className="max-w-lg">
          <form
            onSubmit={event => {
              event.preventDefault()
              void add()
            }}
          >
            <DialogHeader>
              <DialogTitle>{copy.addTitle}</DialogTitle>
              <DialogDescription>{copy.addDescription}</DialogDescription>
            </DialogHeader>
            <Input
              aria-label={copy.urlLabel}
              autoFocus
              className="mt-4"
              onChange={event => setUrl(event.target.value)}
              placeholder="https://github.com/owner/plugin-marketplace"
              value={url}
            />
            {error && <p className="mt-2 text-xs text-destructive">{error}</p>}
            <DialogFooter className="mt-4">
              <Button disabled={busy === 'add'} onClick={() => setAdding(false)} type="button" variant="ghost">
                {copy.cancel}
              </Button>
              <Button disabled={busy === 'add' || !url.trim()} type="submit">
                {busy === 'add' && <Loader2 className="size-3 animate-spin" />}
                {copy.add}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      <Dialog
        onOpenChange={open => {
          if (!open && !installing) {
            setPendingInstall(null)
            setInstallError(null)
          }
        }}
        open={pendingInstall?.scopeKey === scopeKey}
      >
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>{copy.installTitle(pendingInstall?.entry.display_name || '')}</DialogTitle>
            <DialogDescription>
              {copy.installDescription(
                pendingInstall?.entry.display_name || '',
                pendingInstall?.marketplace.name || ''
              )}
            </DialogDescription>
          </DialogHeader>
          {installError && <p className="text-xs text-destructive">{installError}</p>}
          <DialogFooter>
            <Button disabled={installing} onClick={() => setPendingInstall(null)} variant="ghost">
              {copy.cancel}
            </Button>
            <Button disabled={installing} onClick={() => void install()}>
              {installing && <Loader2 className="size-3 animate-spin" />}
              {installing ? copy.installing : copy.install}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </section>
  )
}
