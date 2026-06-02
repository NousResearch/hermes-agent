import { useCallback, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { getHindsightConfig, saveHindsightConfig } from '@/hermes'
import { Check, Loader2, Save } from '@/lib/icons'
import { notify, notifyError } from '@/store/notifications'
import type { HindsightConfig, HindsightMode, HindsightRecallBudget } from '@/types/hermes'

import { CONTROL_TEXT } from './constants'
import { LoadingState, Pill } from './primitives'

const DEFAULT_HINDSIGHT_CONFIG: HindsightConfig = {
  mode: 'cloud',
  api_url: 'https://api.hindsight.vectorize.io',
  bank_id: 'hermes',
  recall_budget: 'mid',
  api_key_set: false
}

const HINDSIGHT_MODES: readonly { description: string; label: string; value: HindsightMode }[] = [
  { value: 'cloud', label: 'Cloud', description: 'Hindsight Cloud API (lightweight, just needs an API key)' },
  { value: 'local_external', label: 'Local External', description: 'Connect to an existing Hindsight instance' }
]

const RECALL_BUDGETS: readonly HindsightRecallBudget[] = ['low', 'mid', 'high']

export function HindsightSettings() {
  const [config, setConfig] = useState<HindsightConfig | null>(null)
  const [apiKey, setApiKey] = useState('')
  const [expanded, setExpanded] = useState(true)
  const [saving, setSaving] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const next = await getHindsightConfig()
      setConfig({ ...DEFAULT_HINDSIGHT_CONFIG, ...next })
    } catch (err) {
      notifyError(err, 'Hindsight settings failed to load')
      setConfig(DEFAULT_HINDSIGHT_CONFIG)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const save = useCallback(async () => {
    if (!config) {
      return
    }

    setSaving(true)

    try {
      await saveHindsightConfig({
        mode: config.mode,
        api_url: config.api_url,
        api_key: apiKey,
        bank_id: config.bank_id,
        recall_budget: config.recall_budget
      })
      setApiKey('')
      notify({ kind: 'success', title: 'Hindsight saved', message: 'Memory provider configuration updated.' })
      await refresh()
    } catch (err) {
      notifyError(err, 'Failed to save Hindsight settings')
    } finally {
      setSaving(false)
    }
  }, [apiKey, config, refresh])

  if (!config) {
    return <LoadingState label="Loading Hindsight settings..." />
  }

  return (
    <section className="py-3">
      <button
        aria-expanded={expanded}
        className="flex w-full items-center justify-between gap-3 rounded-lg bg-background/60 px-3 py-2 text-left hover:bg-accent/50"
        onClick={() => setExpanded(open => !open)}
        type="button"
      >
        <span className="flex min-w-0 items-center gap-2">
          <DisclosureCaret open={expanded} />
          <span className="text-[length:var(--conversation-text-font-size)] font-medium text-foreground">
            Hindsight settings
          </span>
          <Pill>{config.api_key_set ? 'API key set' : 'API key not set'}</Pill>
        </span>
      </button>

      {expanded && (
        <div className="mt-3 grid gap-4 rounded-xl bg-background/60 p-4">
        <label className="grid gap-1.5">
          <span className="text-xs font-medium text-muted-foreground">Mode</span>
          <Select
            onValueChange={value => setConfig(c => (c ? { ...c, mode: value as HindsightMode } : c))}
            value={config.mode}
          >
            <SelectTrigger className={CONTROL_TEXT}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {HINDSIGHT_MODES.map(mode => (
                <SelectItem key={mode.value} value={mode.value}>
                  <span className="grid gap-0.5">
                    <span>{mode.label}</span>
                    <span className="text-xs text-muted-foreground">{mode.description}</span>
                  </span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <span className="text-xs text-muted-foreground">
            {HINDSIGHT_MODES.find(mode => mode.value === config.mode)?.description}
          </span>
        </label>

        <label className="grid gap-1.5">
          <span className="text-xs font-medium text-muted-foreground">API key</span>
          <div className="flex flex-wrap items-center gap-2">
            <Input
              className="min-w-64 flex-1 font-mono"
              onChange={event => setApiKey(event.target.value)}
              placeholder={config.api_key_set ? 'Leave blank to keep current key' : 'Enter Hindsight API key'}
              type="password"
              value={apiKey}
            />
            {config.api_key_set && (
              <Pill tone="primary">
                <Check className="size-3" />
                Set
              </Pill>
            )}
          </div>
        </label>

        <label className="grid gap-1.5">
          <span className="text-xs font-medium text-muted-foreground">API URL</span>
          <Input
            className="font-mono"
            onChange={event => setConfig(c => (c ? { ...c, api_url: event.target.value } : c))}
            value={config.api_url}
          />
        </label>

        <label className="grid gap-1.5">
          <span className="text-xs font-medium text-muted-foreground">Bank ID</span>
          <Input
            className="font-mono"
            onChange={event => setConfig(c => (c ? { ...c, bank_id: event.target.value } : c))}
            value={config.bank_id}
          />
        </label>

        <label className="grid gap-1.5">
          <span className="text-xs font-medium text-muted-foreground">Recall budget</span>
          <Select
            onValueChange={value =>
              setConfig(c => (c ? { ...c, recall_budget: value as HindsightRecallBudget } : c))
            }
            value={config.recall_budget}
          >
            <SelectTrigger className={CONTROL_TEXT}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {RECALL_BUDGETS.map(budget => (
                <SelectItem key={budget} value={budget}>
                  {budget}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </label>

        <div className="flex justify-end">
          <Button disabled={saving} onClick={() => void save()} size="sm">
            {saving ? <Loader2 className="size-3.5 animate-spin" /> : <Save />}
            Save
          </Button>
        </div>
        </div>
      )}
    </section>
  )
}
