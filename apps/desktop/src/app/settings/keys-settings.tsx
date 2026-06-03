import { useCallback, useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Input } from '@/components/ui/input'
import { deleteEnvVar, getEnvVars, revealEnvVar, setEnvVar } from '@/hermes'
import { useTranslation } from '@/i18n'
import { Check, Eye, EyeOff, Save, Settings2, Trash2, Zap } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import type { EnvVarInfo } from '@/types/hermes'

import { CONTROL_TEXT } from './constants'
import {
  asText,
  includesQuery,
  prettyName,
  providerGroup,
  providerPriority,
  redactedValue,
  withoutKey
} from './helpers'
import { LoadingState, Pill, SectionHeading, SettingsContent } from './primitives'
import type { EnvPatch, EnvRowProps, ProviderGroup, SearchProps } from './types'

interface EnvActionsProps {
  varKey: string
  info: EnvVarInfo
  saving: string | null
  onEdit: () => void
  onClear: (key: string) => void
  onReveal: (key: string) => void
  isRevealed: boolean
  showReveal?: boolean
}

function EnvActions({
  varKey,
  info,
  saving,
  onEdit,
  onClear,
  onReveal,
  isRevealed,
  showReveal = true
}: EnvActionsProps) {
  const t = useTranslation()

  return (
    <div className="flex shrink-0 items-center gap-1.5">
      {info.url && (
        <Button asChild size="xs" title={t('settings.keys.actions.openDocs')} variant="ghost">
          <a href={info.url} rel="noreferrer" target="_blank">
            {t('settings.keys.actions.docs')}
          </a>
        </Button>
      )}
      {info.is_set && showReveal && (
        <Button
          onClick={() => onReveal(varKey)}
          size="icon-xs"
          title={isRevealed ? t('settings.keys.actions.hideValue') : t('settings.keys.actions.revealValue')}
          variant="ghost"
        >
          {isRevealed ? <EyeOff /> : <Eye />}
        </Button>
      )}
      <Button onClick={onEdit} size="xs" variant="outline">
        {info.is_set ? t('settings.keys.actions.replace') : t('settings.keys.actions.set')}
      </Button>
      {info.is_set && (
        <Button
          disabled={saving === varKey}
          onClick={() => onClear(varKey)}
          size="icon-xs"
          title={t('settings.keys.actions.clearValue')}
          variant="ghost"
        >
          <Trash2 />
        </Button>
      )}
    </div>
  )
}

function EnvVarRow({
  varKey,
  info,
  edits,
  revealed,
  saving,
  setEdits,
  onSave,
  onClear,
  onReveal,
  compact = false
}: EnvRowProps) {
  const t = useTranslation()
  const isEditing = edits[varKey] !== undefined
  const isRevealed = revealed[varKey] !== undefined
  const value = isRevealed ? revealed[varKey] : info.redacted_value
  const startEdit = () => setEdits(c => ({ ...c, [varKey]: '' }))

  if (compact && !isEditing) {
    return (
      <div className="flex items-center justify-between gap-3 py-1.5">
        <div className="min-w-0">
          <div className="truncate font-mono text-[0.72rem] text-muted-foreground">{varKey}</div>
          <div className="truncate text-[0.68rem] text-muted-foreground/70">{info.description}</div>
        </div>
        <EnvActions
          info={info}
          isRevealed={isRevealed}
          onClear={onClear}
          onEdit={startEdit}
          onReveal={onReveal}
          saving={saving}
          showReveal={false}
          varKey={varKey}
        />
      </div>
    )
  }

  return (
    <div className="grid gap-2 rounded-xl bg-background/55 p-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-xs font-medium">{varKey}</span>
            <Pill tone={info.is_set ? 'primary' : 'muted'}>
              {info.is_set && <Check className="size-3" />}
              {info.is_set ? t('settings.keys.status.set') : t('settings.keys.status.notSet')}
            </Pill>
          </div>
          <p className="mt-1 text-xs leading-5 text-muted-foreground">{info.description}</p>
        </div>
        <EnvActions
          info={info}
          isRevealed={isRevealed}
          onClear={onClear}
          onEdit={startEdit}
          onReveal={onReveal}
          saving={saving}
          varKey={varKey}
        />
      </div>

      {!isEditing && info.is_set && (
        <div
          className={cn(
            'rounded-md px-3 py-2 font-mono text-xs',
            isRevealed ? 'bg-background text-foreground' : 'bg-muted/30 text-muted-foreground'
          )}
        >
          {value || '---'}
        </div>
      )}

      {isEditing && (
        <div className="flex flex-wrap items-center gap-2">
          <Input
            autoFocus
            className={cn('min-w-56 flex-1 font-mono', CONTROL_TEXT)}
            onChange={e => setEdits(c => ({ ...c, [varKey]: e.target.value }))}
            placeholder={info.is_set ? t('settings.keys.placeholders.replaceValue') : t('settings.keys.placeholders.enterValue')}
            type={info.is_password ? 'password' : 'text'}
            value={edits[varKey]}
          />
          <Button disabled={saving === varKey || !edits[varKey]} onClick={() => onSave(varKey)} size="sm">
            <Save />
            {saving === varKey ? t('settings.keys.actions.saving') : t('common.save')}
          </Button>
          <Button onClick={() => setEdits(c => withoutKey(c, varKey))} size="sm" variant="outline">
            <Codicon name="close" />
            {t('common.cancel')}
          </Button>
        </div>
      )}
    </div>
  )
}

function EnvProviderGroup({
  group,
  rowProps
}: {
  group: ProviderGroup
  rowProps: Omit<EnvRowProps, 'varKey' | 'info'>
}) {
  const t = useTranslation()
  const setCount = group.entries.filter(([, info]) => info.is_set).length
  // Default-expand providers that already have at least one key set; the
  // user is much more likely to be coming back to edit those than to start
  // configuring a fresh provider from scratch.
  const [expanded, setExpanded] = useState(setCount > 0)

  return (
    <div className="overflow-hidden rounded-xl bg-background/60">
      <button
        className="flex w-full items-center justify-between gap-3 bg-transparent px-3 py-2.5 text-left hover:bg-accent/50"
        onClick={() => setExpanded(e => !e)}
        type="button"
      >
        <span className="flex min-w-0 items-center gap-2">
          <Zap className="size-4 shrink-0 text-muted-foreground" />
          <span className="truncate text-sm font-medium">
            {group.name === 'Other' ? t('settings.keys.otherProviders') : group.name}
          </span>
          {setCount > 0 && <Pill tone="primary">{t('settings.keys.countSet', { count: setCount })}</Pill>}
        </span>
        <span className="text-xs text-muted-foreground">{t('settings.keys.countKeys', { count: group.entries.length })}</span>
      </button>
      {expanded && (
        <div className="grid gap-2 bg-muted/20 p-3">
          {group.entries.map(([key, info]) => (
            <EnvVarRow compact={!info.is_set} info={info} key={key} varKey={key} {...rowProps} />
          ))}
        </div>
      )}
    </div>
  )
}

export function KeysSettings({ query }: SearchProps) {
  const t = useTranslation()
  const [vars, setVars] = useState<Record<string, EnvVarInfo> | null>(null)
  const [edits, setEdits] = useState<Record<string, string>>({})
  const [revealed, setRevealed] = useState<Record<string, string>>({})
  const [saving, setSaving] = useState<string | null>(null)

  // We used to hide ~80% of rows behind a global "Show advanced" toggle, but
  // everything in this view is configuration-level — "advanced" was a poor
  // distinction. The full list is rendered now and provider groups
  // default-collapsed-unless-set keep the surface manageable.
  useEffect(() => {
    try {
      window.localStorage.removeItem('desktop.settings.keys.show_advanced')
    } catch {
      // Ignore — old key cleanup is best-effort.
    }
  }, [])

  useEffect(() => {
    let cancelled = false

    void (async () => {
      try {
        const next = await getEnvVars()

        if (!cancelled) {
          setVars(next)
        }
      } catch (err) {
        notifyError(err, t('settings.keys.loadError'))
      }
    })()

    return () => void (cancelled = true)
  }, [t])

  const filterEnv = useCallback((info: EnvVarInfo, key: string, q: string, cat: string, extra?: string) => {
    if (asText(info.category) !== cat) {
      return false
    }

    if (!q) {
      return true
    }

    return (
      key.toLowerCase().includes(q) ||
      includesQuery(info.description, q) ||
      Boolean(extra && extra.toLowerCase().includes(q))
    )
  }, [])

  const providerGroups = useMemo<ProviderGroup[]>(() => {
    if (!vars) {
      return []
    }

    const q = query.trim().toLowerCase()

    const entries = Object.entries(vars).filter(([key, info]) =>
      filterEnv(info, key, q, 'provider', providerGroup(key))
    )

    const groups = new Map<string, [string, EnvVarInfo][]>()

    for (const entry of entries) {
      const name = providerGroup(entry[0])
      groups.set(name, [...(groups.get(name) ?? []), entry])
    }

    return Array.from(groups, ([name, entries]) => ({
      name,
      priority: providerPriority(name),
      entries: entries.sort(([a], [b]) => a.localeCompare(b)),
      hasAnySet: entries.some(([, info]) => info.is_set)
    })).sort((a, b) => a.priority - b.priority || a.name.localeCompare(b.name))
  }, [filterEnv, query, vars])

  const otherGroups = useMemo(() => {
    if (!vars) {
      return []
    }

    const q = query.trim().toLowerCase()

    const labels: Record<string, string> = {
      tool: t('settings.keys.categories.tools'),
      messaging: t('settings.keys.categories.messaging'),
      setting: t('settings.keys.categories.settings')
    }

    return ['tool', 'messaging', 'setting'].flatMap(cat => {
      const entries = Object.entries(vars)
        .filter(([key, info]) => filterEnv(info, key, q, cat))
        .sort(([a], [b]) => a.localeCompare(b))

      return entries.length === 0 ? [] : [{ category: cat, label: labels[cat] ?? prettyName(cat), entries }]
    })
  }, [filterEnv, query, t, vars])

  function patchVar(key: string, patch: EnvPatch) {
    setVars(c => (c ? { ...c, [key]: { ...c[key], ...patch } } : c))
  }

  function clearLocalState(key: string) {
    setEdits(c => withoutKey(c, key))
    setRevealed(c => withoutKey(c, key))
  }

  async function handleSave(key: string) {
    const value = edits[key]

    if (!value) {
      return
    }

    setSaving(key)

    try {
      await setEnvVar(key, value)
      patchVar(key, { is_set: true, redacted_value: redactedValue(value) })
      clearLocalState(key)
      notify({ kind: 'success', title: t('settings.keys.notifications.saved'), message: t('settings.keys.notifications.updated', { key }) })
    } catch (err) {
      notifyError(err, t('settings.keys.notifications.saveFailed', { key }))
    } finally {
      setSaving(null)
    }
  }

  async function handleClear(key: string) {
    if (!window.confirm(t('settings.keys.confirmRemove', { key }))) {
      return
    }

    setSaving(key)

    try {
      await deleteEnvVar(key)
      patchVar(key, { is_set: false, redacted_value: null })
      clearLocalState(key)
      notify({ kind: 'success', title: t('settings.keys.notifications.removed'), message: t('settings.keys.notifications.removedMessage', { key }) })
    } catch (err) {
      notifyError(err, t('settings.keys.notifications.removeFailed', { key }))
    } finally {
      setSaving(null)
    }
  }

  async function handleReveal(key: string) {
    if (revealed[key]) {
      setRevealed(c => withoutKey(c, key))

      return
    }

    try {
      const result = await revealEnvVar(key)
      setRevealed(c => ({ ...c, [key]: result.value }))
    } catch (err) {
      notifyError(err, t('settings.keys.notifications.revealFailed', { key }))
    }
  }

  if (!vars) {
    return <LoadingState label={t('settings.keys.loading')} />
  }

  const rowProps = {
    edits,
    revealed,
    saving,
    setEdits,
    onSave: handleSave,
    onClear: handleClear,
    onReveal: handleReveal
  }

  const configuredCount = providerGroups.filter(g => g.hasAnySet).length

  return (
    <SettingsContent>
      <div className="mb-6">
        <SectionHeading
          icon={Zap}
          meta={t('settings.keys.countConfigured', { configured: configuredCount, total: providerGroups.length })}
          title={t('settings.keys.llmProviders')}
        />
        <div className="grid gap-2">
          {providerGroups.map(group => (
            <EnvProviderGroup group={group} key={group.name} rowProps={rowProps} />
          ))}
        </div>
      </div>

      {otherGroups.map(group => (
        <div className="mb-6" key={group.category}>
          <SectionHeading
            icon={Settings2}
            meta={t('settings.keys.countSetOfTotal', {
              count: group.entries.filter(([, i]) => i.is_set).length,
              total: group.entries.length
            })}
            title={group.label}
          />
          <div className="grid gap-2">
            {group.entries.map(([key, info]) => (
              <EnvVarRow info={info} key={key} varKey={key} {...rowProps} />
            ))}
          </div>
        </div>
      ))}
    </SettingsContent>
  )
}
