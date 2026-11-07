import { useQuery, useQueryClient } from '@tanstack/react-query'
import { type ReactElement, useEffect, useState } from 'react'

import {
  getLocalModelSettings,
  resetLocalModelSettings,
  saveLocalModelSettings
} from '@/api/local-models-settings'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { useI18n } from '@/i18n'
import type { Translations } from '@/i18n/types'
import { SlidersHorizontal, Zap } from '@/lib/icons'
import { prettyName } from '@/lib/text'
import { localModelsKey, localModelsRequestScope, useLocalModelsStatus } from '@/store/local-runtime-jobs'

import { useScopedLocalModelsOwner } from './local-models-owner'
import { ListRow, Pill, RowFootnoteAction, SettingsSection, SettingsSkeleton } from './primitives'

type LocalModelsCopy = Translations['settings']['localModels']

// ── contract types (pinned by tests/hermes_cli/test_local_model_settings_api.py) ──

export interface LocalModelSettingField {
  apply: string
  choices: null | string[]
  key: string
  kind: 'bool' | 'enum' | 'float' | 'int'
  maximum: null | number
  minimum: null | number
  supports_auto: boolean
}

export interface LocalModelSettingsResponse {
  effective: { sources: Record<string, string>; values: Record<string, unknown> }
  fields: LocalModelSettingField[]
  key: { engine: string; model_fingerprint: string; workflow: string }
  loading: { context_length: null | number; policy: string }
  model_id: string
  overrides: Record<string, unknown>
  revision: number
  runtime: { backend: null | string; engine: string; installed_tag: null | string; pinned_tag: null | string }
  schema_version: number
}

// A tagged AUTO value is EXACTLY {auto:true} — the one encoding the backend
// accepts for “let the stack decide”. Never derived from falsiness: an
// explicit 0 is a decision, not an empty field.
function isAutoValue(value: unknown): boolean {
  return Boolean(value) && typeof value === 'object' && (value as { auto?: unknown }).auto === true
}

// The electron REST bridge rejects with `Error("409: {\"detail\":{…}}")`
// (optionally wrapped by the IPC bridge); pull out the status + safe code.
function settingsErrorInfo(error: unknown): { code: null | string; status: null | number } {
  const raw = error instanceof Error ? error.message : String(error)
  const marker = /(?:^\s*|error:\s*)(\d{3}):/i.exec(raw)
  const brace = raw.indexOf('{')

  if (brace !== -1) {
    try {
      const detail = (JSON.parse(raw.slice(brace)) as { detail?: unknown }).detail

      if (detail && typeof detail === 'object' && typeof (detail as { error?: unknown }).error === 'string') {
        return { code: (detail as { error: string }).error, status: marker ? Number(marker[1]) : null }
      }
    } catch {
      // Not JSON — the status marker alone still identifies the refusal class.
    }
  }

  return { code: null, status: marker ? Number(marker[1]) : null }
}

interface FieldDraft {
  invalid: boolean
  text: string
  value: undefined | unknown
}

type SourceLabelKey =
  | 'settingsSourceCatalog'
  | 'settingsSourceEngine'
  | 'settingsSourceInherit'
  | 'settingsSourceModel'
  | 'settingsSourceProfile'
  | 'settingsSourcePublisher'
  | 'settingsSourceRequest'

const SOURCE_LABELS: Record<string, SourceLabelKey> = {
  catalog: 'settingsSourceCatalog',
  engine: 'settingsSourceEngine',
  inherit: 'settingsSourceInherit',
  model: 'settingsSourceModel',
  profile: 'settingsSourceProfile',
  publisher: 'settingsSourcePublisher',
  request: 'settingsSourceRequest'
}

function rangeHint(field: LocalModelSettingField): string {
  const { maximum, minimum } = field

  if (minimum != null && maximum != null) {
    return `${minimum} – ${maximum}`
  }

  if (minimum != null) {
    return `≥ ${minimum}`
  }

  return maximum != null ? `≤ ${maximum}` : ''
}

export function LocalModelsSettingsPanel(): ReactElement {
  const owner = useScopedLocalModelsOwner()
  const client = useQueryClient()
  const { t } = useI18n()
  const copy: LocalModelsCopy = t.settings.localModels

  const { data: status } = useLocalModelsStatus(owner, true, true)
  const models = status?.models ?? []
  const [selectedId, setSelectedId] = useState<string>('')

  const selectedModel: string =
    selectedId && models.some(m => m.id === selectedId) ? selectedId : models[0]?.id ?? ''

  const queryKey = [...localModelsKey(owner, 'settings'), selectedModel]

  const { data: response, error: loadError, refetch } = useQuery<LocalModelSettingsResponse>({
    enabled: selectedModel !== '',
    queryFn: async (): Promise<LocalModelSettingsResponse> =>
      (await getLocalModelSettings(selectedModel, localModelsRequestScope(owner))) as LocalModelSettingsResponse,
    queryKey,
    retry: false
  })

  // Unsaved edits live ONLY here; the last GET/PUT response body is the
  // authoritative stored state (revision + fingerprint ride every write).
  const [drafts, setDrafts] = useState<Record<string, FieldDraft>>({})
  const [notice, setNotice] = useState<null | { kind: 'error' | 'info'; text: string }>(null)
  const [busy, setBusy] = useState<boolean>(false)

  // A different model resets the whole surface; a different artifact binding
  // (model_changed) only invalidates the drafts. A notice set while handling
  // a write must SURVIVE the refetch's fingerprint effect — hence the split.
  const fingerprint: string = response?.key.model_fingerprint ?? ''

  useEffect((): void => {
    setNotice(null)
  }, [selectedModel])

  useEffect((): void => {
    setDrafts({})
  }, [selectedModel, fingerprint])

  function currentStored(key: string): unknown {
    return response?.overrides[key] ?? response?.effective.values[key]
  }

  function displayedValue(key: string): unknown {
    const draft = drafts[key]

    if (draft) {
      return draft.value
    }

    return currentStored(key)
  }

  function setDraft(key: string, draft: null | FieldDraft): void {
    setDrafts(prev => {
      const next = { ...prev }

      if (draft === null) {
        delete next[key]
      } else {
        next[key] = draft
      }

      return next
    })
  }

  // Full replacement: every field keeps its stored value unless it has a
  // draft. An empty numeric field drafts {auto:true} — never an omission and
  // never a coerced 0 (an explicit zero is a value the user chose).
  function buildOverrides(): Record<string, unknown> {
    const body: Record<string, unknown> = {}

    if (!response) {
      return body
    }

    for (const field of response.fields) {
      const draft = drafts[field.key]
      const value = draft ? draft.value : response.overrides[field.key]

      if (value !== undefined) {
        body[field.key] = value
      }
    }

    return body
  }

  async function handleWriteError(error: unknown): Promise<void> {
    const { code, status: httpStatus } = settingsErrorInfo(error)

    if (httpStatus === 409 && code === 'model_changed') {
      // The bytes behind these settings are gone: drop pending edits and
      // re-read against the NEW binding. Never replay the refused write.
      setDrafts({})
      await refetch()
      setNotice({ kind: 'info', text: copy.settingsModelChangedReloaded })

      return
    }

    if (httpStatus === 409 && code === 'revision_conflict') {
      // Someone else wrote: re-read the authority, keep the user's drafts so
      // they can re-apply — and NEVER auto-retry the lost write.
      await refetch()
      setNotice({ kind: 'info', text: copy.settingsConflictReloaded })

      return
    }

    if (code === 'unknown_setting') {
      setNotice({ kind: 'error', text: copy.settingsErrorUnknownSetting })

      return
    }

    if (code === 'invalid_setting_value') {
      setNotice({ kind: 'error', text: copy.settingsErrorInvalidValue })

      return
    }

    setNotice({ kind: 'error', text: copy.settingsErrorRejected })
  }

  async function save(): Promise<void> {
    if (!response) {
      return
    }

    setBusy(true)
    setNotice(null)

    try {
      const body = (await saveLocalModelSettings(
        selectedModel,
        {
          expected_fingerprint: response.key.model_fingerprint,
          expected_revision: response.revision,
          overrides: buildOverrides()
        },
        localModelsRequestScope(owner)
      )) as LocalModelSettingsResponse

      // The save response IS the new authority.
      client.setQueryData(queryKey, body)
      setDrafts({})
    } catch (error) {
      await handleWriteError(error)
    } finally {
      setBusy(false)
    }
  }

  async function resetKeys(keys: null | string[]): Promise<void> {
    if (!response) {
      return
    }

    setBusy(true)
    setNotice(null)

    try {
      const body = (await resetLocalModelSettings(
        selectedModel,
        {
          expected_fingerprint: response.key.model_fingerprint,
          expected_revision: response.revision,
          keys
        },
        localModelsRequestScope(owner)
      )) as LocalModelSettingsResponse

      client.setQueryData(queryKey, body)

      if (keys === null) {
        setDrafts({})
      } else {
        for (const key of keys) {
          setDraft(key, null)
        }
      }
    } catch (error) {
      await handleWriteError(error)
    } finally {
      setBusy(false)
    }
  }

  if (!status) {
    return <SettingsSkeleton sections={[{ rows: 6 }]} />
  }

  if (models.length === 0) {
    return <p className="text-sm text-muted-foreground">{copy.settingsEmpty}</p>
  }

  if (!response) {
    return loadError ? (
      <p className="text-sm text-destructive" role="alert">
        {copy.settingsLoadFailed}
      </p>
    ) : (
      <SettingsSkeleton sections={[{ rows: 6 }]} />
    )
  }

  const dirty = Object.keys(drafts).length > 0
  const anyInvalid = Object.values(drafts).some(draft => draft.invalid)
  const hasStored = Object.keys(response.overrides).length > 0

  const contextLabel = response.loading.context_length == null
    ? copy.settingsAuto
    : String(response.loading.context_length)

  return (
    <div aria-busy={busy}>
      <SettingsSection icon={SlidersHorizontal} title={copy.settingsModelLabel}>
        <Select
          disabled={busy}
          onValueChange={value => setSelectedId(value)}
          value={selectedModel}
        >
          <SelectTrigger aria-label={copy.settingsModelLabel} className="w-72 max-w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {models.map(model => (
              <SelectItem key={model.id} value={model.id}>
                {model.id}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </SettingsSection>

      <div className="grid gap-1">
        {response.fields.map(field => (
          <SamplingFieldRow
            copy={copy}
            current={displayedValue(field.key)}
            disabled={busy}
            draft={drafts[field.key]}
            field={field}
            key={field.key}
            onAutoToggle={(): void => {
              if (isAutoValue(displayedValue(field.key))) {
                // Leaving Auto resumes the last explicit value if one exists,
                // otherwise an empty field the user fills in.
                const explicit = [response.overrides[field.key], response.effective.values[field.key]].find(
                  value => value !== undefined && !isAutoValue(value)
                )

                setDraft(field.key, {
                  invalid: false,
                  text: explicit === undefined ? '' : String(explicit),
                  value: explicit
                })
              } else {
                setDraft(field.key, { invalid: false, text: '', value: { auto: true } })
              }
            }}
            onDraftChange={(draft): void => setDraft(field.key, draft)}
            onReset={(): void => {
              if (!busy) {
                void resetKeys([field.key])
              }
            }}
            source={response.effective.sources[field.key]}
          />
        ))}
      </div>

      <div className="mt-4 flex items-center gap-2">
        <Button disabled={!dirty || anyInvalid || busy} onClick={() => void save()} size="sm">
          {busy ? copy.settingsSaving : copy.settingsSave}
        </Button>
        <Button disabled={!hasStored || busy} onClick={() => void resetKeys(null)} size="sm" variant="outline">
          {copy.settingsResetAll}
        </Button>
      </div>

      {notice && (
        <p
          className={`mt-2 text-xs ${notice.kind === 'error' ? 'text-destructive' : 'text-muted-foreground'}`}
          role="status"
        >
          {notice.text}
        </p>
      )}

      <div className="mt-6">
        <SettingsSection icon={Zap} title={copy.settingsLoadingTitle}>
          {/* Read-only on purpose: loading knobs stay Auto policy until a real
              planner binds them — the editor must never pretend otherwise. */}
          <ListRow description={copy.settingsLoadingDetail(response.loading.policy, contextLabel)} title={copy.settingsLoadingTitle} />
        </SettingsSection>
      </div>
    </div>
  )
}

function SamplingFieldRow({
  copy,
  current,
  disabled,
  draft,
  field,
  onAutoToggle,
  onDraftChange,
  onReset,
  source
}: {
  copy: LocalModelsCopy
  current: unknown
  disabled: boolean
  draft: null | FieldDraft | undefined
  field: LocalModelSettingField
  onAutoToggle: () => void
  onDraftChange: (draft: null | FieldDraft) => void
  onReset: () => void
  source: string | undefined
}): ReactElement {
  const label = prettyName(field.key)
  const auto = isAutoValue(current)
  const sourceKey = source ? SOURCE_LABELS[source] : undefined
  const sourceLabel = sourceKey ? copy[sourceKey] : source

  function textControl(): ReactElement {
    if (field.kind === 'bool') {
      return (
        <Switch
          aria-label={label}
          checked={auto ? false : Boolean(current)}
          disabled={disabled || auto}
          onCheckedChange={on => onDraftChange({ invalid: false, text: String(on), value: on })}
        />
      )
    }

    if (field.kind === 'enum') {
      const value = auto || typeof current !== 'string' ? '' : current

      return (
        <Select
          disabled={disabled || auto}
          onValueChange={next => onDraftChange({ invalid: false, text: next, value: next })}
          value={value || undefined}
        >
          <SelectTrigger aria-label={label} className="w-40">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {(field.choices ?? []).map(choice => (
              <SelectItem key={choice} value={choice}>
                {choice}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )
    }

    return (
      <Input
        aria-invalid={draft?.invalid || undefined}
        aria-label={label}
        className="w-36"
        disabled={disabled || auto}
        max={field.maximum ?? undefined}
        min={field.minimum ?? undefined}
        onChange={event => {
          const raw = event.target.value
          const trimmed = raw.trim()

          if (trimmed === '') {
            // A cleared field is AUTO, never an accidental 0 or a silent omit.
            onDraftChange({ invalid: false, text: '', value: field.supports_auto ? { auto: true } : undefined })

            return
          }

          const parsed = Number(trimmed)

          const invalid =
            !Number.isFinite(parsed) ||
            (field.minimum != null && parsed < field.minimum) ||
            (field.maximum != null && parsed > field.maximum)

          // Explicit zero survives: parsed stays 0, and it rides into the save body.
          onDraftChange({ invalid, text: raw, value: invalid ? undefined : parsed })
        }}
        placeholder={rangeHint(field)}
        type="number"
        // A draft's raw text rules the box (typing); before a draft the stored/
        // effective value shows, and an AUTO value renders as an empty, disabled
        // box against the Auto badge.
        value={draft ? draft.text : auto || current === undefined || current === null ? '' : String(current)}
      />
    )
  }

  return (
    <ListRow
      action={
        <>
          {textControl()}
          {field.supports_auto && (
            <Button
              aria-pressed={auto}
              disabled={disabled}
              onClick={onAutoToggle}
              size="sm"
              variant={auto ? 'default' : 'outline'}
            >
              {copy.settingsAuto}
            </Button>
          )}
        </>
      }
      below={
        <>
          {draft?.invalid && (
            <div className="mt-1 text-xs text-destructive" role="alert">
              {rangeHint(field)}
            </div>
          )}
          <RowFootnoteAction onClick={onReset}>{copy.settingsResetField}</RowFootnoteAction>
        </>
      }
      data-tour={field.key}
      title={
        <span className="inline-flex items-center gap-2">
          {label}
          {sourceLabel && (
            <Pill data-testid={`source-${field.key}`} tone={source === 'model' ? 'primary' : 'muted'}>
              {sourceLabel}
            </Pill>
          )}
        </span>
      }
    />
  )
}
