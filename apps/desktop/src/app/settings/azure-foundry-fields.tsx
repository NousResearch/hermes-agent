import type { FoundryCredential, FoundrySettings, FoundryUpdate } from '@/api/azure-foundry'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'
import { Loader2, Trash2 } from '@/lib/icons'

interface FoundryFieldsProps {
  busy: 'test' | 'save' | 'remove' | null
  form: FoundryUpdate
  settings: FoundrySettings | null
  setForm: (form: FoundryUpdate) => void
  removeKey: () => void
  submit: (action: 'test' | 'save') => void
}

export function FoundryFields({ busy, form, settings, setForm, submit, removeKey }: FoundryFieldsProps) {
  const { t } = useI18n()
  const copy = t.settings.foundry
  const ce = t.settings.customEndpoints
  const credential = form.auth.kind === 'entra_id' ? form.auth.entra : null

  function updateCredential(entra: FoundryCredential) {
    setForm({ ...form, auth: { kind: 'entra_id', entra } })
  }

  return (
    <fieldset className="grid min-w-0 gap-4 text-xs text-muted-foreground" disabled={busy !== null}>
      <label className="grid gap-1.5">
        {ce.fields.endpointUrl}
        <Input
          onChange={event => setForm({ ...form, base_url: event.target.value })}
          placeholder="https://your-resource.services.ai.azure.com/openai/v1"
          value={form.base_url}
        />
      </label>
      <label className="grid gap-1.5">
        {copy.deployment}
        <Input onChange={event => setForm({ ...form, model: event.target.value })} value={form.model} />
      </label>
      <fieldset className="grid min-w-0 gap-1.5">
        <legend className="mb-1.5">{ce.credentials.authentication}</legend>
        <SegmentedControl
          className="w-full max-w-full"
          onChange={kind =>
            setForm({
              ...form,
              auth: kind === 'api_key' ? { kind } : { kind, entra: { credential_source: 'default', scope: '' } }
            })
          }
          options={[
            { id: 'api_key', label: ce.fields.apiKey },
            { id: 'entra_id', label: 'Microsoft Entra ID' }
          ]}
          value={form.auth.kind}
        />
      </fieldset>
      {form.auth.kind === 'api_key' && (
        <label className="grid gap-1.5">
          {ce.fields.apiKey}
          <Input
            autoComplete="off"
            onChange={event => setForm({ ...form, auth: { kind: 'api_key', api_key: event.target.value } })}
            placeholder={settings?.api_key_set ? ce.fields.apiKeyNewPlaceholder : ce.fields.apiKeyPlaceholder}
            type="password"
            value={form.auth.api_key ?? ''}
          />
        </label>
      )}
      {credential && (
        <fieldset className="grid min-w-0 gap-1.5">
          <legend className="mb-1.5">{copy.source}</legend>
          <SegmentedControl
            className="w-full max-w-full"
            onChange={credential_source =>
              updateCredential(
                credential_source === 'default'
                  ? { credential_source, scope: credential.scope }
                  : { credential_source, scope: credential.scope, tenant_id: '', subscription: '' }
              )
            }
            options={[
              { id: 'default', label: copy.defaultChain },
              { id: 'azure_cli', label: copy.azureCli }
            ]}
            value={credential.credential_source}
          />
          <p>{credential.credential_source === 'azure_cli' ? copy.cliHint : copy.defaultHint}</p>
        </fieldset>
      )}
      <details className="grid gap-3">
        <summary className="cursor-pointer">{copy.advanced}</summary>
        <div className="mt-3 grid gap-3">
          {credential?.credential_source === 'azure_cli' && (
            <>
              <label className="grid gap-1.5">
                {copy.tenant}
                <Input
                  onChange={event => updateCredential({ ...credential, tenant_id: event.target.value })}
                  placeholder={copy.cliDefault}
                  value={credential.tenant_id}
                />
              </label>
              <label className="grid gap-1.5">
                {copy.subscription}
                <Input
                  onChange={event => updateCredential({ ...credential, subscription: event.target.value })}
                  placeholder={copy.cliDefault}
                  value={credential.subscription}
                />
              </label>
            </>
          )}
          {credential && (
            <label className="grid gap-1.5">
              {copy.scope}
              <Input
                onChange={event => updateCredential({ ...credential, scope: event.target.value })}
                placeholder="https://ai.azure.com/.default"
                value={credential.scope}
              />
            </label>
          )}
          <fieldset className="grid min-w-0 gap-1.5">
            <legend className="mb-1.5">{ce.apiMode}</legend>
            <SegmentedControl
              className="w-full max-w-full"
              onChange={api_mode => setForm({ ...form, api_mode })}
              options={[
                { id: 'chat_completions', label: 'Chat Completions' },
                { id: 'codex_responses', label: 'Responses API' },
                { id: 'anthropic_messages', label: 'Anthropic Messages' }
              ]}
              value={form.api_mode}
            />
          </fieldset>
        </div>
      </details>
      {settings?.api_key_set && (
        <div className="flex items-center justify-between gap-2">
          <span>{ce.fields.apiKey}</span>
          <Button
            aria-label={t.settings.credentials.remove}
            disabled={busy !== null}
            onClick={removeKey}
            variant="ghost"
            size="sm"
          >
            {busy === 'remove' ? <Loader2 className="animate-spin" /> : <Trash2 />}
            {t.settings.credentials.remove}
          </Button>
        </div>
      )}
      <div className="flex flex-wrap gap-2">
        <Button
          disabled={!form.base_url.trim() || !form.model.trim() || busy !== null}
          onClick={() => void submit('test')}
          size="sm"
          variant="outline"
        >
          {busy === 'test' && <Loader2 className="animate-spin" />}
          {ce.test}
        </Button>
        <Button
          disabled={!form.base_url.trim() || !form.model.trim() || busy !== null}
          onClick={() => void submit('save')}
          size="sm"
        >
          {busy === 'save' && <Loader2 className="animate-spin" />}
          {copy.saveAndUse}
        </Button>
      </div>
      <p>{copy.profileHint}</p>
    </fieldset>
  )
}
