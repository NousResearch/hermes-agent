import { Input } from '@/components/ui/input'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'
import type { CustomEndpoint, CustomEndpointCredentialSource, CustomEndpointUpdate } from '@/types/hermes'

export type EndpointCredential = { kind: 'api_key'; apiKey: string } | { kind: 'command'; command: string }

function emptyCredential(kind: CustomEndpointCredentialSource): EndpointCredential {
  switch (kind) {
    case 'api_key':
      return { kind, apiKey: '' }

    case 'command':
      return { kind, command: '' }
    default: {
      const exhaustive: never = kind

      return exhaustive
    }
  }
}

export function credentialFromEndpoint(endpoint: CustomEndpoint): EndpointCredential {
  return emptyCredential(endpoint.credential_source ?? 'api_key')
}

export function credentialPayload(
  credential: EndpointCredential
): Pick<CustomEndpointUpdate, 'credential_source' | 'api_key' | 'key_cmd'> {
  switch (credential.kind) {
    case 'api_key':
      return { credential_source: credential.kind, api_key: credential.apiKey.trim() || undefined }

    case 'command':
      return { credential_source: credential.kind, key_cmd: credential.command.trim() || undefined }
    default: {
      const exhaustive: never = credential

      return exhaustive
    }
  }
}

interface EndpointCredentialsProps {
  credential: EndpointCredential
  editing: boolean
  onChange: (credential: EndpointCredential) => void
}

export function EndpointCredentials({ credential, editing, onChange }: EndpointCredentialsProps) {
  const { t } = useI18n()
  const ce = t.settings.customEndpoints
  const copy = ce.credentials

  const options: { id: CustomEndpointCredentialSource; label: string }[] = [
    { id: 'api_key', label: ce.fields.apiKey },
    { id: 'command', label: copy.command }
  ]

  return (
    <fieldset className="grid min-w-0 gap-1.5 text-xs text-muted-foreground">
      <legend className="mb-1.5">{copy.authentication}</legend>
      <SegmentedControl
        className="w-full max-w-full"
        onChange={kind => onChange(emptyCredential(kind))}
        options={options}
        value={credential.kind}
      />
      {credential.kind === 'api_key' && (
        <label className="grid gap-1.5">
          {ce.fields.apiKey}
          <Input
            onChange={event => onChange({ kind: 'api_key', apiKey: event.target.value })}
            placeholder={editing ? ce.fields.apiKeyNewPlaceholder : ce.fields.apiKeyPlaceholder}
            type="password"
            value={credential.apiKey}
          />
        </label>
      )}
      {credential.kind === 'command' && (
        <>
          <label className="grid gap-1.5">
            {copy.commandLabel}
            <Input
              onChange={event => onChange({ kind: 'command', command: event.target.value })}
              placeholder={editing ? copy.savedCommandPlaceholder : copy.commandPlaceholder}
              type="password"
              value={credential.command}
            />
          </label>
          <p>{copy.commandHint}</p>
        </>
      )}
    </fieldset>
  )
}
