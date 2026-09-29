import { capabilityScoped, type ProfileScope } from './client'

export interface SecretsAudit {
  env_secrets: string[]
  items: { kind: 'file' | 'folder'; name: string; too_open: boolean }[]
  needs_fix: boolean
  permissions_checked: boolean
  storage: 'file'
}

const call = (profile: ProfileScope | undefined, path: string, body?: unknown) =>
  window.hermesDesktop.api<SecretsAudit>({ ...capabilityScoped(profile), body, method: body === undefined ? undefined : 'POST', path })

export const getSecretsAudit = (profile?: ProfileScope) => call(profile, '/api/secrets/audit')

export const fixSecretsPermissions = (profile?: ProfileScope) => call(profile, '/api/secrets/fix', {})
