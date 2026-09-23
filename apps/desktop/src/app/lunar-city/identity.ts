import type { EntityIdentity, EntityKey } from './model'

type FieldName =
  'board' | 'connection' | 'kind' | 'profile' | 'project' | 'run' | 'session' | 'subagent' | 'task' | 'worker'

function required(name: FieldName, value: string): string {
  if (value.trim().length === 0) {
    throw new Error(`Lunar City identity ${name === 'connection' ? 'connectionId' : `${name}Id`} is required`)
  }

  return value
}

function field(name: FieldName, value: string): string {
  const encoded = encodeURIComponent(value)

  return `${name}:string:${value.length}:${encoded}`
}

function optionalField(name: FieldName, value: string | undefined): string {
  if (value === undefined) {
    return `${name}:undefined`
  }

  return field(name, value)
}

/** Stable project-compound identity. Projects are shared per connection, not
 * per display label or active profile; encoded fields avoid delimiter aliases. */
export function projectCompoundKey(connectionId: string, projectId: string): string {
  return `compound:${field('connection', required('connection', connectionId))}:${field('project', required('project', projectId))}`
}

/**
 * Builds an opaque, canonical key from typed source identity only.  The field
 * labels, types, and length-prefixed values make the representation safe even
 * when external IDs contain the delimiter, URL syntax, Unicode, or the legacy
 * sentinel spellings used by earlier versions.
 */
export function entityKey(identity: EntityIdentity): EntityKey {
  const common = [field('kind', identity.kind), field('connection', required('connection', identity.connectionId))]

  if (identity.kind === 'profile') {
    return [...common, field('profile', required('profile', identity.profile))].join(':') as EntityKey
  }

  if (identity.kind === 'session') {
    return [
      ...common,
      field('profile', required('profile', identity.profile)),
      field('session', required('session', identity.sessionId))
    ].join(':') as EntityKey
  }

  if (identity.kind === 'subagent') {
    return [
      ...common,
      field('profile', required('profile', identity.profile)),
      field('session', required('session', identity.sessionId)),
      field('subagent', required('subagent', identity.subagentId))
    ].join(':') as EntityKey
  }

  return [
    ...common,
    field('profile', required('profile', identity.profile)),
    field('board', required('board', identity.board)),
    field('task', required('task', identity.taskId)),
    optionalField('run', identity.runId),
    optionalField('worker', identity.workerId)
  ].join(':') as EntityKey
}
