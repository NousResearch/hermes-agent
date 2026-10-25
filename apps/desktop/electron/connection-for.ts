import { resolveSettingsProfileConnection } from './connection-config'

interface ConnectionForDependencies<T> {
  assertOwner: (expected: unknown, connection: T) => void
  resolve: (connectionId: string, profile: string) => Promise<T>
}

export async function resolveConnectionForRequest<T>(
  payload: unknown,
  { assertOwner, resolve }: ConnectionForDependencies<T>
): Promise<T> {
  const request = payload && typeof payload === 'object' ? (payload as Record<string, unknown>) : {}
  const connectionId = typeof request.connectionId === 'string' ? request.connectionId.trim() : ''
  const profile = String(request.profile ?? '').trim() || 'default'

  if (!Object.hasOwn(request, 'expectedOwner')) {
    return resolve(connectionId, profile)
  }

  const connection = await resolveSettingsProfileConnection(
    profile,
    request.expectedOwner,
    candidate => resolve(connectionId, candidate)
  )

  if (request.expectedConnectionOwner) {
    assertOwner(request.expectedConnectionOwner, connection)
  }

  return connection
}
