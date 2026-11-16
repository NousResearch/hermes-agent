/**
 * A local update hands control to a detached updater that waits for the
 * Desktop window to exit, so every remote update transaction must settle
 * first — otherwise a managed SSH update can outlive the local updater's
 * exit deadline and abort the whole update.
 */
export async function updateConnectionsBeforeLocal<T extends { kind: string }, R>(
  connections: T[],
  update: (connection: T) => Promise<R>
): Promise<R[]> {
  const results = new Array<R>(connections.length)
  const indexed = connections.map((connection, index) => ({ connection, index }))

  const run = async ({ connection, index }: (typeof indexed)[number]) => {
    results[index] = await update(connection)
  }

  await Promise.all(indexed.filter(({ connection }) => connection.kind !== 'local').map(run))
  await Promise.all(indexed.filter(({ connection }) => connection.kind === 'local').map(run))

  return results
}
