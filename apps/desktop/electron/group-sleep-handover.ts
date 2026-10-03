interface Client { request(method: string, params?: Record<string, unknown>): Promise<any>; close(): void }

/** Before this computer sleeps, ask its own gateway to hand the groups it hosts to their standby. Best effort:
 * bounded by `timeoutMs`, silent on every outcome, and `connect` must never start a backend that isn't running.
 * If the handover doesn't happen, the host simply goes offline and the normal paths apply. */
export async function handOverGroupsBeforeSleep(connect: () => Promise<Client>, timeoutMs = 3000): Promise<'handed_over' | 'skipped'> {
  let client: Client | undefined
  let timer: ReturnType<typeof setTimeout> | undefined

  const work = (async () => {
    client = await connect()
    const capability = await client.request('groups.capabilities')

    if (!Array.isArray(capability?.methods) || !capability.methods.includes('groups.succession.handover_all')) {return 'skipped' as const}
    await client.request('groups.succession.handover_all', { reason: 'sleep' })

    return 'handed_over' as const
  })()

  // A connection that arrives after the deadline is closed as soon as it does.
  void work.then(() => client?.close(), () => client?.close())

  try {
    return await Promise.race([work, new Promise<'skipped'>(resolve => {timer = setTimeout(() => resolve('skipped'), timeoutMs)})])
  } catch {
    return 'skipped'
  } finally {
    clearTimeout(timer)
    client?.close()
  }
}
