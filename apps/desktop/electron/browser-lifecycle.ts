type Driver = { call: (name: string, args: Record<string, unknown>) => Promise<any> }

export async function refreshConversationLifecycle(driver: Driver, browserSession?: string) {
  // Native PC inspection also needs its implicit lifecycle when browser tools
  // use a separate Edge/Firefox adapter. Never revive an expired transport.
  if (!await refreshLiveLifecycle(driver, {})) return false
  return browserSession === undefined || await refreshLiveLifecycle(driver, { session: browserSession })
}

export async function startBrowserLifecycle(driver: Driver, session: string) {
  // A fresh prepare is an explicit, locally approved request to start a browser run.
  for (const args of [{}, { session }]) {
    const result = await driver.call('start_session', args)
    if (result?.isError) throw new Error('The browser lifecycle could not be started. No browser action was replayed.')
  }
}

export async function refreshLiveLifecycle(driver: Driver, args: Record<string, unknown>) {
  const status = await driver.call('get_session', args)
  const state = status?.structuredContent
  // Never revive a session that has ended, or one too close to expiry to renew safely.
  if (status?.isError || state?.state !== 'active' || typeof state.expires_in_seconds !== 'number' || state.expires_in_seconds < 45) return false
  const renewed = await driver.call('start_session', args)
  if (renewed?.isError || renewed?.structuredContent?.revived === true) return false
  return true
}
