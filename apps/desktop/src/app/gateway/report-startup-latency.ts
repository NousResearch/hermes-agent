type GatewayRequest = (
  method: 'shared_metrics.startup_latency',
  params: { elapsed_ms: number; surface: 'desktop_attach' }
) => Promise<unknown>

/** Fire-and-forget: the main process hands out the launch latency to exactly one renderer
 *  boot per app launch; the backend buckets it and drops it unless the user opted in.
 *  An older Electron shell (no bridge) or backend (no method) just skips the metric. */
export async function reportStartupLatency(
  desktop: Pick<Window['hermesDesktop'], 'claimStartupLatency'> | undefined,
  request: GatewayRequest
): Promise<void> {
  const elapsedMs = (await desktop?.claimStartupLatency?.().catch(() => null)) ?? null

  if (elapsedMs === null) {
    return
  }

  // Declared, not env-detected: a URL/cloud backend has no HERMES_DESKTOP to tell it who attached.
  await request('shared_metrics.startup_latency', { elapsed_ms: elapsedMs, surface: 'desktop_attach' }).catch(
    () => undefined
  )
}
