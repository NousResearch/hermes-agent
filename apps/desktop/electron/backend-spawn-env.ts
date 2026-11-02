const FOREIGN_SESSION_IDS = new Set(['CODEX_THREAD_ID', 'CODEX_SESSION_ID', 'CLAUDE_SESSION_ID'])

export function desktopBackendSpawnEnv(base: NodeJS.ProcessEnv, platform = process.platform): NodeJS.ProcessEnv {
  const env = { ...base }
  // A desktop opened by another agent inherits that agent's current conversation.
  // It is not the Hermes session. Keep provider configuration, but never hand a
  // foreign current-session identity to a backend or gateway lifecycle client.

  for (const key of Object.keys(env)) {
    if (FOREIGN_SESSION_IDS.has(platform === 'win32' ? key.toUpperCase() : key)) {
      delete env[key]
    }
  }

  return {
    ...env,
    // The desktop spawns `hermes serve` directly — no systemd/launchd supervisor will
    // revive a failure exit, so EX_TEMPFAIL only severs the UI's websockets and drops
    // in-flight assistant messages (#118080). Tell the gateway to stay alive on
    // all-adapters-down and let the reconnect watcher recover instead. Written LAST
    // so no earlier spread can override it. Older runtimes ignore the unknown
    // variable and keep the historical exit behavior.
    GATEWAY_ON_ALL_ADAPTERS_DOWN: 'stay_alive'
  }
}
