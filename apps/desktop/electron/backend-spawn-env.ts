export function desktopBackendSpawnEnv(base: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  return {
    ...base,
    // The desktop spawns `hermes serve` directly — no systemd/launchd supervisor will
    // revive a failure exit, so EX_TEMPFAIL only severs the UI's websockets and drops
    // in-flight assistant messages (#118080). Tell the gateway to stay alive on
    // all-adapters-down and let the reconnect watcher recover instead. Written LAST
    // so no earlier spread can override it. Older runtimes ignore the unknown
    // variable and keep the historical exit behavior.
    GATEWAY_ON_ALL_ADAPTERS_DOWN: 'stay_alive'
  }
}
