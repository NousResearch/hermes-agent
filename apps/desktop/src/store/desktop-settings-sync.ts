/**
 * Mirror small desktop settings through the active gateway. The packaged
 * renderer briefly moved from file:// to loopback HTTP (d428df2a65, since
 * reverted), leaving origin-scoped localStorage choices inaccessible. A server
 * copy can recover them after a future origin change, provided a previous edit
 * reached that gateway. A stale server read never replaces a newer local edit.
 */
import { $gateway, activeGatewayProfileKey, requestGatewayForProfile } from '@/store/gateway'

export interface JsonMirror<T> {
  configKey: string
  get: () => T
  set: (value: T) => void
  isEmpty: (value: T) => boolean
  /** Nanostores .listen, not .subscribe: no write at module init. */
  onChange: (listener: () => void) => () => void
}

interface Mirror<T> extends JsonMirror<T> {
  revision: number
}

const mirrors: Mirror<unknown>[] = []

function push(mirror: Mirror<unknown>): void {
  // config.set is scoped to the viewed profile, not the launch profile.
  void requestGatewayForProfile(activeGatewayProfileKey(), 'config.set', {
    key: mirror.configKey,
    value: mirror.get()
  }).catch(() => {
    // Offline or older gateway. Retry on the next edit/connection.
  })
}

async function pullOrPush<T>(mirror: Mirror<T>): Promise<void> {
  const revision = ++mirror.revision
  const profile = activeGatewayProfileKey()
  if (!mirror.isEmpty(mirror.get())) {
    push(mirror as Mirror<unknown>)
    return
  }
  try {
    const result = (await requestGatewayForProfile(profile, 'config.get', {
      key: mirror.configKey
    })) as { value?: T }
    if (mirror.revision !== revision || activeGatewayProfileKey() !== profile || !mirror.isEmpty(mirror.get())) {
      return
    }
    const serverValue = result?.value
    if (serverValue !== undefined && !mirror.isEmpty(serverValue)) {
      mirror.set(serverValue)
    }
  } catch {
    // No import on an unavailable/older gateway; retry on connection.
  }
}

/** On edits, push; on connection, restore an empty store or push local choices. */
export function syncJsonSetting<T>(mirror: JsonMirror<T>): void {
  if (typeof window === 'undefined') return
  const entry: Mirror<T> = { ...mirror, revision: 0 }
  mirrors.push(entry as Mirror<unknown>)
  mirror.onChange(() => {
    entry.revision++
    push(entry as Mirror<unknown>)
  })
}

if (typeof window !== 'undefined') {
  // Some component tests mock only the gateway functions they use.
  try {
    if (typeof $gateway?.listen === 'function') {
      $gateway.listen(() => {
        for (const mirror of mirrors) void pullOrPush(mirror)
      })
    }
  } catch {
    // Incomplete gateway mock; this module has no work in that test.
  }
}
