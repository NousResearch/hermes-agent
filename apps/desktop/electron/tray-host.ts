import { createClient, type Message, type Variant } from 'dbus-native'

// dbus-native 0.15 defaults plainValues to true, so a property Get returns the
// bare boolean. Older builds still wrap it as { value }.
function statusNotifierHostRegistered(value: unknown): boolean {
  if (value === true) {
    return true
  }

  return typeof value === 'object' && value !== null && (value as { value?: unknown }).value === true
}

// GNOME's AppIndicator extension drops its watcher name while the screen is
// locked. Chromium re-registers its item the moment the name returns, but the
// extension shows nothing for that early registration; a later one resets the
// indicator and makes it visible (#145282).
const REREGISTER_DELAY_MS = 1000

// Electron can construct a Linux Tray even when the desktop has no tray host
// (notably stock GNOME). Never use that object alone as proof of a restore path.
// onChange reports the host leaving and returning until the session bus closes.
export async function watchLinuxTrayHost(onChange: (available: boolean) => void): Promise<() => void> {
  const service = 'org.kde.StatusNotifierWatcher'

  const bus = createClient({
    busAddress:
      process.env.DBUS_SESSION_BUS_ADDRESS ||
      `unix:path=${process.env.XDG_RUNTIME_DIR || `/run/user/${process.getuid!()}`}/bus`,
    authMethods: ['EXTERNAL'],
    direct: true,
    timeout: 2000
  })

  let disposed = false
  let ready = false
  let available = false
  let owner = ''
  let item = ''
  let losses = 0
  let returnTimer: ReturnType<typeof setTimeout> | undefined

  const dispose = () => {
    if (disposed) {
      return
    }

    disposed = true
    clearTimeout(returnTimer)
    bus.connection.stream.destroy()
  }

  const report = (next: boolean) => {
    if (disposed) {
      return
    }

    if (!ready) {
      // Loss during the probe fails the probe, as before.
      if (!next) {
        dispose()
      }

      return
    }

    if (!next) {
      losses += 1
      clearTimeout(returnTimer)
    }

    if (available !== next) {
      available = next
      onChange(next)
    }
  }

  const hostRegistered = async (options?: { signal: AbortSignal; timeout: number }) =>
    statusNotifierHostRegistered(
      await bus.invoke<Variant>(
        {
          destination: owner,
          path: '/StatusNotifierWatcher',
          interface: 'org.freedesktop.DBus.Properties',
          member: 'Get',
          signature: 'ss',
          body: [service, 'IsStatusNotifierHostRegistered']
        },
        options ?? { timeout: 2000 }
      )
    )

  const hostReturned = async () => {
    const since = losses

    try {
      if (disposed || !(await hostRegistered())) {
        return
      }

      if (item) {
        await bus.invoke({
          destination: owner,
          path: '/StatusNotifierWatcher',
          interface: service,
          member: 'RegisterStatusNotifierItem',
          signature: 's',
          body: [item]
        })
      }

      if (since === losses) {
        report(true)
      }
    } catch {
      // The next owner change or host registration retries.
    }
  }

  const scheduleReturn = () => {
    clearTimeout(returnTimer)
    returnTimer = setTimeout(() => void hostReturned(), REREGISTER_DELAY_MS)
  }

  const closed = () => {
    report(false)
    dispose()
  }

  bus.connection.on('error', closed)
  bus.connection.on('close', closed)
  bus.connection.on('message', (message: Message) => {
    if (message.type !== 4) {
      return
    }

    if (
      message.sender === 'org.freedesktop.DBus' &&
      message.interface === 'org.freedesktop.DBus' &&
      message.member === 'NameOwnerChanged' &&
      message.body?.[0] === service
    ) {
      if (owner && message.body[1] === owner) {
        report(false)
      }

      if (message.body[2]) {
        owner = String(message.body[2])
        scheduleReturn()
      }

      return
    }

    if (message.sender !== owner || message.path !== '/StatusNotifierWatcher' || message.interface !== service) {
      return
    }

    if (message.member === 'StatusNotifierHostUnregistered') {
      report(false)
    } else if (message.member === 'StatusNotifierHostRegistered') {
      scheduleReturn()
    } else if (
      message.member === 'StatusNotifierItemRegistered' &&
      String(message.body?.[0]).startsWith(`org.freedesktop.StatusNotifierItem-${process.pid}-`)
    ) {
      item = String(message.body?.[0])
    }
  })
  const abort = new AbortController()
  const timer = setTimeout(() => abort.abort(), 2000)
  const options = { signal: abort.signal, timeout: 2000 }

  try {
    bus.name = await bus.invokeDbus<string>({ member: 'Hello' }, options)

    for (const rule of [
      `type='signal',sender='org.freedesktop.DBus',interface='org.freedesktop.DBus',member='NameOwnerChanged',arg0='${service}'`,
      `type='signal',interface='${service}',member='StatusNotifierHostUnregistered'`,
      `type='signal',interface='${service}',member='StatusNotifierHostRegistered'`,
      `type='signal',interface='${service}',member='StatusNotifierItemRegistered'`
    ]) {
      await bus.invokeDbus({ member: 'AddMatch', signature: 's', body: [rule] }, options)
    }

    owner = await bus.invokeDbus<string>({ member: 'GetNameOwner', signature: 's', body: [service] }, options)

    if (!(await hostRegistered(options)) || disposed) {
      throw new Error('No system tray host is available')
    }

    ready = true
    available = true

    return dispose
  } catch (error) {
    dispose()
    throw error
  } finally {
    clearTimeout(timer)
  }
}
