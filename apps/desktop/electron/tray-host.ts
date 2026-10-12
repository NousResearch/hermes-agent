import { createClient, type Message, type Variant } from 'dbus-native'

// dbus-native 0.15 defaults plainValues to true, so a property Get returns the
// bare boolean. Older builds still wrap it as { value }.
function statusNotifierHostRegistered(value: unknown): boolean {
  if (value === true) {
    return true
  }

  return typeof value === 'object' && value !== null && (value as { value?: unknown }).value === true
}

// Electron can construct a Linux Tray even when the desktop has no tray host
// (notably stock GNOME). Never use that object alone as proof of a restore path.
export async function watchLinuxTrayHost(onLost: () => void): Promise<() => void> {
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
  let owner = ''

  const dispose = () => {
    if (disposed) {
      return
    }

    disposed = true
    bus.connection.stream.destroy()
  }

  const lost = () => {
    if (disposed) {
      return
    }

    dispose()

    if (ready) {
      onLost()
    }
  }

  bus.connection.on('error', lost)
  bus.connection.on('close', lost)
  bus.connection.on('message', (message: Message) => {
    if (message.type !== 4) {
      return
    }

    if (
      message.sender === 'org.freedesktop.DBus' &&
      message.interface === 'org.freedesktop.DBus' &&
      message.member === 'NameOwnerChanged' &&
      message.body?.[0] === service &&
      message.body[1] === owner
    ) {
      lost()
    }

    if (
      message.sender === owner &&
      message.path === '/StatusNotifierWatcher' &&
      message.interface === service &&
      message.member === 'StatusNotifierHostUnregistered'
    ) {
      lost()
    }
  })
  const abort = new AbortController()
  const timer = setTimeout(() => abort.abort(), 2000)
  const options = { signal: abort.signal, timeout: 2000 }

  try {
    bus.name = await bus.invokeDbus<string>({ member: 'Hello' }, options)

    for (const rule of [
      `type='signal',sender='org.freedesktop.DBus',interface='org.freedesktop.DBus',member='NameOwnerChanged',arg0='${service}'`,
      `type='signal',interface='${service}',member='StatusNotifierHostUnregistered'`
    ]) {
      await bus.invokeDbus({ member: 'AddMatch', signature: 's', body: [rule] }, options)
    }

    owner = await bus.invokeDbus<string>({ member: 'GetNameOwner', signature: 's', body: [service] }, options)

    const value = await bus.invoke<Variant>(
      {
        destination: owner,
        path: '/StatusNotifierWatcher',
        interface: 'org.freedesktop.DBus.Properties',
        member: 'Get',
        signature: 'ss',
        body: [service, 'IsStatusNotifierHostRegistered']
      },
      options
    )

    if (disposed || !statusNotifierHostRegistered(value)) {
      throw new Error('No system tray host is available')
    }

    ready = true

    return dispose
  } catch (error) {
    dispose()
    throw error
  } finally {
    clearTimeout(timer)
  }
}
