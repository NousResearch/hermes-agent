import { EventEmitter } from 'node:events'

import { createClient, type Message, Variant } from 'dbus-native'

const SERVICE = 'org.freedesktop.Notifications'
const PATH = '/org/freedesktop/Notifications'
const DBUS = 'org.freedesktop.DBus'
const DELIVERY_TIMEOUT_MS = 5000
const RETRY_COOLDOWN_MS = 10_000
const MAX_PENDING = 32

interface NotificationOptions {
  title: string
  body: string
  silent: boolean
  icon?: string
  actions: { text: string }[]
}

interface Connection {
  bus: ReturnType<typeof createClient>
  ready: Promise<void>
  generation: number
  live: Map<string, { owner: string; receive: (member: string, value: unknown) => void }>
}

// Electron's Linux presenter uses synchronous libnotify calls for capabilities,
// show AND close. Keep all of those off the main thread by using async D-Bus,
// rather than timing a native call that prevents its own watchdog from running.
export function createLinuxNotifications() {
  let connection: Connection | undefined
  let retryAfter = 0
  let pending = 0

  function connect(): Connection {
    if (connection) {
      return connection
    }

    // Own Hello so a rejected/timed-out handshake is a delivery failure, not
    // dbus-native's automatic Hello callback throwing out of the event loop.
    const bus = createClient({
      busAddress:
        process.env.DBUS_SESSION_BUS_ADDRESS ||
        `unix:path=${process.env.XDG_RUNTIME_DIR || `/run/user/${process.getuid!()}`}/bus`,
      authMethods: ['EXTERNAL'],
      direct: true,
      timeout: DELIVERY_TIMEOUT_MS
    })

    const state: Connection = { bus, ready: Promise.resolve(), generation: 0, live: new Map() }
    connection = state

    const disconnect = () => {
      if (connection !== state) {
        return
      }
      connection = undefined
      state.generation++

      for (const item of state.live.values()) {
        item.receive('failed', undefined)
      }
      state.live.clear()
      bus.connection.stream.destroy()
    }

    bus.connection.on('error', disconnect)
    bus.connection.on('close', disconnect)
    bus.connection.on('message', (message: Message) => {
      if (message.type !== 4) {
        return
      }

      if (
        message.sender === DBUS &&
        message.interface === DBUS &&
        message.member === 'NameOwnerChanged' &&
        message.body?.[0] === SERVICE
      ) {
        state.generation++

        for (const item of state.live.values()) {
          if (item.owner === message.body[1]) {
            item.receive('failed', undefined)
          }
        }

        return
      }

      if (message.path !== PATH || message.interface !== SERVICE || !message.member) {
        return
      }
      const [id, value] = message.body ?? []
      state.live.get(`${message.sender}:${id}`)?.receive(message.member, value)
    })
    state.ready = (async () => {
      const startup = new AbortController()
      const timer = setTimeout(() => startup.abort(), DELIVERY_TIMEOUT_MS)
      const options = { signal: startup.signal, timeout: DELIVERY_TIMEOUT_MS }
      try {
        bus.name = await bus.invokeDbus<string>({ member: 'Hello' }, options)

        for (const rule of [
          `type='signal',interface='${SERVICE}',path='${PATH}'`,
          `type='signal',sender='${DBUS}',interface='${DBUS}',member='NameOwnerChanged',arg0='${SERVICE}'`
        ]) {
          await bus.invokeDbus({ member: 'AddMatch', signature: 's', body: [rule] }, options)
        }
      } catch (error) {
        disconnect()
        throw error
      } finally {
        clearTimeout(timer)
      }
    })()

    return state
  }

  function create(options: NotificationOptions) {
    const notification = new EventEmitter()
    let delivered: { state: Connection; owner: string; id: number } | undefined
    let finished = false
    const abort = new AbortController()

    const release = () => {
      finished = true

      if (delivered) {
        delivered.state.live.delete(`${delivered.owner}:${delivered.id}`)
      }
      delivered = undefined
    }

    const close = () => {
      const target = delivered
      abort.abort()
      release()

      if (!target) {
        return
      }
      const { state, owner, id } = target
      // The retention timer must never re-enter synchronous libnotify either.
      void state.bus
        .invoke(
          {
            destination: owner,
            path: PATH,
            interface: SERVICE,
            member: 'CloseNotification',
            signature: 'u',
            body: [id]
          },
          { timeout: DELIVERY_TIMEOUT_MS }
        )
        .catch(() => {
          // Already dismissed is an error in the protocol. Closing is best-effort,
          // never retried, and must not suppress unrelated healthy deliveries.
        })
    }

    const show = async (): Promise<boolean> => {
      if (finished || Date.now() < retryAfter || pending >= MAX_PENDING) {
        release()
        notification.emit('failed')

        return false
      }

      pending++
      const timer = setTimeout(() => abort.abort(), DELIVERY_TIMEOUT_MS)
      const callOptions = { signal: abort.signal, timeout: DELIVERY_TIMEOUT_MS }
      // The owner this delivery was addressed to; the catch below reads it to
      // tell a daemon that failed from a daemon that was replaced under us.
      let addressed: { state: Connection; generation: number } | undefined

      try {
        const state = connect()
        await state.ready
        const { bus } = state
        const getOwner = () =>
          new Promise<{ owner: string; generation: number }>((resolve, reject) => {
            bus.invokeDbus(
              { member: 'GetNameOwner', signature: 's', body: [SERVICE] },
              callOptions,
              (error, owner: string) => {
                if (error) {
                  return reject(error)
                }
                // Snapshot before another message in this read batch can replace the
                // owner. An await followed by reading generation pairs stale/new data.
                resolve({ owner, generation: state.generation })
              }
            )
          })
        let destination: { owner: string; generation: number }
        try {
          destination = await getOwner()
        } catch (error) {
          if ((error as { dbusName?: string })?.dbusName !== 'org.freedesktop.DBus.Error.NameHasNoOwner') {
            throw error
          }
          // A running daemon need not have an activation file, but an unowned
          // activatable service is healthy too. Never make this an owner-only guard.
          await bus.invokeDbus({ member: 'StartServiceByName', signature: 'su', body: [SERVICE, 0] }, callOptions)
          destination = await getOwner()
        }

        const { owner, generation } = destination
        addressed = { state, generation }
        if (state.generation !== generation) {
          throw new Error('Notification owner changed during lookup')
        }

        const caps = await bus.invoke<string[]>(
          { destination: owner, path: PATH, interface: SERVICE, member: 'GetCapabilities' },
          callOptions
        )

        if (state.generation !== generation) {
          throw new Error('Notification owner changed before delivery')
        }

        const actions =
          caps.includes('actions') && process.env.ELECTRON_USE_UBUNTU_NOTIFIER === undefined
            ? ['default', 'View', ...options.actions.flatMap((action, index) => [String(index), action.text])]
            : []

        return await new Promise<boolean>((resolve, reject) => {
          bus.invoke(
            {
              destination: owner,
              path: PATH,
              interface: SERVICE,
              member: 'Notify',
              signature: 'susssasa{sv}i',
              body: [
                'Hermes',
                0,
                options.icon || '',
                options.title,
                options.body,
                actions,
                {
                  urgency: new Variant('y', 1),
                  'desktop-entry': new Variant('s', 'hermes'),
                  'suppress-sound': new Variant('b', options.silent)
                },
                -1
              ]
            },
            callOptions,
            (error, id: number) => {
              if (error) {
                return reject(error)
              }

              if (state.generation !== generation) {
                return reject(new Error('Notification owner changed during delivery'))
              }
              if (!Number.isInteger(id) || id <= 0) {
                return reject(new Error(`Notify returned an invalid id: ${String(id)}`))
              }

              delivered = { state, owner, id }
              // Register in the reply callback, before a signal in the same D-Bus
              // read batch can arrive; awaiting the reply first loses that race.
              state.live.set(`${owner}:${id}`, {
                owner,
                receive: (member, value) => {
                  if (
                    member === 'ActionInvoked' &&
                    typeof value === 'string' &&
                    actions.some((key, index) => index % 2 === 0 && key === value)
                  ) {
                    if (value === 'default') {
                      notification.emit('click')
                    } else if (/^\d+$/.test(value) && Number(value) < options.actions.length) {
                      notification.emit('action', { actionIndex: Number(value) })
                    } else {
                      return
                    }

                    close()
                  } else if (member === 'NotificationClosed' || member === 'failed') {
                    release()
                    notification.emit(member === 'failed' ? 'failed' : 'close')
                  }
                }
              })
              notification.emit('show')
              resolve(true)
            }
          )
        })
      } catch {
        // A timeout may still have delivered remotely. Never replay it, and do
        // not discard older healthy notifications' callbacks on this connection.
        // A failure against an owner that has since been replaced says nothing
        // about the new daemon, so it earns no cooldown.
        if (!addressed || addressed.state.generation === addressed.generation) {
          retryAfter = Date.now() + RETRY_COOLDOWN_MS
        }
        release()
        notification.emit('failed')

        return false
      } finally {
        clearTimeout(timer)
        pending--
      }
    }

    return Object.assign(notification, { show, close })
  }

  return { create }
}
