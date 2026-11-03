import { atom, type WritableAtom } from 'nanostores'

import { readKey, writeKey } from './storage'
import { isSharedPreferenceKey } from './ui-preferences/keys'

// A nanostore that auto-persists. Reads its seed from localStorage through the
// storage choke point (so every read/write is observable in one place) and
// writes back on every change — no per-atom subscribe boilerplate.
//
//   export const $foo = persistentAtom('hermes.desktop.foo', false, Codecs.bool)

// Maps a value to/from its stored string form. `decode` only ever sees a real
// stored string (absence falls back); `encode` returning null removes the key.
export interface Codec<T> {
  decode(raw: string): T
  encode(value: T): null | string
}

export const Codecs = {
  bool: { decode: raw => raw === 'true', encode: (value: boolean) => String(value) } as Codec<boolean>,
  nullableText: { decode: raw => raw, encode: value => value } as Codec<null | string>,
  text: { decode: raw => raw, encode: (value: string) => value } as Codec<string>,
  // Mirrors storedStringArray/persistStringArray: drops non-strings, empty → removed.
  stringArray: {
    decode: raw => {
      const parsed = JSON.parse(raw) as unknown

      return Array.isArray(parsed)
        ? parsed.filter((item): item is string => typeof item === 'string' && item.length > 0)
        : []
    },
    encode: value => (value.length === 0 ? null : JSON.stringify(value))
  } as Codec<string[]>,
  // Mirrors storedStringRecord/persistStringRecord: keeps only string values.
  stringRecord: {
    decode: raw => {
      const parsed = JSON.parse(raw) as unknown

      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        return {}
      }

      return Object.fromEntries(
        Object.entries(parsed).filter((entry): entry is [string, string] => typeof entry[1] === 'string')
      )
    },
    encode: value => JSON.stringify(value)
  } as Codec<Record<string, string>>,
  /** JSON with an optional sanitizer for untrusted persisted shapes. */
  json<T>(sanitize?: (value: unknown) => T): Codec<T> {
    return {
      decode: raw => {
        const parsed = JSON.parse(raw) as unknown

        return sanitize ? sanitize(parsed) : (parsed as T)
      },
      encode: value => JSON.stringify(value)
    }
  }
}

function decodeOr<T>(raw: null | string, fallback: T, codec: Codec<T>): T {
  if (raw === null) {
    return fallback
  }

  try {
    return codec.decode(raw)
  } catch {
    return fallback
  }
}

export function persistentAtom<T>(key: string, fallback: T, codec: Codec<T> = Codecs.json<T>()): WritableAtom<T> {
  const $value = atom<T>(decodeOr(readKey(key), fallback, codec))

  // Persist CHANGES only — never the creation-time value. nanostores'
  // subscribe fires immediately, and writing what was just read back is a
  // no-op at best; at worst it is a data-loss clobber: on a cold boot the
  // renderer bundle can run against a storage snapshot that has not caught
  // up yet (an early hidden/boot load of the same bundle sees an empty
  // area), and echoing the fallback back out overwrites the real record
  // other loads are about to read. Observed with the unread-dot records:
  // the early load wrote `{}` over a populated store between the disk read
  // and the main window's module init.
  let creationEmission = true
  // A value adopted from storage (another window, or the machine's shared
  // preferences store) is already persisted; writing it back would turn a
  // peer's "back to default" removal into a stored fallback.
  let adopting = false

  $value.subscribe(value => {
    if (creationEmission || adopting) {
      creationEmission = false

      return
    }

    writeKey(key, codec.encode(value))
  })

  // Machine-level preferences follow changes made elsewhere live (#145213).
  if (isSharedPreferenceKey(key) && typeof window !== 'undefined') {
    window.addEventListener('storage', event => {
      if (event.key !== key) {
        return
      }

      adopting = true

      try {
        // The stored value, not the event's: events from other windows queue,
        // and an older one must not undo a newer edit made here meanwhile.
        $value.set(decodeOr(readKey(key), fallback, codec))
      } finally {
        adopting = false
      }
    })
  }

  return $value
}
