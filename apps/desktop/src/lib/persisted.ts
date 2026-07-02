import { atom, type WritableAtom } from 'nanostores'

import { readKey, writeKey } from './storage'

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

export function persistentAtom<T>(key: string, fallback: T, codec: Codec<T> = Codecs.json<T>()): WritableAtom<T> {
  const raw = readKey(key)
  let initial = fallback

  if (raw !== null) {
    try {
      initial = codec.decode(raw)
    } catch {
      initial = fallback
    }
  }

  const $value = atom<T>(initial)

  $value.subscribe(value => writeKey(key, codec.encode(value)))

  return $value
}
