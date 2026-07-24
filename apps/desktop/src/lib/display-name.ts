import type { Translations } from '@/i18n'

// "default" is a Hermes-reserved entity name: the default profile, the
// built-in MoA preset, and the default skin all use it as their stored key.
// Like an OS file manager showing the "Desktop" folder under a localized
// label, only the *presentation* localizes — the stored key never changes.
//
// Rules for call sites:
// - Display text only (labels, tooltips, list rows, badges). Never feed the
//   result back into config writes, RPC payloads, or equality checks against
//   backend data — those keep the raw name.
// - User-created entities can never collide: "default" is reserved by the
//   backend (the default profile/preset already owns the key), so a raw name
//   of "default" always denotes the built-in entity.
export function displayEntityName(name: string, t: Translations): string {
  return name === 'default' ? t.common.defaultName : name
}
