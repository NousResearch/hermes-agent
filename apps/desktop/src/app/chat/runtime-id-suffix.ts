// Runtime message-id dedup markers.
//
// The chat runtime funnel (ChatRuntimeBoundary in ./index) suffixes duplicate
// message ids (`id#1`, `id#2`) so assistant-ui's MessageRepository never sees a
// colliding render key — a single duplicate id throws `performOp/link` and
// crashes the whole renderer. Real message ids are `${timestamp}-${index}-${role}`
// and never contain `#`, so any trailing `#<n>` is a render-only dedup marker.
//
// These helpers strip that marker before an id crosses BACK to the $messages
// store or the gateway (edit / reload / branch / restore); otherwise the backend
// lookup misses the real message. Idempotent on un-suffixed ids.

export function stripRuntimeIdSuffix(id: string): string {
  return id.replace(/#\d+$/, '')
}

export function stripRuntimeIdSuffixNullable(id: string | null): string | null {
  return id === null ? null : stripRuntimeIdSuffix(id)
}
