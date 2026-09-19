// Spectrum's shared builders/types are a superset of what the macOS-local
// iMessage provider implements. Keep transport-specific routing decisions in
// one place so exported builders are not mistaken for runtime support.
const LOCAL_UNSUPPORTED = new Set([
  "effect",
  "poll",
  "reaction",
  "read",
]);

export function supportsProviderCapability(localMode, capability) {
  return !localMode || !LOCAL_UNSUPPORTED.has(capability);
}
