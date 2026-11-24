// Main-process refusals from a local `hermes gateway ensure` (electron/local-gateway.ts) that no
// automatic redial can fix: every redial of the scope re-spawns the same ensure client against the
// same install and gets the same verdict. They need an update, a profile fix or a path repair.
// The renderer sees them wrapped by ipcRenderer.invoke ("Error invoking remote method …: Error: …"),
// so match the stable message heads, never the whole string.
const PERMANENT_ENSURE_FAILURES = [
  // ensureLocalGateway: protocol verdict `incompatible` (runtime_protocol, session authority missing).
  'Gateway incompatible (',
  // parseGatewayEnsureOutput: the installed `hermes` has no ensure protocol, or the profile is missing.
  'hermes gateway ensure produced no result',
  // assertLocalGatewayEndpoint: a ready verdict that is not a loopback session-authority endpoint.
  'Invalid local gateway endpoint',
  // privateNode / the group-writable ticket refusal: needs a chmod/chown, never a re-ensure.
  'Unsafe gateway control path'
]

export function isPermanentLocalGatewayEnsureError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error ?? '')

  return PERMANENT_ENSURE_FAILURES.some(head => message.includes(head))
}
