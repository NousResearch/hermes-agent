/**
 * Whether This device would need the installer, resolved without starting it.
 * A missing bridge fails closed: neither a click nor the gateway keeper may
 * reach runBootstrap unseen.
 */
export async function localBootstrapNeeded(): Promise<boolean> {
  const probe = window.hermesDesktop?.probeLocalBackend

  if (!probe) {
    return true
  }

  return (await probe()).bootstrapNeeded === true
}
