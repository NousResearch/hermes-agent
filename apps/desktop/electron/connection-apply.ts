async function applyConnectionChange({
  cancelAndWait,
  isPrimary,
  rehomePrimary = null,
  scope,
  sendApplied,
  stopPool,
  teardownPrimary,
  teardownSsh
}) {
  await cancelAndWait(scope)
  await teardownSsh(scope)

  if (!isPrimary) {
    stopPool(scope)

    return
  }

  if (rehomePrimary) {
    await rehomePrimary()

    return
  }

  await teardownPrimary()
  sendApplied()
}

function commitConnectionFailure(current, starting, commit) {
  if (current !== starting) {
    return false
  }

  commit()

  return true
}

async function resolveTerminalConnection(getTarget, ensureBackend) {
  let target = getTarget()

  if (target !== 'pending') {
    return target
  }

  await ensureBackend()
  target = getTarget()

  if (target === 'pending') {
    throw new Error('Remote connection is not ready yet. Try again in a moment.')
  }

  return target
}

async function resolveTerminalConnectionForSender(webContentsId, getTarget, ensureBackend) {
  return resolveTerminalConnection(
    () => getTarget(webContentsId),
    () => ensureBackend(webContentsId)
  )
}

/** A second before-quit must still wait for an in-flight remote kill.
 *
 *  teardownSshConnection deletes the sshConnections entry first, then
 *  SSH-execs kill. backendShutdown's finally() calls app.quit() and
 *  re-enters before-quit with an empty map. Without `inFlight`, Electron
 *  exits while disconnect is running and the detached serve --isolated
 *  stays at pid 1 (post-#95085 leftover on #91668: window X on Windows). */
function sshQuitShouldBlock({
  teardownDone,
  connectionCount,
  bootstrapPending,
  inFlight
}: {
  teardownDone: boolean
  connectionCount: number
  bootstrapPending: number
  inFlight: Promise<unknown> | null
}): boolean {
  if (teardownDone) {
    return false
  }

  return connectionCount > 0 || bootstrapPending > 0 || Boolean(inFlight)
}

async function teardownSshState(state, { cleanupRemote }) {
  // Remote process first, while the SSH channel can still exec kill.
  // Then drop the local forward and close the transport. Each step is
  // best-effort so a failed remote cleanup cannot trap Cmd+Q (#91668).
  try {
    await cleanupRemote(state.ssh, state.ownershipId)
  } catch {
    // Remote teardown is best-effort; always release the local tunnel and SSH transport.
  }

  try {
    if (state.localPort && state.remotePort) {
      await state.ssh.cancelForward(state.localPort, state.remotePort)
    }
  } catch {
    // Best effort; closing the transport below drops any remaining forwards.
  }

  try {
    await state.ssh.close()
  } catch {
    // The app must still be able to quit when SSH teardown fails.
  }
}

export {
  applyConnectionChange,
  commitConnectionFailure,
  resolveTerminalConnection,
  resolveTerminalConnectionForSender,
  sshQuitShouldBlock,
  teardownSshState
}

export async function rollbackSshBootstrap(ssh, result, profile, boundaryError, { stopWindows, stopPosix, forget }) {
  const cleanupErrors: string[] = []

  try {
    await stopOwnedSshDashboard(ssh, result, profile, { stopWindows, stopPosix })
  } catch (error: any) {
    cleanupErrors.push(String(error?.message || error))
  }

  try {
    await ssh.cancelForward(result.localPort, result.remotePort)
  } catch (error: any) {
    cleanupErrors.push(String(error?.message || error))
  }

  try {
    await ssh.close()
  } catch (error: any) {
    cleanupErrors.push(String(error?.message || error))
  }

  forget()

  if (cleanupErrors.length > 0) {
    const unsafe: any = new Error(
      `An SSH bootstrap crossed the managed-update gate and its exact owned serve could not be fenced: ${cleanupErrors.join('; ')}`
    )

    unsafe.code = 'managed-update-bootstrap-fence-failed'
    unsafe.unsafeManagedBootstrap = true
    unsafe.cause = boundaryError
    throw unsafe
  }
}


/** Shared owners are attached; only legacy desktop-owned dashboards may be killed. */
export async function stopOwnedSshDashboard(ssh, state, profile, { stopWindows, stopPosix }) {
  if (state.gatewayEndpoint) {
    state.dispose?.()

    return
  }

  const expected = {
    ownershipId: state.ownershipId, pid: state.pid, spawnNonce: state.spawnNonce, profile,
    hermesPath: state.hermesPath, hermesHome: state.hermesHome, startedAt: state.startedAt,
    creationTimeNs: state.creationTimeNs, creationTime: state.creationTime
  }

  const platform = state.remotePlatform || state.platform?.os

  if (platform === 'Windows') {
    await stopWindows(ssh, { hermesPath: state.hermesPath, hermesHome: state.hermesHome, python: state.pythonPath }, expected)
  } else if (platform === 'Linux' || platform === 'Darwin') {
    await stopPosix(ssh, expected)
  } else {
    throw new Error(`Unsupported remote platform ${platform || 'unknown'}`)
  }
}
