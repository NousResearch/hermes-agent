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
  teardownSshState
}
