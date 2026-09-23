import crypto from 'node:crypto'
import path from 'node:path'

import { fenceManagedSshBootstrapPublication } from './managed-ssh-update'
import { sshConfigFingerprint } from './ssh-bootstrap-coordinator'

// Main owns the live SSH registry and update gate. This runtime only owns the
// bootstrap transaction; callbacks remain bound to the current main state.
export interface DesktopSshBootstrapRuntimeDeps {
  GUEST_ONBOARDING: boolean
  SshConnection: any
  adoptServedDashboardToken: any
  buildRemoteConnection: any
  connectWindowsRemote: any
  detectRemotePlatform: any
  execText: any
  managedConnectionUpdateGate: any
  persistSshConnectionToken: any
  pickLocalPort: any
  remoteLifecycle: any
  resolveRemoteSshDashboardProfile: any
  sshBootstrapCoordinator: any
  sshConnections: Map<string, any>
  sshIsolatedKeepalives: any
  sshOwnershipKey: any
  sshProbeReuseProof: any
  sshRememberLog: any
  sshScopeKey: any
  teardownSshConnection: any
  terminateOwnedWindowsDashboardForUpdate: any
  waitForHermes: any
}

export function createDesktopSshBootstrapRuntime(deps: DesktopSshBootstrapRuntimeDeps) {
  const {
    GUEST_ONBOARDING,
    SshConnection,
    adoptServedDashboardToken,
    buildRemoteConnection,
    connectWindowsRemote,
    detectRemotePlatform,
    execText,
    managedConnectionUpdateGate,
    persistSshConnectionToken,
    pickLocalPort,
    remoteLifecycle,
    resolveRemoteSshDashboardProfile,
    sshBootstrapCoordinator,
    sshConnections,
    sshIsolatedKeepalives,
    sshOwnershipKey,
    sshProbeReuseProof,
    sshRememberLog,
    sshScopeKey,
    teardownSshConnection,
    terminateOwnedWindowsDashboardForUpdate,
    waitForHermes
  } = deps

  async function effectiveSshConfigFingerprint(sshConfig) {
    const ssh =
      process.platform === 'win32'
        ? path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'OpenSSH', 'ssh.exe')
        : 'ssh'

    const args = ['-G']

    if (sshConfig.port) {
      args.push('-p', String(sshConfig.port))
    }

    if (sshConfig.keyPath) {
      args.push('-i', sshConfig.keyPath)
    }

    args.push('--', sshConfig.user ? `${sshConfig.user}@${sshConfig.host}` : sshConfig.host)
    const output = await execText(ssh, args, { timeout: 10_000 })

    return crypto.createHash('sha256').update(output).digest('hex')
  }

  async function bootstrapSshConnection(
    profile,
    sshConfig,
    reuseToken,
    source,
    resolvedEffectiveFingerprint?,
    metadata: any = {}
  ) {
    const scope = sshScopeKey(profile)
    const effectiveConfigFingerprint = resolvedEffectiveFingerprint || (await effectiveSshConfigFingerprint(sshConfig))
    const resolvedConfig = { ...sshConfig, effectiveConfigFingerprint }
    const fingerprint = sshConfigFingerprint(scope, resolvedConfig)

    return sshBootstrapCoordinator.start(
      scope,
      fingerprint,
      lease => bootstrapSshConnectionInner(profile, resolvedConfig, reuseToken, source, metadata, fingerprint, lease),
      metadata
    )
  }

  // Tear down a bootstrap result whose publication lost the managed-update
  // fence: exact-terminate the serve this bootstrap owns (never a foreign one),
  // drop its forward and transport, and surface a fence error so the managed
  // updater refuses to mutate a remote install with an unfenced serve.
  async function rollbackSshBootstrapResult(ssh, result, profile, sshConfig, boundaryError) {
    const cleanupErrors: string[] = []
    const scope = sshScopeKey(profile)

    try {
      const expected = {
        ownershipId: result.ownershipId,
        pid: result.pid,
        spawnNonce: result.spawnNonce,
        profile: resolveRemoteSshDashboardProfile(sshConfig.remoteProfile, profile),
        hermesPath: result.hermesPath,
        hermesHome: result.hermesHome,
        startedAt: result.startedAt,
        creationTimeNs: result.creationTimeNs,
        creationTime: result.creationTime
      }

      if (result.platform?.os === 'Windows') {
        await terminateOwnedWindowsDashboardForUpdate(
          ssh,
          { hermesPath: result.hermesPath, hermesHome: result.hermesHome, python: result.pythonPath },
          expected
        )
      } else if (result.platform?.os === 'Linux' || result.platform?.os === 'Darwin') {
        await remoteLifecycle.terminateOwnedDashboardForUpdate(ssh, expected)
      } else {
        cleanupErrors.push(`unsupported remote platform ${result.platform?.os || 'unknown'}`)
      }
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

    if (sshConnections.get(scope)?.ssh === ssh) {
      sshIsolatedKeepalives.stop(scope)
      sshConnections.delete(scope)
    }

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

  async function bootstrapSshConnectionInner(profile, sshConfig, reuseToken, source, metadata, fingerprint, lease) {
    const scope = sshScopeKey(profile)
    const hostLabel = sshConfig.user ? `${sshConfig.user}@${sshConfig.host}` : sshConfig.host
    const existing = sshConnections.get(scope)

    if (existing && existing.fingerprint !== fingerprint) {
      await teardownSshConnection(profile)
    }

    let ssh = sshConnections.get(scope)?.ssh

    if (ssh && !(await ssh.isAlive())) {
      try {
        await ssh.close()
      } catch {
        void 0
      }

      ssh = null
      sshIsolatedKeepalives.stop(scope)
      sshConnections.delete(scope)
    }

    const created = !ssh

    let removeForceCleanup = () => {}

    if (created) {
      ssh = new SshConnection(
        { host: sshConfig.host, user: sshConfig.user, port: sshConfig.port, keyPath: sshConfig.keyPath },
        {
          rememberLog: sshRememberLog,
          ownershipId: sshOwnershipKey(profile),
          scope,
          effectiveConfigFingerprint: sshConfig.effectiveConfigFingerprint
        }
      )
      removeForceCleanup = lease.onForceCleanup(() => ssh.close())
      await ssh.open({ signal: lease.signal })
    }

    let result: any

    try {
      if (metadata.registryConnectionId) {
        managedConnectionUpdateGate.assertCanDial(
          metadata.registryConnectionId,
          metadata.managedUpdateCorrelation || ''
        )
      }

      const platform = await detectRemotePlatform(ssh, sshConfig.remoteHermesPath || '')
      const lifecycle = platform.os === 'Windows' ? connectWindowsRemote : remoteLifecycle.connect
      result = await lifecycle({
        ssh,
        platform,
        profile: resolveRemoteSshDashboardProfile(sshConfig.remoteProfile, profile),
        remoteHermesPath: sshConfig.remoteHermesPath || '',
        ownershipId: sshOwnershipKey(profile),
        reuseToken: reuseToken || '',
        forward: (localPort, remotePort) => ssh.forward(localPort, remotePort),
        cancelForward: (localPort, remotePort) => ssh.cancelForward(localPort, remotePort),
        pickLocalPort,
        waitForHermes: (baseUrl, token) => waitForHermes(baseUrl, token, lease.signal, 'token'),
        probeReuseProof: sshProbeReuseProof,
        adoptServedToken: adoptServedDashboardToken,
        rememberLog: sshRememberLog,
        // Same launch-time free-tier decision the local spawns get; the POSIX
        // spawn command adds HERMES_GUEST_ONBOARDING=1 only when this is on.
        guestOnboarding: GUEST_ONBOARDING,
        signal: lease.signal
      })
    } catch (error: any) {
      if (created) {
        try {
          await ssh.close()
        } catch {
          void 0
        }
      } else {
        // The cached master was reused but the lifecycle probe against it
        // failed ("Could not verify the existing SSH backend"). Keeping the
        // stale entry means every subsequent boot re-attempts through the same
        // wedged master/tunnel and fails identically until the user re-enters
        // the connection details (whose changed fingerprint forces a teardown).
        // Tear it down now so the next attempt — automatic retry included —
        // bootstraps a fresh master, which is exactly what manual re-entry
        // did (#82679).
        try {
          await teardownSshConnection(profile)
        } catch {
          void 0
        }
      }

      const err = new Error(error.message) as any
      err.sshError = error.kind || 'unknown'
      err.isSshBootstrap = true
      throw err
    }

    try {
      lease.assertCurrent()
    } catch (error) {
      await rollbackSshBootstrapResult(ssh, result, profile, sshConfig, error)
      throw error
    }

    await fenceManagedSshBootstrapPublication({
      assertCanPublish: () => {
        if (metadata.registryConnectionId) {
          managedConnectionUpdateGate.assertCanDial(
            metadata.registryConnectionId,
            metadata.managedUpdateCorrelation || ''
          )
        }
      },
      publish: () => {
        persistSshConnectionToken(profile, source, result.token, metadata.registryConnectionId)
        removeForceCleanup()
        sshConnections.set(scope, {
          ssh,
          fingerprint,
          ownershipId: result.ownershipId || sshOwnershipKey(profile),
          localPort: result.localPort,
          remotePort: result.remotePort,
          pid: result.pid,
          host: sshConfig.host,
          hostLabel,
          hermesVersion: result.hermesVersion || '',
          remotePlatform: result.platform?.os || '',
          reused: result.reused,
          spawnNonce: result.spawnNonce,
          creationTimeNs: result.creationTimeNs,
          creationTime: result.creationTime,
          startedAt: result.startedAt,
          hermesPath: result.hermesPath,
          hermesHome: result.hermesHome,
          pythonPath: result.pythonPath,
          remoteProfile: resolveRemoteSshDashboardProfile(sshConfig.remoteProfile, profile),
          registryConnectionId:
            metadata.registryConnectionId ||
            (typeof source === 'string' && source.startsWith('registry:') ? source.slice('registry:'.length) : ''),
          // Never infer primary ownership from a non-composite scope key: legacy
          // per-profile pools also use bare keys. Only startHermes' explicit call
          // site may label a registry-qualified SSH scope as the primary backend.
          primaryRegistryScope: metadata.primaryRegistryScope === true
        })
        sshIsolatedKeepalives.start(scope, { baseUrl: result.baseUrl, token: result.token })
      },
      rollback: error => rollbackSshBootstrapResult(ssh, result, profile, sshConfig, error)
    })

    sshRememberLog(
      `[ssh] connection ${result.reused ? 'REUSED' : 'spawned'} dashboard: ` +
        `${result.hermesVersion || 'hermes (version unknown)'} at ${result.hermesPath || '?'}`
    )

    const connection = await buildRemoteConnection(
      result.baseUrl,
      'token',
      result.token,
      source,
      hostLabel,
      'ssh',
      result.ownershipId
    )

    return {
      ...connection,
      remoteHermesVersion: result.hermesVersion || '',
      ssh: {
        effectiveConfigFingerprint: sshConfig.effectiveConfigFingerprint,
        host: sshConfig.host,
        keyPath: sshConfig.keyPath,
        port: sshConfig.port,
        remoteHermesPath: sshConfig.remoteHermesPath,
        remoteProfile: sshConfig.remoteProfile,
        user: sshConfig.user
      }
    }
  }

  return { effectiveSshConfigFingerprint, bootstrapSshConnection }
}
