'use strict'
const { execFile } = require('node:child_process')

const PLAYER_TITLE = 'Hermes Player Lab — Service Guest'
const ADDRESS_RE = /^0x[\da-f]+$/i
const finitePair = value => {
  if (!Array.isArray(value) || value.length !== 2) return null
  const pair = value.map(Number)
  return pair.every(Number.isFinite) ? pair : null
}

function runHyprctl(args, env = process.env) {
  return new Promise(resolve => {
    execFile(
      'hyprctl',
      args,
      {
        env,
        shell: false,
        windowsHide: true,
        timeout: 2500,
        maxBuffer: 1024 * 1024
      },
      (error, stdout = '', stderr = '') => {
        const text = String(stdout || stderr).trim()
        if (error && !text) return resolve(null)
        resolve(text)
      }
    )
  })
}
function supported(env = process.env) {
  return process.platform === 'linux' && Boolean(env.HYPRLAND_INSTANCE_SIGNATURE && env.WAYLAND_DISPLAY)
}
function parseClientList(raw, pid, title) {
  let clients
  try {
    clients = JSON.parse(raw)
  } catch {
    return { status: 'invalid-client-list' }
  }
  if (!Array.isArray(clients)) return { status: 'invalid-client-list' }
  const matches = clients.filter(c => c && c.pid === pid && c.title === title)
  if (matches.length === 0) return { status: 'not-found' }
  if (matches.length !== 1) return { status: 'ambiguous' }
  const client = matches[0]
  if (typeof client.address !== 'string' || !ADDRESS_RE.test(client.address)) return { status: 'invalid-address' }
  return {
    status: 'found',
    client: {
      address: client.address.toLowerCase(),
      pid: client.pid,
      title: client.title,
      floating: client.floating === true,
      pinned: client.pinned === true,
      fullscreen: Number.isInteger(client.fullscreen) ? client.fullscreen : null,
      fullscreenClient: Number.isInteger(client.fullscreenClient) ? client.fullscreenClient : null,
      at: finitePair(client.at),
      size: finitePair(client.size),
      monitor: Number.isInteger(client.monitor) ? client.monitor : null,
      workspaceId: Number.isInteger(client.workspace?.id) ? client.workspace.id : null
    }
  }
}
function classify(reply) {
  if (reply == null) return 'failed'
  const trimmed = reply.trim()
  if (trimmed === 'ok') return 'ok'
  if (trimmed.startsWith('Invalid dispatcher') || trimmed.startsWith('error:')) return 'wrong-syntax'
  return 'failed'
}
function commandArgs(action, address, enabled, syntax) {
  if (!ADDRESS_RE.test(address)) throw new Error('invalid-address')
  const selector = `address:${address}`
  if (action === 'float')
    return syntax === 'legacy'
      ? ['dispatch', 'setfloating', selector]
      : ['dispatch', `hl.dsp.window.float({ action = "${enabled ? 'enable' : 'disable'}", window = "${selector}" })`]
  const state = enabled ? 'enable' : 'disable'
  return syntax === 'legacy'
    ? ['dispatch', 'pin', selector]
    : ['dispatch', `hl.dsp.window.pin({ action = "${state}", window = "${selector}" })`]
}
async function inspect(request, pid, title) {
  const raw = await request(['-j', 'clients'])
  if (!raw) return { status: 'query-failed' }
  return parseClientList(raw, pid, title)
}
async function applyHyprlandPlayerPolicy({
  enabled,
  pid = process.pid,
  title = PLAYER_TITLE,
  env = process.env,
  request = args => runHyprctl(args, env),
  attempts = 18,
  delayMs = 100
}) {
  if (!supported(env))
    return { requested: true, supported: false, status: 'unsupported', reason: 'Hyprland Wayland session not detected' }
  let found
  for (let n = 0; n < attempts; n++) {
    found = await inspect(request, pid, title)
    if (found.status === 'found' || ['ambiguous', 'invalid-client-list', 'invalid-address'].includes(found.status))
      break
    if (n + 1 < attempts) await new Promise(resolve => setTimeout(resolve, delayMs))
  }
  if (!found || found.status !== 'found')
    return { requested: true, supported: true, status: found?.status || 'not-found' }
  let client = found.client
  if (!client.floating) {
    const floated = await mutateOwnClient({
      request,
      pid,
      title,
      address: client.address,
      field: 'floating',
      expected: false,
      action: 'float',
      enabled: true
    })
    if (floated.status !== 'applied')
      return {
        requested: true,
        supported: true,
        status: 'float-failed',
        dispatch: floated.status,
        observation: floated.observation || client
      }
    client = floated.observation
  }
  if (client.pinned !== enabled) {
    const pinned = await mutateOwnClient({
      request,
      pid,
      title,
      address: client.address,
      field: 'pinned',
      expected: client.pinned,
      action: 'pin',
      enabled
    })
    if (pinned.status !== 'applied')
      return {
        requested: true,
        supported: true,
        status: 'pin-failed',
        dispatch: pinned.status,
        observation: pinned.observation || client
      }
    client = pinned.observation
  }
  return { requested: true, supported: true, status: 'applied', observation: client }
}
async function observeHyprlandPlayer({
  pid = process.pid,
  title = PLAYER_TITLE,
  env = process.env,
  request = args => runHyprctl(args, env)
} = {}) {
  if (!supported(env)) return { requested: false, supported: false, status: 'unsupported' }
  const found = await inspect(request, pid, title)
  return found.status === 'found'
    ? { requested: true, supported: true, status: 'observed', observation: found.client }
    : { requested: true, supported: true, status: found.status }
}
const sameGeometry = (a, b) =>
  Boolean(
    a?.at &&
    a?.size &&
    b?.at &&
    b?.size &&
    a.monitor === b.monitor &&
    a.workspaceId === b.workspaceId &&
    a.at.every((v, i) => v === b.at[i]) &&
    a.size.every((v, i) => v === b.size[i])
  )
async function exactOwnClient(request, pid, title, expectedAddress = null) {
  const found = await inspect(request, pid, title)
  if (found.status !== 'found') return found
  if (expectedAddress && found.client.address !== expectedAddress) return { status: 'address-changed' }
  return found
}
async function mutateOwnClient({ request, pid, title, address, field, expected, action, enabled }) {
  const before = await exactOwnClient(request, pid, title, address)
  if (before.status !== 'found') return { status: before.status }
  if (before.client[field] !== expected) return { status: 'state-changed-before-mutation', field }
  const reply = await request(commandArgs(action, address, enabled, 'lua'))
  const outcome = classify(reply)
  if (outcome !== 'ok') return { status: `${action}-dispatch-${outcome}` }
  const after = await exactOwnClient(request, pid, title, address)
  if (after.status !== 'found' || after.client[field] !== enabled)
    return { status: `${action}-not-observed`, observation: after.client || null }
  return { status: 'applied', observation: after.client }
}
async function dispatchFullscreen({
  request,
  pid,
  title,
  address,
  expectedInternal,
  expectedClient,
  internal,
  client
}) {
  const before = await exactOwnClient(request, pid, title, address)
  if (before.status !== 'found') return { status: before.status }
  const state = before.client
  if (!Number.isInteger(state.fullscreen) || !Number.isInteger(state.fullscreenClient))
    return { status: 'fullscreen-fields-unavailable' }
  if (state.fullscreen !== expectedInternal || state.fullscreenClient !== expectedClient)
    return { status: 'fullscreen-state-changed-before-mutation', observation: state }
  const lua = `hl.dsp.window.fullscreen_state({ action = "set", internal = ${internal}, client = ${client}, window = "address:${address}" })`
  const outcome = classify(await request(['dispatch', lua]))
  if (outcome !== 'ok') return { status: `fullscreen-dispatch-${outcome}` }
  const after = await exactOwnClient(request, pid, title, address)
  if (after.status !== 'found') return { status: after.status }
  return { status: 'dispatched', observation: after.client }
}
function wait(ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}
async function waitForClientState(request, pid, title, address, predicate, timeoutMs = 3000, requiredSamples = 3) {
  const deadline = Date.now() + timeoutMs
  let consecutive = 0
  let last = null
  while (Date.now() < deadline) {
    const found = await exactOwnClient(request, pid, title, address)
    if (found.status !== 'found') return found
    last = found.client
    consecutive = predicate(found.client) ? consecutive + 1 : 0
    if (consecutive >= requiredSamples) return found
    await wait(100)
  }
  return { status: 'not-converged', observation: last }
}
const sameClientState = (a, b) =>
  Boolean(
    a &&
    b &&
    a.address === b.address &&
    a.floating === b.floating &&
    a.pinned === b.pinned &&
    a.fullscreen === b.fullscreen &&
    a.fullscreenClient === b.fullscreenClient &&
    sameGeometry(a, b)
  )
async function waitForStableClient(request, pid, title, address, timeoutMs = 3000, requiredSamples = 3) {
  const deadline = Date.now() + timeoutMs
  let previous = null,
    latest = null,
    consecutive = 0
  while (Date.now() < deadline) {
    const found = await exactOwnClient(request, pid, title, address)
    if (found.status !== 'found') return found
    latest = found.client
    consecutive = sameClientState(previous, latest) ? consecutive + 1 : 1
    if (consecutive >= requiredSamples) return found
    previous = latest
    await wait(100)
  }
  return { status: 'not-stable', observation: latest }
}
async function restorePinSafely({ request, pid, title, address, desiredPinned }) {
  desiredPinned = pinPreference(desiredPinned)
  const current = await exactOwnClient(request, pid, title, address)
  if (current.status !== 'found') return { status: `pin-recovery-${current.status}` }
  if (current.client.fullscreen !== 0 || current.client.fullscreenClient !== 0)
    return { status: 'pin-recovery-deferred-fullscreen' }
  if (current.client.pinned === desiredPinned) return { status: 'pin-already-restored', observation: current.client }
  const result = await mutateOwnClient({
    request,
    pid,
    title,
    address,
    field: 'pinned',
    expected: current.client.pinned,
    action: 'pin',
    enabled: desiredPinned
  })
  return {
    status: result.status === 'applied' ? 'pin-restored' : `pin-recovery-${result.status}`,
    observation: result.observation || null
  }
}
function pinPreference(value) {
  return typeof value === 'function' ? Boolean(value()) : Boolean(value)
}
async function setHyprlandPlayerFullscreen({
  enabled,
  pid = process.pid,
  title = PLAYER_TITLE,
  desiredPinned,
  env = process.env,
  request = args => runHyprctl(args, env),
  timeoutMs = 3000
} = {}) {
  if (!supported(env)) return { supported: false, status: 'unsupported' }
  const observed = await inspect(request, pid, title)
  if (observed.status !== 'found') return { supported: true, status: observed.status }
  const initial = enabled
    ? await waitForStableClient(request, pid, title, observed.client.address, timeoutMs)
    : observed
  if (initial.status !== 'found')
    return { supported: true, status: initial.status, observation: initial.observation || null }
  const address = initial.client.address
  if (enabled) {
    if (initial.client.fullscreen === 2 && initial.client.fullscreenClient === 2)
      return { supported: true, status: 'already-fullscreen', observation: initial.client }
    const snapshot = {
      address,
      floating: initial.client.floating,
      pinned: initial.client.pinned,
      at: initial.client.at,
      size: initial.client.size,
      monitor: initial.client.monitor,
      workspaceId: initial.client.workspaceId
    }
    let temporaryUnpinApplied = false
    if (initial.client.pinned) {
      const unpin = await mutateOwnClient({
        request,
        pid,
        title,
        address,
        field: 'pinned',
        expected: true,
        action: 'pin',
        enabled: false
      })
      if (unpin.status !== 'applied')
        return {
          supported: true,
          status: 'temporary-unpin-failed',
          reason: unpin.status,
          observation: unpin.observation || initial.client,
          pinRecovery: { status: 'not-needed' }
        }
      temporaryUnpinApplied = true
    }
    const target = await exactOwnClient(request, pid, title, address)
    if (target.status !== 'found' || target.client.pinned) {
      const recovery = temporaryUnpinApplied
        ? await restorePinSafely({ request, pid, title, address, desiredPinned: pinPreference(desiredPinned) })
        : { status: 'not-needed' }
      return {
        supported: true,
        status: 'own-client-changed-before-fullscreen',
        reason: target.status || 'still-pinned',
        pinRecovery: recovery
      }
    }
    const dispatched = await dispatchFullscreen({
      request,
      pid,
      title,
      address,
      expectedInternal: 0,
      expectedClient: 0,
      internal: 2,
      client: 2
    })
    if (dispatched.status !== 'dispatched') {
      const recovery = temporaryUnpinApplied
        ? await restorePinSafely({ request, pid, title, address, desiredPinned: pinPreference(desiredPinned) })
        : { status: 'not-needed' }
      return {
        supported: true,
        status: dispatched.status,
        observation: dispatched.observation || null,
        pinRecovery: recovery
      }
    }
    const settled = await waitForClientState(
      request,
      pid,
      title,
      address,
      c => c.fullscreen === 2 && c.fullscreenClient === 2,
      timeoutMs
    )
    if (settled.status !== 'found' || settled.client.fullscreen !== 2 || settled.client.fullscreenClient !== 2) {
      let recovery = { status: 'not-needed' }
      if (temporaryUnpinApplied) {
        const latest = await exactOwnClient(request, pid, title, address)
        if (latest.status === 'found' && latest.client.fullscreen === 2 && latest.client.fullscreenClient === 2) {
          const rollback = await dispatchFullscreen({
            request,
            pid,
            title,
            address,
            expectedInternal: 2,
            expectedClient: 2,
            internal: 0,
            client: 0
          })
          if (rollback.status === 'dispatched') {
            const windowed = await waitForClientState(
              request,
              pid,
              title,
              address,
              c => c.fullscreen === 0 && c.fullscreenClient === 0,
              timeoutMs
            )
            recovery =
              windowed.status === 'found' && windowed.client.fullscreen === 0 && windowed.client.fullscreenClient === 0
                ? await restorePinSafely({ request, pid, title, address, desiredPinned: pinPreference(desiredPinned) })
                : { status: 'fullscreen-rollback-not-converged' }
          } else recovery = { status: `fullscreen-rollback-${rollback.status}` }
        } else if (
          latest.status === 'found' &&
          latest.client.fullscreen === 0 &&
          latest.client.fullscreenClient === 0
        ) {
          recovery = await restorePinSafely({
            request,
            pid,
            title,
            address,
            desiredPinned: pinPreference(desiredPinned)
          })
        } else recovery = { status: 'pin-recovery-refused-unknown-state' }
      }
      return {
        supported: true,
        status: 'fullscreen-enter-not-converged',
        observation: settled.client || settled.observation || null,
        pinRecovery: recovery
      }
    }
    fullscreenSnapshots.set(`${pid}:${title}`, snapshot)
    return {
      supported: true,
      status: 'fullscreen',
      observation: settled.client,
      fullscreenSnapshot: {
        floating: snapshot.floating,
        pinned: snapshot.pinned,
        geometry: { at: snapshot.at, size: snapshot.size, monitor: snapshot.monitor, workspaceId: snapshot.workspaceId }
      }
    }
  }
  const snapshot = fullscreenSnapshots.get(`${pid}:${title}`) || {
    address,
    floating: initial.client.floating,
    pinned: initial.client.pinned,
    at: initial.client.at,
    size: initial.client.size,
    monitor: initial.client.monitor,
    workspaceId: initial.client.workspaceId
  }
  if (snapshot.address !== address) return { supported: true, status: 'address-changed-before-exit' }
  let current = await exactOwnClient(request, pid, title, address)
  if (current.status !== 'found') return { supported: true, status: current.status }
  if (current.client.fullscreen !== 0 || current.client.fullscreenClient !== 0) {
    const exit = await dispatchFullscreen({
      request,
      pid,
      title,
      address,
      expectedInternal: current.client.fullscreen,
      expectedClient: current.client.fullscreenClient,
      internal: 0,
      client: 0
    })
    if (exit.status !== 'dispatched')
      return { supported: true, status: exit.status, observation: exit.observation || null }
    current = await waitForClientState(
      request,
      pid,
      title,
      address,
      c => c.fullscreen === 0 && c.fullscreenClient === 0,
      timeoutMs
    )
  }
  if (current.status !== 'found' || current.client.fullscreen !== 0 || current.client.fullscreenClient !== 0)
    return { supported: true, status: 'fullscreen-exit-not-converged', observation: current.client || null }
  current = await waitForClientState(
    request,
    pid,
    title,
    address,
    c => c.fullscreen === 0 && c.fullscreenClient === 0 && sameGeometry(snapshot, c),
    timeoutMs
  )
  if (current.status !== 'found' || current.client.fullscreen !== 0 || current.client.fullscreenClient !== 0)
    return { supported: true, status: 'fullscreen-exit-state-lost', observation: current.client || null }
  const geometryRestored = sameGeometry(snapshot, current.client)
  if (current.client.floating !== snapshot.floating) {
    const floating = await mutateOwnClient({
      request,
      pid,
      title,
      address,
      field: 'floating',
      expected: current.client.floating,
      action: 'float',
      enabled: snapshot.floating
    })
    if (floating.status !== 'applied')
      return {
        supported: true,
        status: 'floating-state-restore-failed',
        reason: floating.status,
        geometryRestored,
        observation: floating.observation || current.client
      }
    current = { status: 'found', client: floating.observation }
  }
  const desired = pinPreference(desiredPinned)
  if (current.client.pinned !== desired) {
    const pin = await mutateOwnClient({
      request,
      pid,
      title,
      address,
      field: 'pinned',
      expected: current.client.pinned,
      action: 'pin',
      enabled: desired
    })
    if (pin.status !== 'applied')
      return {
        supported: true,
        status: 'pin-preference-restore-failed',
        reason: pin.status,
        geometryRestored,
        observation: pin.observation || current.client
      }
    current = { status: 'found', client: pin.observation }
  }
  fullscreenSnapshots.delete(`${pid}:${title}`)
  return {
    supported: true,
    status: geometryRestored ? 'windowed' : 'windowed-geometry-mismatch',
    geometryRestored,
    observation: current.client,
    fullscreenSnapshot: {
      floating: snapshot.floating,
      pinned: snapshot.pinned,
      geometry: { at: snapshot.at, size: snapshot.size, monitor: snapshot.monitor, workspaceId: snapshot.workspaceId }
    }
  }
}
async function setHyprlandPlayerPin({
  enabled,
  pid = process.pid,
  title = PLAYER_TITLE,
  env = process.env,
  request = args => runHyprctl(args, env)
} = {}) {
  if (!supported(env)) return { supported: false, status: 'unsupported' }
  const found = await inspect(request, pid, title)
  if (found.status !== 'found') return { supported: true, status: found.status }
  const address = found.client.address
  if (found.client.fullscreen !== 0 || found.client.fullscreenClient !== 0)
    return { supported: true, status: 'pin-deferred-fullscreen', observation: found.client }
  if (found.client.pinned === Boolean(enabled))
    return { supported: true, status: 'observed', observation: found.client }
  const result = await mutateOwnClient({
    request,
    pid,
    title,
    address,
    field: 'pinned',
    expected: found.client.pinned,
    action: 'pin',
    enabled: Boolean(enabled)
  })
  return {
    supported: true,
    status: result.status === 'applied' ? 'applied' : result.status,
    observation: result.observation || null
  }
}
const fullscreenSnapshots = new Map()
function createTransitionGate() {
  let active = false
  return {
    isActive: () => active,
    run: async work => {
      if (active) return { ok: false, status: 'busy' }
      active = true
      try {
        return await work()
      } finally {
        active = false
      }
    },
    afterCurrent: async (work, timeoutMs = 10000) => {
      const deadline = Date.now() + timeoutMs
      while (active && Date.now() < deadline) await wait(25)
      if (active) return { ok: false, status: 'busy-timeout' }
      const result = await (async () => {
        if (active) return null
        active = true
        try {
          return await work()
        } finally {
          active = false
        }
      })()
      return result === null ? { ok: false, status: 'busy' } : result
    }
  }
}
module.exports = {
  PLAYER_TITLE,
  applyHyprlandPlayerPolicy,
  observeHyprlandPlayer,
  setHyprlandPlayerFullscreen,
  setHyprlandPlayerPin,
  createTransitionGate
}
