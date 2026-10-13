/**
 * Inline per-profile MCP setup: the `mcp.servers.*` RPC wrapper, its
 * feature-detect, and the button a capability row renders.
 *
 * Shared leaf: the advanced profile editor and the create dialog both render
 * the button, so it lives below both.
 */

import { Button, host, Input, useI18n } from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'

import { useBots } from './i18n'
import type { ProfileRoute } from './types'

// -- inline MCP setup (per-profile), driven by the mcp.servers.* gateway RPCs --
// Feature-detected: if the gateway behind the TARGET doesn't know those RPCs
// the setup button hides and the row falls back to the "run hermes mcp /
// Settings" hint. Every request rides the captured setup target — the bot's
// own connection and the backend profile name its config lives in — never the
// foreground socket.

/** Body of an `mcp.servers.*` reply. Some gateway builds wrap it in a second
 *  `result` envelope, which every call site below unwraps — hence the
 *  self-reference. */
interface McpServerPayload {
  auth_url?: string
  error?: string
  error_message?: string
  ok?: boolean
  result?: McpServerPayload
  session_id?: string
  status?: string
  verification_url?: string
}

/** `mcpRpc`'s outcome. `unsupported` separates an older gateway that doesn't
 *  know the method from a real failure. */
interface McpRpcResult {
  error?: string
  ok: boolean
  result?: McpServerPayload
  unsupported?: boolean
}

/** Where one MCP setup is pinned. `route` is the bot's full route descriptor
 *  (null = plain local bot on the active gateway); `profile` is the backend
 *  profile NAME sent as `params.profile` — never a route descriptor object. */
export interface McpSetupTarget {
  route: null | ProfileRoute
  profile: string
}

/** Identity of a target for stale-completion checks: any field changing means
 *  the in-flight setup belongs to a different (connection, profile). */
function targetIdentity(target: null | undefined | McpSetupTarget): string {
  if (!target) {
    return 'unavailable'
  }

  const route = target.route
    ? `${target.route.connectionId}::${target.route.profile}::${target.route.targetProfile}`
    : 'ambient'

  return `${route}::${target.profile}`
}

async function mcpRpc(
  target: McpSetupTarget,
  method: string,
  params: Record<string, unknown>
): Promise<McpRpcResult> {
  // Returns { ok, result } or { ok:false, unsupported:true } when the gateway
  // doesn't know the method (older backend) vs a real error. The wire always
  // carries the backend profile NAME; an explicit setup action dials
  // foreground so a cold backend spawn is not queued behind passive work.
  const payload = { ...params, profile: target.profile }

  try {
    let res: McpServerPayload

    if (target.route) {
      if (typeof host.requestProfile !== 'function') {
        // Fail closed: an older shell without routed requests must not
        // silently execute a credential write against whichever gateway
        // happens to be active.
        throw new Error(`Cannot route ${method} for ${target.route.connectionId}::${target.route.profile}`)
      }

      res = await host.requestProfile<McpServerPayload>(target.route, method, payload, undefined, {
        spawnPriority: 'foreground'
      })
    } else {
      res = await host.request<McpServerPayload>(method, payload)
    }

    return {
      ok: true,
      result: res
    }
  } catch (err: any) {
    const msg = String((err && err.message) || err || '')

    if (/unknown method/i.test(msg)) {
      return {
        ok: false,
        unsupported: true
      }
    }

    return {
      ok: false,
      error: msg
    }
  }
}

/** The reply body, one unwrapping deeper when a build double-wraps it. */
function payloadBody(res: McpServerPayload | undefined): McpServerPayload | undefined {
  return res?.result && typeof res.result === 'object' ? res.result : res
}

/** True only when the (possibly double-wrapped) body affirms `ok: true` — the
 *  gateway contract carries `ok` in every `mcp.servers.*` result, so anything
 *  else is unverified, not success. A fulfilled RPC whose body is `ok: false`
 *  is a FAILED configuration, never a success; like the routing above, an
 *  unrecognizable envelope fails closed rather than claiming a credential
 *  write landed. */
function payloadOk(res: McpServerPayload | undefined): boolean {
  return payloadBody(res)?.ok === true
}

function payloadError(res: McpServerPayload | undefined): string {
  const body = payloadBody(res)

  return String(body?.error || body?.error_message || '')
}

/** One row of the capability pane's MCP list (catalog entry or installed server). */
interface McpCatalogEntry {
  auth?: null | string
  fromCatalog?: boolean
  installed?: boolean
  name: string
  requires?: string[]
}

interface McpSetupButtonProps {
  /** Create flow: materialize the profile and return its complete setup
   *  target (the selected connection plus the created slug). Null retires the
   *  setup — the selection moved or creation was refused. */
  ensureTarget?: () => Promise<null | McpSetupTarget>
  entry: McpCatalogEntry
  onDone?: () => void
  /** Pinned setup target for an existing profile (Edit Profile). Null when the
   *  bot's owning connection is gone: setup reports an unavailable target and
   *  never borrows the ambient gateway for a credential write. */
  target?: null | McpSetupTarget
}

export function McpSetupButton({ target, entry, onDone, ensureTarget }: McpSetupButtonProps) {
  const { t } = useI18n()
  const b = useBots()
  // entry: { name, requires:[env keys], auth?, fromCatalog, installed }
  // target is null at first for an orphaned row; the create dialog pairs a
  // planned target with ensureTarget(), which materializes the profile on the
  // first setup action and hands back the complete target, so OAuth / API-key
  // setup works DURING creation, not only in Edit.
  const [phase, setPhase] = useState<'busy' | 'done' | 'error' | 'idle' | 'keys' | 'oauth'>('idle') // idle | keys | oauth | busy | done | error
  const [supported, setSupported] = useState<boolean | null>(null)
  const [keyValues, setKeyValues] = useState<Record<string, string>>({})
  const [message, setMessage] = useState('')
  // The ONE captured target of the running setup (catalog add, every key
  // write, test, completion) — captured at setup start and reused verbatim,
  // tagged with the identity it was captured under.
  const opTargetRef = useRef<null | { identity: string; target: McpSetupTarget }>(null)

  const identity = targetIdentity(target)
  const identityRef = useRef(identity)

  identityRef.current = identity
  const targetRef = useRef(target)

  targetRef.current = target
  const unmountedRef = useRef(false)

  useEffect(
    () => () => {
      unmountedRef.current = true
    },
    []
  )

  // A continuation is stale when the component is gone or the target moved
  // on: it must not touch state, run later steps, or complete onto the new
  // owner.
  const stale = (capturedIdentity: string) =>
    unmountedRef.current || identityRef.current !== capturedIdentity

  // Target replaced: retire the running flow and clear typed key values.
  // A captured target is only usable while its captured identity is current.
  useEffect(() => {
    setPhase('idle')
    setMessage('')
    setKeyValues(prev => (Object.keys(prev).length ? {} : prev))
  }, [identity])

  // Capture one immutable target for a whole setup run. The create flow's
  // selection is read before creation awaits; if the target moved on by the
  // time it settles, the run retires instead of guessing a new owner.
  const captureTarget = async (): Promise<
    { status: 'ok'; captured: string; target: McpSetupTarget } | { status: 'retired' } | { status: 'unavailable' }
  > => {
    const captured = identityRef.current

    if (ensureTarget) {
      const resolved = await ensureTarget()

      if (!resolved) {
        return { status: 'retired' }
      }

      return stale(captured) ? { status: 'retired' } : { captured, status: 'ok', target: resolved }
    }

    const base = targetRef.current

    if (!base) {
      return { status: 'unavailable' }
    }

    return { captured, status: 'ok', target: { ...base } }
  }

  // Support probe: per target BACKEND (the route), cancelled on replacement —
  // one backend's answer never hides setup on another. Only a confirmed
  // unknown method means unsupported; a transient failure stays retryable.
  const routeIdentity = target?.route
    ? `${target.route.connectionId}::${target.route.profile}::${target.route.targetProfile}`
    : target
      ? 'ambient'
      : 'unavailable'

  useEffect(() => {
    const probeTarget = targetRef.current

    if (!probeTarget) {
      setSupported(null)

      return
    }

    let cancelled = false

    setSupported(null)
    void mcpRpc(probeTarget, 'mcp.servers.list', {}).then(result => {
      if (cancelled) {
        return
      }

      setSupported(result.unsupported ? false : true)
    })

    return () => {
      cancelled = true
    }
  }, [routeIdentity])
  const isOAuth = (entry.auth || '').toLowerCase() === 'oauth'
  const requires = entry.requires || []

  const beginKeys = async () => {
    // Ensure the server exists in the target profile first (add from catalog).
    setPhase('busy')
    setMessage('')
    const capture = await captureTarget()

    if (capture.status === 'unavailable') {
      setPhase('error')
      setMessage(b.tools.noTarget)

      return
    }

    if (capture.status !== 'ok') {
      setPhase('idle')

      return
    }

    const { captured, target: opTarget } = capture

    if (entry.fromCatalog && !entry.installed) {
      const add = await mcpRpc(opTarget, 'mcp.servers.add', {
        name: entry.name,
        preset: entry.name
      })

      if (stale(captured)) {
        return
      }

      if (!add.ok || !payloadOk(add.result)) {
        setPhase('error')
        setMessage(add.error || payloadError(add.result) || b.tools.addServerFailed)

        return
      }
    }

    opTargetRef.current = { identity: captured, target: opTarget }
    setPhase('keys')
  }

  const submitKeys = async () => {
    const record = opTargetRef.current

    if (!record || record.identity !== identityRef.current) {
      setPhase('error')
      setMessage(b.tools.noTarget)

      return
    }

    const opTarget = record.target
    const captured = record.identity

    setPhase('busy')
    // An accepted key write is never rolled back: a later failure reports how
    // much landed (never the key values themselves).
    let saved = 0

    for (const k of requires) {
      const val = (keyValues[k] || '').trim()

      if (!val) {
        continue
      }

      const r = await mcpRpc(opTarget, 'mcp.servers.set_api_key', {
        name: entry.name,
        env_var: k,
        value: val
      })

      if (stale(captured)) {
        return
      }

      if (!r.ok || !payloadOk(r.result)) {
        setPhase('error')
        const failure = r.error || payloadError(r.result) || b.tools.setKeyFailed(k)

        setMessage(saved > 0 ? `${failure} · ${b.tools.setKeyPartial(saved)}` : failure)

        return
      }

      saved += 1
    }

    // Verify via test.
    const probe = await mcpRpc(opTarget, 'mcp.servers.test', {
      name: entry.name
    })

    if (stale(captured)) {
      return
    }

    if (probe.ok && payloadOk(probe.result)) {
      setPhase('done')
      host.notify({
        kind: 'success',
        message: b.tools.configured(entry.name)
      })
      onDone && onDone()
    } else {
      setPhase('error')
      setMessage(payloadError(probe.result) || probe.error || b.tools.testFailed)
    }
  }

  const beginOAuth = async () => {
    setPhase('busy')
    setMessage('')
    const capture = await captureTarget()

    if (capture.status === 'unavailable') {
      setPhase('error')
      setMessage(b.tools.noTarget)

      return
    }

    if (capture.status !== 'ok') {
      setPhase('idle')

      return
    }

    const { captured, target: opTarget } = capture

    opTargetRef.current = { identity: captured, target: opTarget }

    // Pinned to the captured target's scope — the same {connectionId, profile}
    // contract completeMcpOAuth routes its own catalog add through.
    const scope = {
      connectionId: opTarget.route?.connectionId ?? host.state.connectionId.get(),
      profile: opTarget.profile
    }

    try {
      setPhase('oauth')
      setMessage(b.tools.completeSignIn)
      await host.completeMcpOAuth({
        serverName: entry.name,
        profile: scope,
        catalogPreset: entry.fromCatalog && !entry.installed ? entry.name : undefined,
        cancelled: () => stale(captured)
      })

      if (stale(captured)) {
        return
      }

      setPhase('done')
      host.notify({ kind: 'success', message: b.tools.authenticated(entry.name) })
      onDone?.()
    } catch (error) {
      if (stale(captured)) {
        return
      }

      setPhase('error')
      setMessage(error instanceof Error ? error.message : String(error))
    }
  }

  if (supported === false) {
    return (
      <span className="ml-1.5 text-[0.65rem] text-(--ui-text-quaternary)">
        {b.tools.needsSetup(requires.join(', '))}
      </span>
    )
  }

  if (phase === 'done') {
    return <span className="ml-1.5 text-[0.65rem] text-(--ui-success)">{b.tools.setUpDone}</span>
  }

  if (phase === 'keys') {
    return (
      <div className="mt-1 grid gap-1">
        {requires.map(k => (
          <Input
            className="h-6 text-[0.7rem]"
            key={k}
            onChange={e =>
              setKeyValues(prev => ({
                ...prev,
                [k]: e.target.value
              }))
            }
            placeholder={k}
            type="password"
            value={keyValues[k] || ''}
          />
        ))}
        <div className="flex gap-1">
          <Button onClick={() => void submitKeys()} size="xs" variant="secondary">
            {b.tools.saveTest}
          </Button>
          <Button onClick={() => setPhase('idle')} size="xs" variant="ghost">
            {t.common.cancel}
          </Button>
        </div>
      </div>
    )
  }

  if (phase === 'oauth') {
    return <span className="ml-1.5 text-[0.65rem] text-(--ui-text-quaternary)">{message || b.tools.authorizing}</span>
  }

  if (phase === 'busy') {
    return <span className="ml-1.5 text-[0.65rem] text-(--ui-text-quaternary)">{b.tools.working}</span>
  }

  if (phase === 'error') {
    return (
      <span className="ml-1.5 text-[0.65rem] text-(--ui-danger,#f87171)">
        {(message || b.tools.setupFailed) + ' '}
        <Button className="underline" onClick={() => setPhase('idle')} size="inline" variant="link">
          {t.common.retry}
        </Button>
      </span>
    )
  }

  // idle
  return (
    <Button
      className="ml-1.5 text-[0.65rem] text-(--ui-accent) underline"
      onClick={() => void (isOAuth ? beginOAuth() : beginKeys())}
      size="inline"
      variant="link"
    >
      {isOAuth ? b.tools.signIn : b.tools.setUp}
    </Button>
  )
}
