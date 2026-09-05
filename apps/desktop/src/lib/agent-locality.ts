import { $connection } from '@/store/session'

/**
 * Whether the agent runs on a different computer than this app.
 *
 * Almost every Desktop-gated tool acts on the machine the user is looking at:
 * the agent asks the renderer over the gateway bridge and the renderer answers
 * for this computer. `computer_use` is the exception — it drives a cua-driver
 * process on whatever host the gateway runs on. Those are the same computer on
 * a local backend and two different computers on an SSH, URL, or cloud one.
 *
 * Only the client can answer this. The backend behind an SSH tunnel sees a
 * loopback peer whether or not the person is in the room, so a server-side
 * guess is wrong in exactly the direction that matters.
 */
export function isAgentOnAnotherMachine(connection = $connection.get()): boolean {
  return connection?.mode === 'remote'
}

/** Names the host the agent runs on, for copy that must not say "this Mac".
 *  Empty when the agent is already on this machine. */
export function agentMachineLabel(connection = $connection.get()): string {
  if (!connection || !isAgentOnAnotherMachine(connection)) {
    return ''
  }

  if (connection.remoteKind === 'cloud') {
    return 'Hermes Cloud'
  }

  const identity = connection.remoteKind === 'ssh' ? connection.remoteIdentity || connection.remoteHost : undefined

  return identity || connection.remoteHost || hostOf(connection.baseUrl) || 'the connected backend'
}

/** `https://nas.local:9119` reads as a machine once the scheme is off. */
function hostOf(baseUrl: string | undefined): string {
  if (!baseUrl) {
    return ''
  }

  try {
    return new URL(baseUrl).host || baseUrl
  } catch {
    return baseUrl
  }
}

/**
 * Tell the agent whether the window we just enumerated is one it can reach.
 *
 * Answered when the agent asks rather than stamped on the session: a transcript
 * outlives whoever was watching when it was written, and the same session can
 * be reopened from another machine. Only the flag crosses — the backend names
 * itself, so no host or connection detail leaves the client.
 *
 * Local sessions carry nothing, so the common case costs no tokens.
 */
export function withAgentLocality(result: unknown, connection = $connection.get()): unknown {
  if (!result || typeof result !== 'object' || !isAgentOnAnotherMachine(connection)) {
    return result
  }

  return { ...result, agent_on_this_machine: false }
}
