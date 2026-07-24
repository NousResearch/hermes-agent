/**
 * Console forwarder — mirrors renderer console.* calls into desktop.log via
 * IPC. Always on: every console.log/warn/error/info/debug call is forwarded
 * to the main process, which writes it through rememberLog() so it lands in
 * desktop.log alongside the main process's own [hermes] lines.
 *
 * Objects/arrays are JSON-serialized (best-effort); non-serializable values
 * fall back to their toString(). Multiple args are space-joined. This is a
 * one-way fire-and-forget (ipcRenderer.send, no await) so it never blocks the
 * renderer or affects call timing.
 *
 * The original console methods are preserved — devtools still shows the full
 * object inspection experience. This only adds a parallel write to desktop.log.
 */

type ConsoleMethod = 'log' | 'warn' | 'error' | 'info' | 'debug'

const LEVEL_MAP: Record<ConsoleMethod, string> = {
  log: 'info',
  warn: 'warn',
  error: 'error',
  info: 'info',
  debug: 'debug'
}

function serializeArg(arg: unknown): string {
  if (arg === null) {
    return 'null'
  }

  if (arg === undefined) {
    return 'undefined'
  }

  if (typeof arg === 'string') {
    return arg
  }

  if (typeof arg === 'number' || typeof arg === 'boolean' || typeof arg === 'bigint') {
    return String(arg)
  }

  if (arg instanceof Error) {
    return `${arg.name}: ${arg.message}${arg.stack ? `\n${arg.stack}` : ''}`
  }

  // Objects/arrays — try JSON, fall back to String.
  try {
    return JSON.stringify(arg)
  } catch {
    try {
      return String(arg)
    } catch {
      return '[unserializable]'
    }
  }
}

function forward(level: ConsoleMethod, args: unknown[]): void {
  const forwarder = window.hermesDesktop?.forwardConsole

  if (!forwarder) {
    return
  }

  const message = args.map(serializeArg).join(' ')

  // Cap at 4KB per line — a single huge object dump shouldn't flood desktop.log.
  const capped = message.length > 4096 ? `${message.slice(0, 4096)}…(${message.length} chars)` : message

  forwarder(LEVEL_MAP[level], capped)
}

if (typeof window !== 'undefined') {
  for (const method of Object.keys(LEVEL_MAP) as ConsoleMethod[]) {
    const original = console[method].bind(console)

    console[method] = (...args: unknown[]) => {
      original(...args)

      try {
        forward(method, args)
      } catch {
        // Forwarding must never break the original call or throw in the
        // renderer — it already ran above.
      }
    }
  }
}
