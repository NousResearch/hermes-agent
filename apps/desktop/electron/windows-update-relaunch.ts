import fs from 'node:fs'
import path from 'node:path'

const ACK_ARG = '--hermes-update-relaunch-ack='
const REQUEST_ARG = '--hermes-update-relaunch-request='

export type UpdateRelaunchAckRequest = { ackPath: string; requestId: string }

export function parseUpdateRelaunchAckRequest(argv: string[], hermesHome: string): UpdateRelaunchAckRequest | null {
  const rawAck = argv.find(arg => arg.startsWith(ACK_ARG))?.slice(ACK_ARG.length).trim()
  const requestId = argv.find(arg => arg.startsWith(REQUEST_ARG))?.slice(REQUEST_ARG.length).trim()

  if (!rawAck || !requestId || !/^[a-f0-9]{32}$/i.test(requestId)) {
    return null
  }

  const ackPath = path.resolve(rawAck)
  const logsRoot = path.resolve(hermesHome, 'logs')
  const relativeAckPath = path.relative(logsRoot, ackPath)

  if (
    !relativeAckPath ||
    relativeAckPath === '..' ||
    relativeAckPath.startsWith(`..${path.sep}`) ||
    path.isAbsolute(relativeAckPath)
  ) {
    return null
  }

  return { ackPath, requestId }
}

/** Cold-start ACK consumed by the updater before it exits its own job chain. */
export function acknowledgeUpdateRelaunch(argv: string[], hermesHome: string, pid = process.pid): boolean {
  const request = parseUpdateRelaunchAckRequest(argv, hermesHome)

  if (!request || !Number.isInteger(pid) || pid <= 0) {
    return false
  }

  try {
    fs.mkdirSync(path.dirname(request.ackPath), { recursive: true })
    fs.writeFileSync(
      request.ackPath,
      JSON.stringify({ ok: true, pid, requestId: request.requestId, readyAt: new Date().toISOString() }),
      'utf8'
    )

    return true
  } catch {
    return false
  }
}
