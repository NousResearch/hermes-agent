/**
 * Real-process fixtures for the update-marker suites: sleeping owners, reaped
 * pids, and a claimant child that publishes a marker naming ITSELF (its own
 * pid and kernel creation time), the way a hand-off script does.
 */

import fs from 'fs'
import { type ChildProcess, spawn } from 'node:child_process'
import { once } from 'node:events'
import os from 'os'
import path from 'path'

import { formatCreateTime, processCreateTime } from './update-marker'

const homes: string[] = []
const children: ChildProcess[] = []

/** Call from afterEach. */
export function cleanupMarkerFixtures(): void {
  for (const child of children.splice(0)) {
    child.kill('SIGKILL')
  }

  for (const home of homes.splice(0)) {
    fs.rmSync(home, { recursive: true, force: true })
  }
}

export function tmpHome(tag: string): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `hermes-marker-${tag}-`))
  homes.push(dir)

  return dir
}

/** A real, live owner process (sleeps until killed). */
export async function liveOwner(): Promise<ChildProcess & { pid: number }> {
  const child = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { stdio: 'ignore' })
  children.push(child)
  await once(child, 'spawn')

  return child as ChildProcess & { pid: number }
}

/** A pid that existed and is now gone (reaped). */
export async function deadPid(): Promise<number> {
  const child = spawn(process.execPath, ['-e', ''], { stdio: 'ignore' })
  await once(child, 'exit')

  return child.pid as number
}

export function nowSeconds(): number {
  return Math.floor(Date.now() / 1000)
}

export function minutesAgo(minutes: number): number {
  return nowSeconds() - minutes * 60
}

export interface ClaimantOptions {
  startedAt: number
  run?: string | null
  /** Record the claimant's creation time (v2). Default true. */
  withCt?: boolean
  /** Shift the recorded ct (simulate a reused pid). */
  ctOffset?: number
}

/**
 * A real claimant process that publishes `<its pid>\n<startedAt>\nct:<its ct>\n
 * [run:<run>\n]` into `file` itself, then stays alive until killed.
 */
export async function claimantChild(
  file: string,
  { startedAt, run = null, withCt = true, ctOffset = 0 }: ClaimantOptions
): Promise<{ child: ChildProcess & { pid: number }; body: string }> {
  const child = spawn(
    process.execPath,
    [
      '-e',
      `process.stdin.once('data', d => { require('fs').writeFileSync(${JSON.stringify(file)}, d.toString()); process.stdout.write('published\\n') }); setInterval(() => {}, 1000)`
    ],
    { stdio: ['pipe', 'pipe', 'ignore'] }
  )

  children.push(child)
  await once(child, 'spawn')
  const pid = child.pid as number
  const ct = withCt ? await processCreateTime(pid) : null

  if (withCt && ct === null) {
    throw new Error(`no creation time for claimant ${pid}`)
  }

  const body = `${pid}\n${startedAt}\n${ct === null ? '' : `ct:${formatCreateTime(ct + ctOffset)}\n`}${run ? `run:${run}\n` : ''}`
  const published = once(child.stdout!, 'data')
  child.stdin!.write(body)
  await published

  return { child: child as ChildProcess & { pid: number }, body }
}

/** SIGKILL a child and wait until it is reaped. */
export async function killAndReap(child: ChildProcess): Promise<void> {
  const exited = child.exitCode === null && child.signalCode === null ? once(child, 'exit') : Promise.resolve()
  child.kill('SIGKILL')
  await exited
}
