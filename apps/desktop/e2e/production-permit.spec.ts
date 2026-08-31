import { execFileSync, spawn } from 'node:child_process'
import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'

import { type NoProviderFixture, setupNoProvider } from './fixtures'
import { expect, test } from './test'

const REPO_ROOT = path.resolve(import.meta.dirname, '..', '..', '..')
const ARES_ROOT = REPO_ROOT
const RUST_ROOT = path.resolve(REPO_ROOT, '..', 'recursive-agent-production-permit-20260830')

let fixture: NoProviderFixture | null = null

async function waitForSocket(socketPath: string, daemon: ReturnType<typeof spawn>): Promise<void> {
  const deadline = Date.now() + 30_000
  while (Date.now() < deadline) {
    if (fs.existsSync(socketPath)) {
      return
    }
    if (daemon.exitCode !== null) {
      throw new Error(`ra-daemon exited before binding socket: ${daemon.exitCode}`)
    }
    await new Promise(resolve => setTimeout(resolve, 50))
  }
  throw new Error('timed out waiting for ra-daemon socket')
}

test.beforeAll(async () => {
  fixture = await setupNoProvider()
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('Electron main signs an exact permit and Rust accepts the isolated lifecycle', async () => {
  const approval = {
    approval_id: 'approval:electron-e2e-1',
    schema: 'recursive-agent.desktop-production-approval-request/v1',
    mission_ref: 'production-write-mission',
    target_ref: 'path:approved-result',
    call: {
      tool: 'write_file',
      args: {
        path: path.join(ARES_ROOT, '.isolated-electron-e2e.txt'),
        content: 'electron-main-to-rust-daemon'
      },
      frozen_clock: null
    },
    constraints: {
      validity_ms: 300000,
      one_use: true,
      retry_allowed: false,
      network_allowed: false,
      delegation_allowed: false,
      allowed_write_root: ARES_ROOT,
      ambiguous_outcome: 'terminal_quarantine'
    }
  } as const

  const signed = await fixture!.page.evaluate(async envelope => {
    const desktop = (window as typeof window & {
      hermesDesktop?: {
        productionPermit?: {
          publicKey: () => Promise<{
            verifier_enrollment: { schema: string; key_id: string; public_key: string }
          }>
          sign: (request: unknown) => Promise<unknown>
        }
      }
    }).hermesDesktop
    if (!desktop?.productionPermit) {
      throw new Error('production permit preload bridge unavailable')
    }
    return {
      publicKey: await desktop.productionPermit.publicKey(),
      witness: await desktop.productionPermit.sign(envelope)
    }
  }, approval)

  const sandbox = fs.mkdtempSync(path.join(os.tmpdir(), 'ares-electron-permit-'))
  const enrollmentPath = path.join(sandbox, 'verifier.json')
  const runtimeRoot = path.join(sandbox, 'daemon-root')
  const socketPath = path.join(sandbox, 'ra.sock')
  const daemonPath = path.join(RUST_ROOT, 'target', 'debug', 'ra-daemon')
  fs.mkdirSync(runtimeRoot, { mode: 0o700 })
  fs.writeFileSync(enrollmentPath, JSON.stringify(signed.publicKey.verifier_enrollment), { mode: 0o600 })
  fs.chmodSync(enrollmentPath, 0o600)
  if (!fs.existsSync(daemonPath)) {
    execFileSync('cargo', ['build', '-p', 'recursive-agent-daemon', '--bin', 'ra-daemon'], {
      cwd: RUST_ROOT,
      stdio: 'inherit'
    })
  }

  const daemon = spawn(
    daemonPath,
    [
      'serve',
      '--root',
      runtimeRoot,
      '--socket',
      socketPath,
      '--production-verifier-file',
      enrollmentPath
    ],
    { cwd: RUST_ROOT, stdio: 'pipe' }
  )
  try {
    await waitForSocket(socketPath, daemon)
    const script = `
import json, sys
from pathlib import Path
from ares_runtime.collaboration import DaemonPermitReceiptAdapter, PermitBridgeState

class WitnessProvider:
    def __init__(self, witness): self.witness = witness
    def issue_witness(self, **_kwargs): return self.witness

witness = json.loads(sys.argv[1])
socket_path, target_path, content, mission_ref, target_ref = sys.argv[2:]
adapter = DaemonPermitReceiptAdapter(
    {"socket_path": socket_path, "mode": "production_per_call"},
    approval_witness_provider=WitnessProvider(witness),
)
outcome = adapter.consume(
    mission_ref=mission_ref,
    tool_name="write_file",
    args={"path": target_path, "content": content},
    target_ref=target_ref,
)
if outcome.state is not PermitBridgeState.CONSUMED or outcome.facts is None:
    raise SystemExit(f"permit was not consumed: {outcome}")
Path(target_path).write_text(content, encoding="utf-8")
if Path(target_path).read_text(encoding="utf-8") != content:
    raise SystemExit("filesystem readback mismatch")
adapter.record_receipt({
    "permit_ref": outcome.facts["permit_id"],
    "preflight_receipt": outcome.facts["receipt_artifact"],
    "state": "ok",
    "duration_ms": 1,
    "error_type": None,
})
Path(target_path).unlink()
print(json.dumps({"state": outcome.state.value, "permit_id": outcome.facts["permit_id"]}))
`
    const result = execFileSync(
      'uv',
      [
        'run',
        '--extra',
        'dev',
        'python',
        '-c',
        script,
        JSON.stringify(signed.witness),
        socketPath,
        approval.call.args.path,
        approval.call.args.content,
        approval.mission_ref,
        approval.target_ref
      ],
      { cwd: ARES_ROOT, encoding: 'utf8' }
    )
    expect(JSON.parse(result)).toMatchObject({ state: 'consumed' })
    expect(fs.existsSync(approval.call.args.path)).toBe(false)
  } finally {
    daemon.kill('SIGTERM')
    fs.rmSync(sandbox, { force: true, recursive: true })
  }
})
