import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, test, vi } from 'vitest'

vi.mock('./managed-ssh-update', () => ({
  observeManagedRemoteUpdate: vi.fn()
}))

import { installationFingerprint } from './managed-rollout-identity'
import { createManagedRolloutMainIntegration } from './managed-rollout-main-integration'
import { observeManagedRemoteUpdate } from './managed-ssh-update'

const INSTALL_ID = 'a'.repeat(32)
const TARGET_SHA = 'b'.repeat(40)
const ROOT = '/srv/hermes-agent'
const CONNECTION_ID = '11111111-1111-4111-8111-111111111111'
const CORRELATION_ID = '22222222-2222-4222-8222-222222222222'
const REPOSITORY_ID = 'github.com/nousresearch/hermes-agent'

const temporaryDirectories: string[] = []

afterEach(() => {
  vi.restoreAllMocks()

  for (const directory of temporaryDirectories.splice(0)) {fs.rmSync(directory, { recursive: true, force: true })}
})

function makeIntegration(headSha = TARGET_SHA) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'managed-rollout-main-integration-'))
  temporaryDirectories.push(directory)
  let inspectionHeadReads = 0

  const ssh = {
    exec: vi.fn(async (command: string) => {
      if (command.includes("'--show-toplevel'")) {return ROOT}

      if (command.includes("'remote' 'get-url'")) {return 'https://github.com/nousresearch/hermes-agent.git'}

      if (command.includes("'rev-parse' 'HEAD'")) {return inspectionHeadReads++ === 0 ? TARGET_SHA : headSha}

      if (command.includes('echo "${HERMES_HOME:-$HOME/.hermes}"')) {return '~/.hermes'}

      if (command.includes('if [ -f')) {return INSTALL_ID}

      return headSha
    })
  }

  const target = {
    platform: 'Linux',
    hermesPath: `${ROOT}/.venv/bin/hermes`,
    hermesHome: '~/.hermes',
    ssh
  }

  const source = { id: CONNECTION_ID, kind: 'ssh', label: 'test-source' }

  const integration = createManagedRolloutMainIntegration({
    nowMono: () => 1000,
    listSources: () => [source],
    getSource: connectionId => connectionId === CONNECTION_ID ? source : null,
    managedSshConfig: () => ({ user: 'hermes', host: 'source.example.test', port: 22 }),
    openTransport: async () => ({ target, close: async () => undefined }),
    captureScopes: async () => [],
    readHostKeyFingerprint: async () => 'SHA256:host-key',
    effectiveConfigFingerprint: async () => 'config-revision',
    reviewManifestPath: path.join(directory, 'review.json'),
    assuranceRoot: path.join(directory, 'assurance'),
    journalRoot: path.join(directory, 'journal')
  })

  return { integration, source, ssh }
}

function attempt(sourceFingerprint: string, installation: string) {
  return {
    installId: INSTALL_ID,
    installationFingerprint: installation,
    connectionId: CONNECTION_ID,
    sourceFingerprint,
    targetSha: TARGET_SHA,
    reviewedSource: {
      repositoryRoot: ROOT,
      originUrl: 'https://github.com/NousResearch/hermes-agent.git',
      resolvedRef: 'refs/remotes/origin/main',
      targetSha: TARGET_SHA,
      assuranceProfile: 'profile-v1',
      assuranceEvidenceSha256: 'c'.repeat(64),
      assuranceGeneration: 1
    },
    correlationId: CORRELATION_ID,
    wave: 0,
    excluded: false
  }
}

describe('managed rollout main integration', () => {
  test('runs an independent bounded evidence sweep and verifies target HEAD', async () => {
    const { integration } = makeIntegration()
    const snapshot = await integration.adapters.inventoryReader.capture()
    expect(snapshot).not.toBeNull()
    const row: any = snapshot!.observations[0]
    const installation = installationFingerprint({ installId: INSTALL_ID, codeRoot: ROOT, repositoryId: REPOSITORY_ID })
    const currentAttempt = attempt(row.computedSourceFingerprint, installation)
    vi.mocked(observeManagedRemoteUpdate).mockResolvedValue({
      marker: 'absent',
      launchIntent: 'absent',
      receipt: { correlationId: CORRELATION_ID, outcome: 'updated', postSha: TARGET_SHA },
      coordinatorReady: { correlationId: CORRELATION_ID, pid: 1 },
      exitCode: 0
    } as any)

    const proof = await integration.evidence.sweep({
      id: 'rollout-1',
      revision: 1,
      queueGeneration: 1,
      attempts: { [INSTALL_ID]: currentAttempt }
    })

    expect(proof.valid).toBe(true)
    expect(proof.admissions).toHaveLength(1)
    expect(proof.admissions[0].installId).toBe(INSTALL_ID)
  })

  test('refuses promotion evidence when an independent Git HEAD read disagrees', async () => {
    const { integration } = makeIntegration('d'.repeat(40))
    const snapshot = await integration.adapters.inventoryReader.capture()
    const row: any = snapshot!.observations[0]
    const installation = installationFingerprint({ installId: INSTALL_ID, codeRoot: ROOT, repositoryId: REPOSITORY_ID })
    const currentAttempt = attempt(row.computedSourceFingerprint, installation)
    vi.mocked(observeManagedRemoteUpdate).mockResolvedValue({
      marker: 'absent',
      launchIntent: 'absent',
      receipt: { correlationId: CORRELATION_ID, outcome: 'updated', postSha: TARGET_SHA },
      coordinatorReady: { correlationId: CORRELATION_ID, pid: 1 },
      exitCode: 0
    } as any)

    const proof = await integration.evidence.sweep({
      id: 'rollout-1',
      revision: 1,
      queueGeneration: 1,
      attempts: { [INSTALL_ID]: currentAttempt }
    })

    expect(proof.valid).toBe(false)
    expect(proof.reason).toBe('health-evidence-not-proven')
  })
})
