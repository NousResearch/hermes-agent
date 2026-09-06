import assert from 'node:assert/strict'

import { describe, it } from 'vitest'

import {
  buildExactTerminateScript,
  createWindowsHolderTerminator,
  parseTerminateScriptOutput,
  TERMINATE_JOB_WRAPPER_COMMAND
} from './windows-process-terminate'
import type { ForceReleaseHolder } from './windows-update-force-release'

describe('exact terminate script: executable-under-install-root authorization', () => {
  const withRoot = buildExactTerminateScript(4242, 1_700_000_000, 500, { installRoot: 'C:\\hermes' })
  const withoutRoot = buildExactTerminateScript(4242, 1_700_000_000, 500)

  it('requires an install-root claim unconditionally', () => {
    for (const script of [withRoot, withoutRoot]) {
      assert.match(script, /if \(\[string\]::IsNullOrWhiteSpace\(\$installRootClaim\)\) \{\s*throw 'TERMINATION_AUTHORIZATION_CLAIM_MISSING'/)
      // The old gate let both-empty claims skip authorization entirely.
      assert.doesNotMatch(script, /-not \[string\]::IsNullOrWhiteSpace\(\$installRootClaim\) -or -not \[string\]::IsNullOrWhiteSpace\(\$resourceClaim\)/)
      assert.doesNotMatch(script, /TERMINATION_AUTHORIZATION_CLAIM_INCOMPLETE/)
    }

    assert.match(withoutRoot, /^\$installRootClaim = ''$/m)
  })

  it('proves the root image lives under the install root before the resource check', () => {
    const rootImage = withRoot.indexOf('$imagePath = [HermesForceReleaseNative]::ReadImagePath($rootHandle)')
    const rootUnderRoot = withRoot.indexOf('IsSameOrUnderRoot($imageFinal, $installRootFinal)')
    const resourceGate = withRoot.indexOf("if (-not [string]::IsNullOrWhiteSpace($resourceClaim)) {")
    const ownership = withRoot.indexOf('IsCurrentResourceOwner($resourceFinal, $pidTarget, $expectedUnix)')
    const rootAssign = withRoot.indexOf('$handles[[string]$pidTarget] = $rootHandle')

    assert.ok(rootImage > 0 && rootUnderRoot > rootImage, 'root image path is resolved then checked')
    assert.ok(resourceGate > rootUnderRoot, 'the resource claim is consulted only after the image proof')
    assert.ok(ownership > resourceGate && ownership < rootAssign, 'ownership re-proof is inside the resource gate, before assignment')
    assert.match(withRoot, /TERMINATION_EXECUTABLE_OUTSIDE_INSTALL_ROOT/)
    assert.match(withRoot, /TERMINATION_EXECUTABLE_IDENTITY_UNAVAILABLE/)
  })

  it('applies the same image proof to every descendant before it is assigned to the job', () => {
    const open = withRoot.indexOf('$childHandle = [HermesForceReleaseNative]::OpenAuthenticatedProcess($childPid')
    const image = withRoot.indexOf('$childImageFinal = [HermesForceReleaseNative]::ReadFinalPath([HermesForceReleaseNative]::ReadImagePath($childHandle))')
    const assign = withRoot.indexOf('$handles[[string]$childPid] = $childHandle')

    assert.ok(open > 0 && image > open && assign > image, 'descendant image is proven between open and assignment')
    assert.match(
      withRoot,
      /if \(\[string\]::IsNullOrWhiteSpace\(\$childImageFinal\) -or -not \[HermesForceReleaseNative\]::IsSameOrUnderRoot\(\$childImageFinal, \$installRootFinal\)\) \{\s*\[HermesForceReleaseNative\]::CloseHandle\(\$childHandle\) \| Out-Null\s*\[void\]\$foreign\.Add\(\$childPid\)\s*continue\s*\}/
    )
  })

  it('leaves foreign descendants alive and reports them on the success line', () => {
    assert.match(withRoot, /^\$foreign = New-Object 'System\.Collections\.Generic\.List\[int\]'$/m)
    assert.match(withRoot, /if \(\$foreign\.Count -gt 0\) \{ Write-Output \('FOREIGN_DESCENDANTS_LEFT_ALIVE pids=' \+ \(\$foreign -join ','\)\) \}/)
    assert.deepEqual(parseTerminateScriptOutput('TERMINATED\nFOREIGN_DESCENDANTS_LEFT_ALIVE pids=58196,58200', 0), {
      kind: 'terminated'
    })
  })
})

describe('force-release holder routing', () => {
  it.each(['scanner', 'restart-manager'] as const)('passes the install root for a %s holder', async source => {
    const holder: ForceReleaseHolder = {
      pid: 4242, createdAt: 1_700_000_000, name: 'python.exe', cmdline: '', source,
      ...(source === 'restart-manager' ? { resource: 'C:\\hermes\\venv\\native.pyd' } : {})
    }

    const signal = new AbortController().signal

    const terminate = createWindowsHolderTerminator('C:\\hermes', async () => {
      assert.fail('generic holders must not use the plugin service path')
    }, async (target, options) => {
      assert.equal(target, holder)
      assert.deepEqual(options, { budgetMs: 500, deadlineAt: 900, signal, installRoot: 'C:\\hermes' })

      return { kind: 'terminated' }
    })

    assert.deepEqual(await terminate(holder, 500, signal, 900), { kind: 'terminated' })
  })

  it('allows target descendants to break away until explicitly authenticated', () => {
    const script = buildExactTerminateScript(4242, 1_700_000_000, 500, { installRoot: 'C:\\hermes' })
    assert.match(script, /JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE \| JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK/)
    assert.match(TERMINATE_JOB_WRAPPER_COMMAND, /CreateKillOnClose\(\$targetJobName, \$true\)/)
    assert.match(TERMINATE_JOB_WRAPPER_COMMAND, /CreateKillOnClose\(\$helperJobName, \$false\)/)
  })

  it('derives an absolute deadline from the holder budget when none is supplied', async () => {
    const holder: ForceReleaseHolder = {
      pid: 4242, createdAt: 1_700_000_000, name: 'python.exe', cmdline: '', source: 'scanner'
    }

    const before = Date.now()

    const terminate = createWindowsHolderTerminator('C:\\hermes', async () => false, async (_target, options) => {
      assert.ok(options.deadlineAt >= before + 500)
      assert.ok(options.deadlineAt <= Date.now() + 500)

      return { kind: 'terminated' }
    })

    assert.deepEqual(await terminate(holder, 500), { kind: 'terminated' })
  })

  it.each([true, false])('uses the plugin unit stop result (%s) without generic termination', async stopped => {
    const service: NonNullable<ForceReleaseHolder['service']> = {
      pid: 4242, createdAt: 100, name: 'python.exe', cmdline: '', owner: 'desktop',
      role: 'desktop_plugin_wrapper', actionable: true,
      actionability: 'exact_desktop_plugin_service', action: 'terminate_desktop_plugin_service'
    }

    const holder: ForceReleaseHolder = {
      pid: service.pid, createdAt: service.createdAt, name: service.name, cmdline: '', source: 'scanner',
      terminateVia: 'desktop-plugin-service', service
    }

    const terminate = createWindowsHolderTerminator('C:\\hermes', async observed => {
      assert.equal(observed, service)

      return stopped
    }, async () => { assert.fail('plugin units must not use generic termination') })

    assert.deepEqual(await terminate(holder, 500), stopped
      ? { kind: 'terminated' }
      : { kind: 'failed', detail: 'plugin service unit not stopped' })
  })
})
