import { atom } from 'nanostores'
import { afterEach, describe, expect, it } from 'vitest'

import { jobTitle } from '@/app/cron/job-state'
import { remoteBrowserManageRefusal } from '@/app/session/hooks/use-prompt-actions/browser-manage-refusal'
import { friendlyRemoteAttachError } from '@/app/session/hooks/use-prompt-actions/utils'
import { buildToolView, type ToolPart } from '@/components/assistant-ui/tool/fallback-model'
import { signInLabel } from '@/components/boot-failure-reauth'
import { providePortalPresentation } from '@/store/shell-presentation'
import type { CronJob } from '@/types/cron'

import { BROWSER_BRIDGE_STUBS } from './browser-bridge-stubs'
import { UnservedProfileError } from './browser-served-profiles'
import { buildGatewayLogItems } from './gateway-events'
import { shellCopy } from './shell-copy'

const portal = atom(false)

const capError = () => new Error('file is too large (999999999 bytes; limit 268435456 bytes)')
const passwordReauth = { isPassword: true, providerLabel: 'Password', url: 'https://hermes.example' }

async function rejection(run: () => Promise<unknown>): Promise<string> {
  try {
    await run()
  } catch (error) {
    return (error as Error).message
  }

  throw new Error('expected a refusal')
}

const cronCall = (args: Record<string, unknown>, result: Record<string, unknown>): ToolPart => ({
  args,
  isError: false,
  result,
  toolCallId: 'call_cron',
  toolName: 'cronjob',
  type: 'tool-call'
})

const cronView = (args: Record<string, unknown>, result: Record<string, unknown>) => {
  const { detail, subtitle } = buildToolView(cronCall(args, result), '')

  return { detail, subtitle }
}

/** Every hardcoded message the redesign rewords, rendered in the current shell. */
async function messages() {
  return {
    applyConfig: await rejection(() => BROWSER_BRIDGE_STUBS.applyConnectionConfig()),
    attachTooLarge: friendlyRemoteAttachError(capError(), 'pic.png').message,
    browserManage: remoteBrowserManageRefusal('connect'),
    cronAction: cronView({ action: 'pause' }, { ok: true }).subtitle,
    cronList: cronView({ action: 'list' }, { jobs: [{ id: '' }, { name: 'Digest' }] }),
    cronNone: cronView({ action: 'list' }, { jobs: [] }),
    jobFallback: jobTitle({ id: '' } as CronJob),
    logEmpty: buildGatewayLogItems([])[0]?.label,
    probe: (await BROWSER_BRIDGE_STUBS.probeConnectionConfig('https://hermes.example')).error,
    remoteSignIn: signInLabel(passwordReauth),
    unserved: new UnservedProfileError('solo').message
  }
}

afterEach(() => {
  portal.set(false)
  providePortalPresentation(atom(false))
})

describe('hardcoded copy per shell', () => {
  it('keeps the main-branch wording in classic', async () => {
    providePortalPresentation(portal)

    expect(await messages()).toEqual({
      applyConfig: 'Gateway reconfiguration is not available in the browser-hosted Desktop',
      attachTooLarge: 'pic.png is too large to upload to the remote gateway (max 256 MB).',
      browserManage:
        '/browser connect manages a Chromium-family browser on the gateway host — only available when connected to a local gateway.',
      cronAction: 'Cron pause',
      cronList: { detail: '- job\n- Digest', subtitle: '2 cron jobs' },
      cronNone: { detail: 'No cron jobs scheduled', subtitle: 'No cron jobs' },
      jobFallback: 'Cron job',
      logEmpty: 'No recent gateway log lines',
      probe: 'Gateway reconfiguration is unavailable in browser-hosted Desktop',
      remoteSignIn: 'Sign in to remote gateway',
      unserved: 'This gateway does not serve profile "solo", so this tab cannot open it.'
    })
  })

  it('uses the redesign vocabulary when the portal shell is on, re-read per message', async () => {
    providePortalPresentation(portal)
    portal.set(true)

    expect(await messages()).toEqual({
      applyConfig: 'Machine reconfiguration is not available in the browser-hosted Desktop',
      attachTooLarge: 'pic.png is too large to upload to the remote machine (max 256 MB).',
      browserManage:
        '/browser connect manages a Chromium-family browser on the machine running Hermes — only available when connected to the local machine.',
      cronAction: 'Routines · pause',
      cronList: { detail: '- Routine\n- Digest', subtitle: '2 routines' },
      cronNone: { detail: 'No routines yet', subtitle: 'No routines yet' },
      jobFallback: '1 routine',
      logEmpty: 'No recent Hermes service log lines',
      probe: 'Machine reconfiguration is unavailable in browser-hosted Desktop',
      remoteSignIn: 'Sign in to remote machine',
      unserved: 'This machine does not serve agent "solo", so this tab cannot open it.'
    })

    portal.set(false)
    expect(shellCopy({ classic: 'Gateway', redesign: 'Machine' })).toBe('Gateway')
  })
})
