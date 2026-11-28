import fs from 'node:fs'
import { setTimeout } from 'node:timers/promises'

import { getAppSigningResult } from './mac-sign.mjs'
import { remainingNotaryBudgetMs, resolveNotaryAuth, runCommand, submitAndStaple, validateStapled } from './notarize.mjs'

export async function resolveDmgSigningIdentity(packager, { env = process.env } = {}) {
  // Only the certificate electron-builder signed the app with may sign its DMG; never pick another one.
  const recorded = getAppSigningResult(packager)
  if (recorded) {
    return {
      identity: recorded.identity === '-' ? null : recorded.identity,
      keychain: recorded.keychain
    }
  }

  // The app is unsigned: the signing settings only decide whether that was intended.
  const signConfig = packager?.platformOptions?.sign
  if (signConfig === null || (typeof signConfig === 'object' && signConfig?.identity === null)) {
    return { identity: null, keychain: null }
  }

  const rawQualifier = (typeof signConfig === 'object' && signConfig !== null ? signConfig.identity : undefined) ?? env.CSC_NAME
  const qualifier = typeof rawQualifier === 'string' && rawQualifier.trim() ? rawQualifier.trim() : null
  if (qualifier === '-' || (!qualifier && env.CSC_IDENTITY_AUTO_DISCOVERY === 'false')) {
    return { identity: null, keychain: null }
  }

  const info = await packager?.codeSigningInfo?.value
  const keychain = info?.keychainFile || env.CSC_KEYCHAIN || null
  if (keychain || qualifier) {
    throw new Error(
      `notarize-dmg: signing is configured (${keychain || qualifier}), but electron-builder did not sign the app`
    )
  }
  return { identity: null, keychain: null }
}

async function refreshDmgBlockmap(dmgPath, event, buildBlockMap, log) {
  const blockMapFile = `${dmgPath}.blockmap`
  if (!fs.existsSync(blockMapFile) && !event.isWriteUpdateInfo && event.updateInfo == null) {
    return
  }
  const rebuild = buildBlockMap ?? (await import('app-builder-lib/internal')).buildBlockMap
  const updated = await rebuild(dmgPath, 'gzip', blockMapFile)
  if (event.updateInfo && typeof event.updateInfo === 'object') {
    Object.assign(event.updateInfo, updated)
  } else {
    event.updateInfo = updated
  }
  log(`[notarize-dmg] refreshed blockmap and update metadata for ${dmgPath} (${updated.size} bytes)`)
}

export default async function notarizeDmg(
  event,
  {
    run = runCommand,
    env = process.env,
    sleep = setTimeout,
    log = console.log,
    now = () => performance.now(),
    buildBlockMap
  } = {}
) {
  if (!event || event.target?.name !== 'dmg' || typeof event.file !== 'string' || !event.file.endsWith('.dmg')) {
    return
  }
  if (event.packager?.platform?.nodeName && event.packager.platform.nodeName !== 'darwin') {
    return
  }

  const dmgPath = event.file
  if (!fs.existsSync(dmgPath)) {
    throw new Error(`Cannot sign or notarize missing DMG artifact: ${dmgPath}`)
  }

  const notaryAuth = resolveNotaryAuth(env)
  try {
    const { identity, keychain } = await resolveDmgSigningIdentity(event.packager, { env })
    if (!identity) {
      if (notaryAuth) {
        throw new Error(`Cannot notarize DMG without a Developer ID Application signing identity: ${dmgPath}`)
      }
      log('[notarize-dmg] Skipping DMG signing and notarization: no Developer ID Application identity configured.')
      return
    }

    log(`[notarize-dmg] signing ${dmgPath}`)
    const signArgs = ['--force', '--sign', identity, '--timestamp', '--verbose']
    if (keychain) signArgs.push('--keychain', keychain)
    signArgs.push(dmgPath)
    await run('codesign', signArgs)
    await run('codesign', ['--verify', '--verbose=2', dmgPath])

    if (notaryAuth) {
      const waitBudgetMs = remainingNotaryBudgetMs({ now })
      const { deadline } = await submitAndStaple(dmgPath, dmgPath, notaryAuth.auth, {
        run,
        sleep,
        log,
        now,
        label: 'notarize-dmg',
        waitBudgetMs
      })
      await run('codesign', ['--verify', '--verbose=2', dmgPath])
      await validateStapled(dmgPath, { run, sleep, log, now, deadline, label: 'notarize-dmg' })
      await run('spctl', ['-a', '-vv', '-t', 'open', '--context', 'context:primary-signature', dmgPath])
      log(`[notarize-dmg] verified signed, stapled, and Gatekeeper-accepted DMG: ${dmgPath}`)
    } else {
      log(
        '[notarize-dmg] Skipping DMG notarization: APPLE_API_KEY, APPLE_API_KEY_ID, and APPLE_API_ISSUER are not fully configured.'
      )
    }

    await refreshDmgBlockmap(dmgPath, event, buildBlockMap, log)
  } finally {
    notaryAuth?.cleanup()
  }
}
