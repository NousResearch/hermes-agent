import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  brandSessionFromCookies,
  isPortalLoginUrl,
  loginNavigationAction,
  parseOtpEmail,
  parsePinValue,
  selectBrandApp,
  signedOutBrand
} from './brand-scope'

const FUTURE = Date.now() + 60_000

function otp(email: string, exp = FUTURE): string {
  return `${Buffer.from(`${email}|${exp}`).toString('base64url')}.sig`
}

function pin(appIds: string, exp = FUTURE): string {
  return `${appIds}|owner|growth|*|${exp}.sig`
}

test('a brand login keeps only the app ids on the pin', () => {
  const session = brandSessionFromCookies(
    [
      { name: 'otp_verified', value: otp('Ada@Foundrly.com') },
      { name: 'portal_pinned_app', value: pin('foundrly+toba') }
    ],
    FUTURE - 1
  )

  assert.equal(session.signedIn, true)
  assert.equal(session.email, 'ada@foundrly.com')
  assert.deepEqual(session.appIds, ['foundrly', 'toba'])
  assert.equal(session.activeAppId, 'foundrly')
  assert.equal(session.isSuper, false)
})

test('a login with the super marker and no pin is a super admin', () => {
  const session = brandSessionFromCookies(
    [
      { name: 'otp_verified', value: otp('root@intelli-verse-x.ai') },
      { name: 'otp_super', value: otp('root@intelli-verse-x.ai') }
    ],
    FUTURE - 1
  )

  assert.equal(session.isSuper, true)
  assert.equal(session.activeAppId, '')
})

test('an OTP cookie alone is not a super admin', () => {
  assert.deepEqual(
    brandSessionFromCookies([{ name: 'otp_verified', value: otp('root@intelli-verse-x.ai') }], FUTURE - 1),
    signedOutBrand()
  )
})

test('an expired brand pin signs the user out', () => {
  assert.deepEqual(
    brandSessionFromCookies([
      { name: 'otp_verified', value: otp('ada@foundrly.com') },
      { name: 'portal_pinned_app', value: pin('foundrly', Date.now() - 1000) }
    ]),
    signedOutBrand()
  )
})

test('an expired OTP cookie is signed out', () => {
  assert.equal(parseOtpEmail(otp('ada@foundrly.com', Date.now() - 1000)), null)
  assert.equal(parsePinValue(pin('foundrly', Date.now() - 1000)), null)
  assert.deepEqual(
    brandSessionFromCookies([{ name: 'otp_verified', value: otp('ada@foundrly.com', Date.now() - 1000) }]),
    signedOutBrand()
  )
})

test('only the portal login pages stay open in the sign-in window', () => {
  assert.equal(isPortalLoginUrl('https://admin.intelli-verse-x.ai/login'), true)
  assert.equal(isPortalLoginUrl('https://admin.intelli-verse-x.ai/forgot-password'), true)
  assert.equal(isPortalLoginUrl('https://admin.intelli-verse-x.ai/admin/portal/tools'), false)
  assert.equal(isPortalLoginUrl('https://admin.intelli-verse-x.ai/vendor/dashboard'), false)
  assert.equal(isPortalLoginUrl('https://example.com/login'), false)
})

test('a dashboard bounce without a session stays on the login page', () => {
  assert.equal(loginNavigationAction('https://admin.intelli-verse-x.ai/login', false), 'allow')
  assert.equal(loginNavigationAction('https://admin.intelli-verse-x.ai/admin/portal/tools', false), 'stay')
  assert.equal(loginNavigationAction('https://admin.intelli-verse-x.ai/admin/portal/tools', true), 'finish')
})

test('a brand user cannot switch to an app id outside the pin', () => {
  const session = brandSessionFromCookies(
    [
      { name: 'otp_verified', value: otp('ada@foundrly.com') },
      { name: 'portal_pinned_app', value: pin('foundrly') }
    ],
    FUTURE - 1
  )

  assert.equal(selectBrandApp(session, 'toba').activeAppId, 'foundrly')
  assert.equal(selectBrandApp(session, 'quizverse').activeAppId, 'foundrly')
})
