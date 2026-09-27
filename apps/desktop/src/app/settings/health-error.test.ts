import { expect, test } from 'vitest'

import { healthError } from './health-error'

test('distinguishes microphone denial, provider rejection and network failure', () => {
  expect(healthError(new Error('NotAllowedError'))).toContain('mikrofonu')
  expect(healthError(new Error('401 invalid API key'))).toContain('API odrzuciło')
  expect(healthError(new Error('Connection timeout'))).toContain('połączyć')
})
test('never echoes unknown provider errors containing credential fragments', () => {
  const secret = 'sensitive-test-value'
  expect(healthError(`Provider returned ${secret}`)).not.toContain(secret)
})
