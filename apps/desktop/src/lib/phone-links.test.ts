import { describe, expect, it } from 'vitest'

import { linkifyPhoneNumbersInMarkdown, phoneLinkTarget } from './phone-links'

describe('phone links', () => {
  it('normalizes domestic and international dial targets', () => {
    expect(phoneLinkTarget('010-495 64 04')).toBe('tel:0104956404')
    expect(phoneLinkTarget('+46 (0)10 495 64 04')).toBe('tel:+46104956404')
    expect(phoneLinkTarget('+1 (212) 555-0100')).toBe('tel:+12125550100')
    expect(phoneLinkTarget('01 23 45 67 89')).toBe('tel:0123456789')

    for (const value of ['010-495 64 04', '+46 (0)10 495 64 04', '+1 (212) 555-0100']) {
      expect(phoneLinkTarget(value)).toMatch(/^tel:\+?\d{7,15}$/u)
    }

    expect(phoneLinkTarget('+46 (0)12 34')).toBeNull()
  })

  it('rejects common non-phone numeric formats', () => {
    expect(phoneLinkTarget('2026-07-22')).toBeNull()
    expect(phoneLinkTarget('192.168.0.1')).toBeNull()
    expect(phoneLinkTarget('4111 1111 1111 1111')).toBeNull()
    expect(phoneLinkTarget('255 255 255')).toBeNull()
    expect(phoneLinkTarget('1920 1080 60')).toBeNull()
    expect(phoneLinkTarget('123-45-6789')).toBeNull()
    expect(phoneLinkTarget('1234-5678-90')).toBeNull()
    expect(phoneLinkTarget('+2026-07-22')).toBeNull()
    expect(phoneLinkTarget('+192.168.0.1')).toBeNull()
    expect(phoneLinkTarget('+123-45-6789')).toBeNull()
    expect(phoneLinkTarget('+1920 1080 60')).toBeNull()
  })

  it('leaves existing URL and telephone targets unchanged', () => {
    const text = [
      '<https://example.com/call/010-495-6404>',
      'www.example.com/call/010-495-6404',
      'example.com/call/010-495-6404',
      '[Call us](tel:0104956404)',
      '[Call [office] 010-495 64 04](https://example.com/contact)',
      '[010-495 64 04][contact]',
      'tel:010-495-64-04',
      'tel:010 495 64 04',
      'tel: 212 555 0100'
    ].join(' ')

    expect(linkifyPhoneNumbersInMarkdown(text)).toBe(text)
    expect(linkifyPhoneNumbersInMarkdown('tel: 212 555 0100')).toBe('tel: 212 555 0100')
  })

  it('does not link ambiguous numeric formats without phone context', () => {
    for (const text of [
      'Sequence 01 02 03 04',
      'Account 123 4567 8901',
      'Version 12-345-6789',
      'Matrix 123 456 78 90',
      'Build dimensions 192 168 10 24',
      'ORD-010-495-6404',
      'ORD-010-495-6404 contact',
      'ORD-010-495-6404 phone',
      'Email call-010-495-6404@example.com',
      'mailto:call-010-495-6404+tag@example.com'
    ]) {
      expect(linkifyPhoneNumbersInMarkdown(text)).toBe(text)
    }
  })

  it('does not skip phone prose that begins with Markdown delimiters', () => {
    expect(linkifyPhoneNumbersInMarkdown('<Note Phone: 010-495 64 04')).toBe(
      '<Note Phone: [010-495 64 04](tel:0104956404)'
    )
    expect(linkifyPhoneNumbersInMarkdown('[Note Phone: 010-495 64 04')).toBe(
      '[Note Phone: [010-495 64 04](tel:0104956404)'
    )
  })

  it('emits only main-process-compatible telephone targets', () => {
    const inputs = [
      'Phone: 010-495 64 04',
      'Call +46 (0)10 495 64 04',
      'Call +1 212 555 0100'
    ]

    const targets = inputs.flatMap(input =>
      Array.from(linkifyPhoneNumbersInMarkdown(input).matchAll(/\((tel:[^)]+)\)/gu), match => match[1])
    )

    expect(targets).toHaveLength(inputs.length)
    expect(targets.every(target => /^tel:\+?\d{7,15}$/u.test(target))).toBe(true)
  })

  it('stops phone links before following prose numbers', () => {
    expect(linkifyPhoneNumbersInMarkdown('Call +46 (0)10 495 64 04. 24/7 support.')).toBe(
      'Call [+46 (0)10 495 64 04](tel:+46104956404). 24/7 support.'
    )
    expect(linkifyPhoneNumbersInMarkdown('Call +1 (212) 555-0100 - 123')).toBe(
      'Call [+1 (212) 555-0100](tel:+12125550100) - 123'
    )
    expect(linkifyPhoneNumbersInMarkdown('Call +1 212 555 0100 2026 schedule')).toBe(
      'Call [+1 212 555 0100](tel:+12125550100) 2026 schedule'
    )
  })
})
