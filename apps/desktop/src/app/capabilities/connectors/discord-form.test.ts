import { discordMessageReady, discordSnowflake, discordTokenAccepted } from './discord-form'

describe('discord form checks', () => {
  it('accepts a numeric id and a pasted mention', () => {
    expect(discordSnowflake('<@880636691499581512>')).toBe('880636691499581512')
    expect(discordSnowflake('<#1497111463976898620>')).toBe('1497111463976898620')
    expect(discordSnowflake('go880636691499581512')).toBeNull()
  })

  it('rejects an application id used as a bot token', () => {
    expect(discordTokenAccepted('123456789012345678901')).toBe(false)
    expect(discordTokenAccepted('aaa.bbb.ccc-token-value')).toBe(true)
  })

  it('requires a message inside the Discord limit', () => {
    expect(discordMessageReady('  hi  ')).toBe(true)
    expect(discordMessageReady('   ')).toBe(false)
    expect(discordMessageReady('x'.repeat(2001))).toBe(false)
  })
})
