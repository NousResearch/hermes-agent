import { describe, expect, it } from 'vitest'

import {
  SLASH_REFERENCE_DYNAMIC_ROUTES_NOTE,
  slashReferenceCommands,
  slashReferenceDisplayDescription,
  slashReferenceSections,
  slashReferenceSurfaceTag
} from './slash-command-reference'

describe('slash command reference data', () => {
  it('derives screenshot-style surface tags from registry fields', () => {
    expect(slashReferenceSurfaceTag(slashReferenceCommands.find(command => command.name === 'clear')!)).toBe('cli')
    expect(slashReferenceSurfaceTag(slashReferenceCommands.find(command => command.name === 'approve')!)).toBe('chat')
    expect(slashReferenceSurfaceTag(slashReferenceCommands.find(command => command.name === 'verbose')!)).toBe('cfg')
    expect(slashReferenceSurfaceTag(slashReferenceCommands.find(command => command.name === 'retry')!)).toBe('both')
  })

  it('keeps compact display copy separate from registry metadata', () => {
    const newCommand = slashReferenceCommands.find(command => command.name === 'new')!
    const undoCommand = slashReferenceCommands.find(command => command.name === 'undo')!

    expect(newCommand.description).toContain('fresh session ID')
    expect(slashReferenceDisplayDescription(newCommand)).toBe('Start a new session')
    expect(slashReferenceDisplayDescription(undoCommand)).toBe('Back up/remove turn')
  })

  it('groups registry categories into the quick-reference sections', () => {
    expect(slashReferenceSections.map(section => section.title)).toEqual([
      'Session / Flow',
      'Config',
      'Tools / Skills',
      'Info / Exit'
    ])

    const infoExit = slashReferenceSections.find(section => section.title === 'Info / Exit')!

    expect(infoExit.commands.map(command => command.name)).toContain('quit')
  })

  it('shows the current Nous account commands with compact labels', () => {
    const subscription = slashReferenceCommands.find(command => command.name === 'subscription')!
    const topup = slashReferenceCommands.find(command => command.name === 'topup')!

    expect(subscription.aliases).toContain('upgrade')
    expect(slashReferenceDisplayDescription(subscription)).toBe('Nous plan')
    expect(slashReferenceDisplayDescription(topup)).toBe('Balance and billing')
  })

  it('keeps the dynamic routes note visible', () => {
    expect(SLASH_REFERENCE_DYNAMIC_ROUTES_NOTE).toContain('/<skill-name>')
    expect(SLASH_REFERENCE_DYNAMIC_ROUTES_NOTE).toContain('quick commands')
  })
})
