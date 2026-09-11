import { beforeEach, describe, expect, it } from 'vitest'

import { $machine } from './machine'
import { buildChatOnboardingPrompt } from './onboarding-script'

/** Every `::onboarding{step="…"}` the runbook tells Setup to place. */
function cardsIn(runbook: string): string[] {
  return [...runbook.matchAll(/::onboarding\{step="([a-z-]+)"/g)].map(match => match[1])
}

describe('the onboarding runbook', () => {
  beforeEach(() => {
    $machine.set(null)
  })

  // The pacing rule can only hold if it names the cards the script actually
  // uses. A card added to a step but not to RULE 3 is a question the model has
  // no reason to stop after — which is how two of them ended up on screen at
  // once, each waiting on an answer the other was covering up.
  it('names every question card it places in the one-question rule', () => {
    const runbook = buildChatOnboardingPrompt()
    const rule = runbook.slice(runbook.indexOf('RULE 3'), runbook.indexOf('Your first message'))

    // Data-only directives are exempt by design: they render as nothing, so
    // they ride along with the question in their turn.
    const questions = new Set(cardsIn(runbook).filter(step => !['name', 'progress', 'working'].includes(step)))

    for (const step of questions) {
      expect(rule, `step="${step}" is placed but not named in RULE 3`).toContain(`step="${step}"`)
    }

    expect(questions.size).toBeGreaterThan(1)
  })

  it('tells Setup to stop after asking, not to keep going', () => {
    const runbook = buildChatOnboardingPrompt()

    expect(runbook).toContain('ONE question per turn')
    expect(runbook).toContain('Two questions in one message is a failure')
  })

  // A landed `memory` write already draws its own line in the transcript —
  // brain glyph, "Saved to memory", the gold→purple gradient. Setup used to be
  // told to ALSO mention one saved fact in prose, which plays the same beat
  // twice and spends a clause of the handoff turn hand-rolling an affordance
  // the app ships.
  it('lets the handoff persist facts without a duplicate memory write', () => {
    const runbook = buildChatOnboardingPrompt()

    expect(runbook).toMatch(/memory tool/i)
    expect(runbook).toMatch(/handoff saves the agreed/i)
    expect(runbook).toMatch(/Do not duplicate that write/)
    expect(runbook).not.toMatch(/passing aside/i)
  })

  // Every directive shares its paragraph with nothing, because a directive the
  // model tacks onto the end of a sentence used to print as raw markup.
  it('asks for each directive on a line of its own', () => {
    const runbook = buildChatOnboardingPrompt()

    expect(runbook).not.toContain('include the line')
    expect(runbook).toContain('alone as its own paragraph')
  })

  // The tour has nothing to show until the layout pick has assembled the app
  // around the chat — before it, the window IS the conversation.
  it('offers the look around after the layout, with all three ways out', () => {
    const runbook = buildChatOnboardingPrompt()

    expect(runbook.indexOf('Want a look around first?')).toBeGreaterThan(runbook.indexOf('step="layout"'))

    for (const option of ['Show me around', 'Just the basics', "I'll figure it out"]) {
      expect(runbook).toContain(option)
    }

    // Lightest first: the short one has to be the one their eye lands on, so
    // the full look around reads as a deliberate step up rather than the
    // default. Nobody wants to open a new app into a click-through.
    expect(runbook.indexOf('Just the basics')).toBeLessThan(runbook.indexOf('Show me around'))

    // Selectors come from the tool, not from the model's imagination.
    expect(runbook).toMatch(/action="targets" FIRST/)
  })

  // The light option points at the buttons too — it is the same tour, three
  // steps instead of six. Describing a button in prose is worse than
  // highlighting it: the user still has to go find the thing afterwards.
  it('runs the tour for the light option as well, not a written description', () => {
    const runbook = buildChatOnboardingPrompt()

    const basics = runbook.indexOf('Just the basics"')
    const shared = runbook.indexOf('Both of those run the tour tool')

    expect(basics).toBeGreaterThan(-1)
    expect(shared).toBeGreaterThan(basics)
    expect(runbook.slice(basics, shared)).toMatch(/three steps/i)
  })

  // The tour is the flow's most reusable trick and the easiest to never
  // discover, so the one turn that mentions it has to say it is always on
  // offer — including to the user who just waved the tour off.
  it('tells them the tour stays on offer, whichever way they answered', () => {
    const runbook = buildChatOnboardingPrompt()

    expect(runbook).toContain('Whichever they picked')
    expect(runbook).toMatch(/any part of this, any time/)
  })

  // There is one place a first build can land: a session on the user's own
  // profile. A surface attribute in the script is the model being asked to
  // choose between shapes the app no longer has.
  it('hands off without a surface choice', () => {
    const runbook = buildChatOnboardingPrompt()

    expect(runbook).toContain('step="handoff"')
    expect(runbook).not.toMatch(/surface=/)
  })

  // Setup's proactivity rides the build's own progress (first-build.ts). A
  // cron scheduled at handoff fires tomorrow, about a task that finished in
  // four minutes.
  it('never tells Setup to schedule itself a check-in', () => {
    expect(buildChatOnboardingPrompt()).not.toMatch(/cron/i)
  })
})
