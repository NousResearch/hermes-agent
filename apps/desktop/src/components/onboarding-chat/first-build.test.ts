import { beforeEach, describe, expect, it } from 'vitest'

const {
  $setupCheckIn,
  reportFirstBuildToolComplete,
  reportFirstBuildTurnComplete,
  resetFirstBuildForTests,
  watchFirstBuild
} = await import('./first-build')

const BUILD = 'build-session'

/** Run `count` tool calls, settling the turn after each so a due check-in is
 *  never held back by the turn boundary the real stream would provide. */
function work(count: number, finalText = 'Still going.'): void {
  for (let i = 0; i < count; i += 1) {
    reportFirstBuildToolComplete(BUILD)
    reportFirstBuildTurnComplete(BUILD, finalText)
  }
}

describe("Setup's check-in on the first build", () => {
  beforeEach(() => {
    resetFirstBuildForTests()
    watchFirstBuild(BUILD, 'default')
  })

  it('stays quiet until the build has done real work', () => {
    work(4)

    expect($setupCheckIn.get()).toBeNull()
  })

  it('checks in twice over a long build, and no more', () => {
    work(60)

    expect($setupCheckIn.get()?.token).toBe(2)
  })

  // A note injected mid-turn would be a synthetic user message inside an
  // assistant turn — the alternation the agent core forbids.
  it('never speaks inside a turn, only after one completes', () => {
    for (let i = 0; i < 30; i += 1) {
      reportFirstBuildToolComplete(BUILD)
    }

    expect($setupCheckIn.get()).toBeNull()

    reportFirstBuildTurnComplete(BUILD, 'Done with that part.')

    expect($setupCheckIn.get()).not.toBeNull()
  })

  // The runbook has the agent ask for a verdict when the first pass lands.
  // A check-in stacked under that is two questions and no answer.
  it('holds off when the turn already asked the user something', () => {
    work(12, 'Here it is.\n\n::ask{question="Does this match?" options="Yes|No"}')

    expect($setupCheckIn.get()).toBeNull()

    reportFirstBuildTurnComplete(BUILD, 'Carrying on.')

    expect($setupCheckIn.get()).not.toBeNull()
  })

  it('ignores every session that is not the build', () => {
    work(30)
    const seen = $setupCheckIn.get()?.token

    for (let i = 0; i < 30; i += 1) {
      reportFirstBuildToolComplete('some-other-session')
    }

    reportFirstBuildTurnComplete('some-other-session', 'Unrelated.')

    expect($setupCheckIn.get()?.token).toBe(seen)
  })
})
