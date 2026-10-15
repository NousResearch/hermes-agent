# Lessons Learned: Engineering Cybernetics Reform Implementation

## What Worked
- Protocol-layer reforms (memory rules, behavioral constraints) are fast to deploy — they require no code changes, just updated instructions.
- The reform framing ("what would an engineering cyberneticist do?") gave us a concrete mental model for diagnosing systemic failure modes.

## What Didn't Work: Paper Tiger Skills
- 6 'paper tiger' skills were created but never used. These were descriptive documents — prose explaining *how* things should work — not operational tools that actually enforce behavior.
- A skill file that says "always verify before asserting" does nothing if the agent doesn't load it at the right moment, or if the instruction is too vague to act on.
- Lesson: descriptive docs ≠ enforcement mechanisms. If a skill doesn't trigger automatically or produce a concrete artifact, it's decorative.

## The Self-Referential Control Problem
- The agent evaluating its own performance is structurally compromised: it's the controller *and* the controlled system, with no independent sensor.
- OCGS theory warns against this. Without an external feedback loop (human, test suite, or independent monitor), self-assessment tends toward self-justification.
- This is why the performance log exists — to create an auditable, external trace that the agent cannot retroactively rationalize.

## Human-in-the-Loop Is Not Optional
- Per OCGS theory, the human serves as the observer in the feedback loop. Removing the human from the loop removes the only independent measurement of system behavior.
- Every automation that bypasses human review is a feedback loop that's been cut open — the system now runs open-loop.

## Timing: The Window Is Closing
- yaklang/control-theory-skill appeared while we were researching this domain. Others are converging on the same ideas.
- The window for us to establish prior art, published methodology, and credibility is finite.
- Move fast on substance (working code, published papers), not on meta-documentation.

## Next Steps: Code-Layer Reforms via Claude Code
- Protocol-layer reforms (what we've done so far) are fast but fragile — they depend on the model reading and following instructions.
- Code-layer reforms (pre-commit hooks, test gates, CI checks, structured output validators) are slower to build but reliable — they don't depend on the model's compliance.
- Claude Code is the right tool for this: it operates at the code layer, not the prompt layer.
- Priority: build enforcement mechanisms that make bad behavior structurally impossible, not just discouraged.
