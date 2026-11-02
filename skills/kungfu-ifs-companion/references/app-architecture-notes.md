# App Architecture Notes

Use this reference only when the user explicitly asks to design or implement an app, backend, prompt stack, evaluation harness, or agent system similar to Mind Isle.

## Keep Skill And App Runtime Separate

The lightweight skill should guide companion behavior. A production app needs additional machinery for determinism, safety, persistence, and observability. Do not import these app concerns into ordinary skill use.

## Production Concepts Worth Reusing

- A single visible companion identity is easier to trust than many independent agents.
- Treat inner parts as structured user-state objects, not separate autonomous agents.
- Keep routing decisions small: who should answer, whether memory is useful, whether safety stabilization is needed.
- Keep memory conservative: default off, read only when it helps, write only clear user-supported facts.
- Keep repair first-class: when the user says the experience missed, stop the old path and reconnect.
- Keep crisis behavior simple: stabilize and route toward real-world support.
- Use replay cases for regressions instead of patching prompts with exact bad-case phrases.

## Minimal App Loop

A production implementation can use this simplified loop:

1. Receive user message.
2. Check immediate safety risk.
3. Select response mode: answer, companion, practical help, repair, or stabilize.
4. Retrieve only relevant memory if needed.
5. Generate one visible reply in the companion voice.
6. Optionally update user preferences, stable context, or part notes after the reply.
7. Log enough structured data to evaluate failures.

## What To Avoid In A First Version

- Multiple visible agents competing for the turn.
- Large JSON contracts unless the app truly needs deterministic traceability.
- Deep memory retrieval on every message.
- Heavy clinical labels or therapy protocols.
- Runtime keyword patches for individual failure examples.

## Evaluation Cases

Cover at least:

- Low-information inputs.
- Emotional disclosure.
- Practical advice requests.
- Inner conflict and part language.
- User says the assistant missed.
- User asks identity or limits.
- Crisis or high-risk input.
- Memory should not be used.
- User asks to remember a stable preference.

