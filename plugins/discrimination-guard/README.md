# discrimination-guard

Warns when a probe or test the agent is writing **cannot produce the opposite
result**.

## The problem

An agent that writes its own verification code has a specific way of being
confidently wrong: it produces a measurement that looks like evidence but that
would have come out the same way whether or not the claim were true. The
output then reads exactly like a discovery.

Real examples, all from one session:

| what was observed | what it actually meant |
|---|---|
| `pytest` reported `0 failed` | collection **errored**; zero tests ran |
| `tf.format == 2` (PAX) | `format` on a *read* handle is the reader default |
| `hasattr(mod, name)` is `False` | the function is **nested**, not module-level |
| `19 FAILED` in someone's suite | 14 were `ModuleNotFoundError` from an incomplete venv |
| `(rc == 0) == (on == wanted)` held | a tautology — true whichever way the tool behaves |
| `git checkout -B` "worked" | it printed `Aborting`; the next patch hit the old branch |

Every one of those is the same defect: a measurement accepted without first
establishing that it can **distinguish** the claimed outcome from its opposite.

## What it checks

Structural properties, not a list of known-bad snippets — so it fires on shapes
it has never seen:

1. a verdict is emitted with **no negative control** anywhere in the probe
2. an assertion compares **two of the probe's own observations** (tautology)
3. a `subprocess` result is consumed **without inspecting `returncode`**
4. a *"no failures"* conclusion with **no evidence anything ran**
5. **text presence** (`"x" in src`, `grep -c`) used as proof a code path behaves

Only files that look like probes/tests are considered, and only writes over
~120 characters.

## Behaviour

Non-blocking by default: the write proceeds and the warning rides back to the
model in the next turn so it can self-correct. Set
`DISCRIMINATION_GUARD_BLOCK=1` to refuse the write instead.

## False positives

The guard is quiet on probes that already discriminate — one that drives the
real function and asserts a control case, one that checks `returncode`, one
that mutation-verifies its own fix, and on ordinary source files. Those four
shapes are covered as negative cases in the test suite, because a guard that
flags everything gets ignored within a day.

## Testing

```
pytest tests/plugins/test_discrimination_guard_plugin.py
```

8 cases: 4 real wrong-verdict probes that must be flagged, 4 sound probes that
must stay quiet.
