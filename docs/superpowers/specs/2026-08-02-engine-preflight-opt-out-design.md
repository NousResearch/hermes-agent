# Engine Preflight Opt-Out Design

## Problem

Issue #76151 reports that an external context engine can request preflight
maintenance far below Hermes' configured compression threshold. For engines
such as `hermes-lcm`, that maintenance may block the start of a turn for several
minutes and can repeat on later turns. Hermes currently calls
`should_compress_preflight()` whenever the sub-threshold preflight path is
eligible, with no host-level opt-out.

The sub-threshold hook is intentional behavior introduced for #20316. Making it
obey the ordinary compression threshold would break the context-engine contract
and prevent incremental maintenance for every user. The fix therefore needs to
be an explicit operator choice rather than a semantic change to the hook.

## Goals

- Add a `compression.preflight_enabled` boolean setting.
- Preserve current behavior by defaulting the setting to `true`.
- When set to `false`, skip only engine-driven sub-threshold preflight
  maintenance.
- Keep ordinary threshold-triggered compression, manual compression, idle
  compaction, and other compression paths unchanged.
- Document the setting in the example configuration.

## Non-Goals

- Changing `ContextEngine.should_compress_preflight()` or the LCM plugin's
  internal maintenance policy.
- Making engine preflight obey `compression.threshold`.
- Adding timeouts, cancellation, cooldowns, or new retry policy.
- Changing the default behavior for existing installations.

## Considered Approaches

### 1. Host-level boolean gate (selected)

Parse `compression.preflight_enabled`, store the resolved value on the agent,
and include it in the existing engine-preflight dispatch gate. This is small,
backwards-compatible, applies to every context-engine plugin, and directly
provides the requested opt-out.

### 2. Force preflight to respect the ordinary threshold

This would eliminate the reported early maintenance, but it reverses the
explicit #20316 contract that permits engines to perform incremental work below
the host threshold. It would silently disable valid behavior for all users and
is therefore rejected.

### 3. Add host-level timing or cooldown controls

Timeout and cadence controls could reduce stalls while retaining maintenance,
but they introduce cancellation and engine-state semantics beyond the issue's
request. They are better handled separately after the plugin and host contracts
are understood, so they are rejected for this PR.

## Configuration Contract

The new setting is:

```yaml
compression:
  preflight_enabled: true
```

The canonical default config and legacy CLI defaults will both expose `true`.
Runtime parsing will use the project's existing truthy-value convention and
fall back to `true` when the key is absent or the compression section is
partial. Adding the key does not require a config-version bump because the
configuration loader deep-merges new keys.

## Runtime Design

Agent initialization resolves the setting once and stores it as
`agent.compression_preflight_enabled`. The turn-start compression flow keeps all
of its existing eligibility checks. In the sub-threshold `else` arm, Hermes
consults `should_compress_preflight()` only when the new agent flag is true.

With the flag false:

- the engine hook is not called;
- `_compress_context()` is not called by that engine-preflight arm;
- no engine-preflight bookkeeping or status output changes;
- the over-threshold `should_compress()` path remains available.

The default is read through `getattr(..., True)` at the turn boundary so older
or minimal agent test doubles remain compatible.

## Testing

Behavior tests will extend the existing engine-preflight suite:

1. A false flag prevents a true-returning engine hook from being consulted and
   prevents the engine-driven compression pass.
2. A false flag does not prevent the ordinary over-threshold compression path.
3. Existing default-enabled behavior continues to call the hook and retains
   its current once-per-turn semantics.

Configuration coverage will assert that agent initialization resolves the new
setting to true by default and false when explicitly configured, using runtime
behavior rather than reading source text. Tests will run through
`scripts/run_tests.sh` as required by the repository.

## Documentation and Compatibility

`cli-config.yaml.example` will explain that disabling the option affects only
engine-requested sub-threshold maintenance. The setting remains enabled by
default, so existing users and context-engine plugins see no behavior change
unless an operator opts out.
