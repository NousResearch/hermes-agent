# Zero Option Profile

Convert a zero-capital or severe-constraint state into externally testable action loops.

## Triggers

zero capital, first cash, immediate executable opportunity, constrained launch, no budget.

## Conversion

```text
ZERO_STATE → OPTIONALITY → MICRO_EXECUTION → FEEDBACK → LEVERAGE
```

## Runes

- `RUNE.NG.ZERO.NORMALIZE`
- `RUNE.NG.ZERO.EXTRACT_CAPABILITIES`
- `RUNE.NG.ZERO.GENERATE`
- `RUNE.NG.ZERO.SCORE_OPTIONALITY`
- `RUNE.NG.ZERO.BUILD_MICRO_LOOP`
- `RUNE.NG.ZERO.HYPERSTITION_BIND`

## Required inputs

- declared skills / capabilities (explicit list; empty → fail);
- declared access (accounts, audiences, tools, inventory);
- time window;
- capital constraint (often zero);
- proof definition for “done / cash / signal.”

## Hard filters

- cost equals zero when explicitly operating in zero-capital mode;
- executable within the declared time window;
- direct proof path;
- **no fictional credentials, access, tools, or relationships** (Gate G).

## Detectors

`DRIFT`, `STAGNATION`, `NON_EXECUTION`, `FICTIONAL_RESOURCE`.

Narrative must bind to observable action:

```text
Narrative → Action → Result → Reinforcement
```

## Outputs

- `ZeroOptionPacket`
- `NeonGenieRunReceipt`

Schema: `schemas/zero-option-packet.schema.json`

## Fail closed

If skills and access are empty or unusable under constraints → `NOT_COMPUTABLE` with reason (see `evals/cases/zero-option.json`).
