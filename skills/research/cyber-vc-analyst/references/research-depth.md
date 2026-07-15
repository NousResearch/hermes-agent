# Research Depth

Use one of these depth levels across the cyber VC skill family.

## `quick`

Use for:

- fast Slack reads
- triage
- narrow follow-up questions

Behavior:

- minimal evidence gathering
- compact Slack-first output
- no durable memo unless explicitly requested
- only create `research-state` when the evidence is unusually fragmented

## `standard`

Use for:

- normal company memos
- normal thematic memos
- most compare workflows

Behavior:

- current default depth
- durable memo-ready output
- targeted vault search plus ROS market context when available
- create `research-state` when the work becomes multi-step or incomplete

## `deep`

Use for:

- major themes
- competitor landscapes
- high-priority investment decisions
- work that should compound across future runs

Behavior:

- broader vault recovery
- ROS-first market mapping
- stronger cross-artifact reuse
- explicit `research-state` persistence by default
- full verification pass before final output

## Invocation Rule

When the user does not specify depth, default to `standard`.

Accept terse command shapes such as:

- `!cyber-vc-company quick Red Access Security`
- `!cyber-vc-theme deep SOC Automation / AI SOC`
- `!cyber-vc-compare standard Red Access Security vs Noma Security`
