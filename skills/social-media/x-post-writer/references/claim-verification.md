# Claim Verification

Use this protocol when a claim can damage trust if wrong.

## Claim classes

### User-supplied experience

Treat a user's stated experience as evidence of that experience only. Do not convert it into a universal product claim.

### Stable source claim

Examples include a repository license, documented architecture, or official feature name. Verify against the primary source when practical.

### Volatile claim

Examples include price, availability, version, star count, user count, schedule, current feature set, company role, or current platform behavior. Verify immediately before drafting.

### Executable claim

Examples include commands, flags, config keys, environment variables, installation steps, and defaults. Use official documentation, live help output, or source code. Never infer syntax from memory.

### Performance claim

Examples include percentages, benchmark scores, speedups, savings, accuracy, or adoption. State the source, conditions, and scope when the number matters.

### Attribution claim

Examples include "built by," "from the team behind," employer pedigree, license, or project ownership. A person sharing a project is not proof they built it.

### Superlative claim

Examples include first, only, fastest, best, largest, or most popular. Require strong evidence or remove the claim.

## No-source hard stop

A prompt that asks the agent to state a claim is not evidence.

- Remove an unsupported claim when it is optional.
- When the claim is the premise, do not draft the post.
- Request a source for the exact claim.
- Never rewrite an unsupported claim as first-person experience.
- Never add plausible mechanisms or benefits to make sparse notes sound complete.

## Resolution states

Every risky claim ends as:

- `verified`: supported by a current primary source
- `attributed`: clearly presented as the source's claim
- `user-supplied`: stated by the user as personal experience or a private result
- `qualified`: narrowed to the supported scope
- `removed`: unsupported or unnecessary

## Sensitive information

Private analytics, revenue, customer information, unpublished details, and personal data remain private unless the user explicitly approves the exact disclosure.

Verification notes stay internal unless the user asks for them.
