# Bryan/Hermitage premium lesson packet reference routing

Use this reference as the class-level lesson from the Bryan/Darin correction without copying Bryan-specific assets into other coach agents.

## What happened
A coach noted that the agent produced a rough draft packet and cleaned it up into a basic document, but did not produce the expected premium lesson recap output with correct colors/logo/reference format.

Root cause pattern:
- the agent had some premium-packet instructions and local renderer tooling;
- but live handoff still let a rough/basic Google Doc masquerade as terminal success;
- the premium packet reference had been remembered in policy text but not converted into a concrete renderer/reference contract;
- no hard final gate required the premium PDF/artifact to be created and surfaced first.

## Durable lesson
For long-form lesson video/audio/transcript sources, do not let a chat recap, markdown draft, or basic Google Doc become the terminal deliverable when the coach expects a premium student packet.

The correct product rule is:
1. identify whether the source is robust long-form lesson material;
2. route to the coach’s approved premium packet class/reference, if one exists;
3. generate canonical structured packet data;
4. render non-empty premium HTML/PDF or equivalent artifact;
5. surface the premium packet first;
6. only then create/share an optional Google Doc companion;
7. if the premium render is blocked, report the exact blocker instead of silently downgrading.

## Coach-specific quarantine
Bryan/Hermitage-specific packet references, logos, colors, and wording are not portable to other coach agents. Other agents should inherit only the generic product discipline above unless their own approved reference/assets are installed.

## Naming
If the operator dislikes a nickname such as “Connor format,” avoid making that the product-facing label. Prefer “premium student lesson packet,” “approved packet reference,” or the coach/club-specific approved name.
