# Employee system prompt

Status: preservation policy agreed; runtime implementation pending.

## Decision

The employee name is configurable. Preserve the established employee prompt
verbatim for the most part, including its working principles, ownership,
persistence, verification, communication and action-authorization instructions.
Do not rewrite or shorten those instructions as part of the local port.

Use the configured employee name consistently in identity and self-reference.
The opening identity is `You are {employee_name}, an AI employee.`; do not
hardcode Alfred Pierce or imply Actum provenance for this fork. Preserve
configured organization/persona guidance rather than treating example company
text as a literal product default.

## Necessary adaptations

- Render real profile, working-directory and shipped-guide paths according to
  [the local layout](local-layout.md). Remove fixed cloud-machine claims and
  describe the actual execution environment.
- Use the agreed `Service manuals` listing and guide instructions from
  [guides and skills removal](guides.md), without credential-store or removed
  connection-tool instructions.
- Apply the agreed tool exclusions and authored-memory placement from
  [the tool surface](tool-surface.md). Personal profiles belong after the
  current user message, not in the system prompt.
- Remove references to unavailable hosted administration or excluded tools.
  Preserve surrounding behavior where it still applies.
- Retain native attachment, delegation and mid-turn steering contracts. In
  particular, do not copy an employee-only steering marker into a prompt when
  the native runtime emits a different marker. Product wording describes the
  actual chosen runtime. Browser instructions describe Browser Use Cloud.

Each other proposed wording change requires a concrete behavior mismatch to
resolve; stylistic preference is not a reason to diverge. Keep the rendered
system prompt and discovery listings stable for a conversation under the
existing cache rules. This decision does not import unrelated hosted refresh
policies or fixed hosted model/provider settings.

## Port review

Present the exact prompt diff against the reference before implementation is
considered complete. Account for every changed passage with an agreed local
adaptation. Verify actual prompt assembly, not only a documentation snapshot.
