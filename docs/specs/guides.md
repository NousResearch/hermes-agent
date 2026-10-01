# Guides

Ship four guide roots:

- `guides/employee`: native Hermes self-reference skill copied as a guide,
  including references/templates, with only runtime-specific adaptations.
- `guides/responsibility-authoring`: responsibility packages, schedules and
  webhook authoring.
- `guides/file-keeping`: documents/repos under the active Hermes home, lasting
  attachments, cloud-document stubs and reusable filing conventions.

- `guides/connections`: top-level connection workflow and native-derived Google
  Workspace, Himalaya email and GitHub setup/usage guides. MCP links to the
  existing native MCP reference in the employee guide.

Connection work now starts at `connections/guide.md`; the old
`employee/references/service-connections.md` is removed. The prompt pointer and
employee/MCP references route to the new guide on every platform, using the
existing conversation-start assembly and cache lifetime. No tool or auth flow
is added. Google helper scripts are unchanged native copies packaged with the
guide; gh and Himalaya remain external CLIs, installed only when needed.

Setup instructions stay in bundled guides. Every connected service has an
editable `$HERMES_HOME/connections/<service>/manual.md` containing verified
**access and usage patterns**, account/resource selection, helper paths and
credential-source names, never reusable secrets. Copy/adapt those patterns from
the guide; link installation and authorization instructions instead of copying
them. References/scripts live beside the manual only when useful.

Without a bundled guide, use official documentation and existing or official
clients, APIs, databases, native MCP or browser access. Infer required inputs
from available context and ask only for missing information. Provision secrets
through native administration, verify with a harmless read, then document usage.
No new setup guide is required. Pending access is not a working connection.
Responsibilities retain duties/authority/state and link service manuals.

The prompt uses native Hermes help wording, substituting the guide and
`read_file` for skill loading. File reads are native, including pagination and
size limits; follow continuation feedback. References use relative guide paths
and `$HERMES_HOME` for profile-owned knowledge. There is no guide-specific
reader, path interpolation or immutability policy.

Native administration, SOUL and provider choices remain documented as native.
Guide exceptions describe disabled skills, the selected model tool surface,
responsibilities, person memory and Hindsight. Do not rename unsupported skill
commands into invented responsibility commands.

[Provenance](reference/README.md) and the checked-in adaptation diffs record
source and edits. The agreed prompt and storage boundaries are in the
[scope ledger](../downstream/scope.md).
