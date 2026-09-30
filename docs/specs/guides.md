# Guides

Ship three guide roots:

- `guides/employee`: native Hermes self-reference skill copied as a guide,
  including references/templates, with only runtime-specific adaptations.
- `guides/responsibility-authoring`: responsibility packages, schedules and
  webhook authoring.

- `guides/file-keeping`: documents/repos under the active Hermes home, lasting
  attachments, cloud-document stubs and reusable filing conventions.

The standalone connections guide remains removed. Its approved documentation
rule lives in `employee/references/service-connections.md`: read the existing
manual, document every connection and verification, and keep reusable scripts
and references alongside it without secrets.

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
