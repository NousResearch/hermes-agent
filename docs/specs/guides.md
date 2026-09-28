# Guides and skills removal

Status: implemented in the codebase; see [implementation map and validation limits](implementation.md).

## Decision

Remove skills from the employee's model-visible knowledge system. Use the
established employee guides and their reference files verbatim, except where
local execution or explicitly removed capabilities require a change. Do not
summarize, rewrite for style, merge guides or invent replacement doctrine. The
explicit exception is the approved shortened connections guide linked below.

Guides are product-owned instructions read through ordinary file tools, with
entry points referenced by the prompt and links to detailed reference files.
They are not discoverable/installable skills. The employee does not edit its
own product guides; retain that ownership contract without claiming OS-level
protection until the local implementation provides it.

Remove skill listings, skill-tool instructions, skill-authoring/review guidance
and other model-facing routes that would reintroduce the retired primitive.
Retarget learning to the agreed employee knowledge homes. This does not require
indiscriminate deletion of upstream internals used by unrelated functionality.

## Guide treatment

| Guide | Preservation rule |
| --- | --- |
| Responsibility authoring, including schedule/webhook references | Preserve wording and procedures; adapt paths and actual local execution facts. References to unsupported capabilities must be removed or corrected. |
| File keeping | Preserve filing judgment, organization and cloud-document stub conventions. Adapt roots and statements about cache permissions, expiration and backup to actual local behavior. |
| Employee self-reference and its references | Preserve applicable behavior, conversation, delegation and memory instructions. Replace hosted-only operating facts and remove unavailable management actions. |
| Connections and its references | Keep service manuals independently of authentication storage. The shortened local connections guide is approved; omit hosted credential-store, request-link, broker and dashboard mechanics. Adapt references consistently. |
| mcporter | Do not silently ship a guide that claims a bundled mcporter installation or directs all MCP work through it. Native MCP remains supported. Use native MCP instructions; do not ship mcporter-specific setup or impose it as a dependency. |

## Permitted adaptations

Make narrow, reviewable edits for local paths; real host/browser behavior;
startup, sleep and process lifetime; file delivery and retention; and references
to explicitly removed tools or hosted management services. Examples requiring
change include “nothing to install and no local application,” fixed cloud Linux
facts, routine machine replacement, hosted dashboard/billing administration,
`connection` request links and `report_issue` instructions.

Do not infer removal of useful operating knowledge merely because it mentions
a connected service. Removing credential-management tools does not itself decide
the fate of service manuals or the account-selection advice they contain.

## Port verification

Compare every resulting guide and reference with its source text. Each changed
passage must have a concrete local-behavior or agreed-exclusion reason. Preserve
all other wording exactly, check internal links, and verify that prompt/file-tool
entry points load the guides without reintroducing skills. Guide roots follow
[the local layout](local-layout.md); the employee name is configurable. References
must describe the agreed native attachments, delegation and steering behavior,
and Browser Use Cloud, rather than copying incompatible runtime claims.

## Service manuals and prompt discovery

Decision: the employee owns one operating-knowledge folder per service, with
`manual.md`, optional `references/` and `scripts/`. After first verifying access,
it creates the manual if absent and maintains it as it works. Access uses native
CLI authentication, configured secrets, MCP or the browser independently of
these files. There is no generated `credentials.md`.

The [approved connections guide](reference/connections-guide.md) is a versioned
specification asset, not yet a runtime asset. Guide and knowledge roots are agreed in [the local layout](local-layout.md);
replace `<connections-root>` with the resolved profile connections directory
when installing the guide.

Decision: change prompt discovery from `Connected: ...` to
`Service manuals: ...`. The listing is frozen per conversation and means
operating knowledge exists, not that access is currently authenticated or healthy.
Use this instruction in place of the credential-file read requirement:

> Before operating a service, read its manual.md if one exists.
> The listing records operating knowledge, not current access.

Retain the instructions to maintain manuals and to read the connections guide
when establishing access to a new service. Remove instructions to read generated
`credentials.md` or call removed connection tools. A missing entry in the frozen
listing does not establish that no manual exists; the guide instructs the agent
to check the directory.
