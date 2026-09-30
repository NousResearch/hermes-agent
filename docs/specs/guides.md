# Guides and skills removal

Status: implemented in the codebase; see [implementation map and validation limits](implementation.md).

## Decision

Skills stay out of the employee's model-visible knowledge system. The native
`hermes-agent` self-reference skill is carried over as the shipped
`guides/employee/guide.md`, including its applicable references and templates.
Preserve native wording and structure; adapt only concrete runtime differences.
The former custom employee overview and separate connections guide are removed.
Responsibility authoring and file keeping remain separate guides.

The system prompt reuses native `HERMES_AGENT_HELP_GUIDANCE` verbatim except for
replacing the skill label and `skill_view` invocation with a guide label and
`read_file` pointing to the resolved guide path. The guide routes the agent to
native command, configuration, authentication, MCP and troubleshooting references.
Upstream documentation remains the native reference; the guide identifies fixed
fork differences so it cannot accidentally re-enable excluded capabilities.

Guides are product-owned and read through ordinary file tools. They are not
installable/discoverable skills. Existing file-tool protection prevents editing
them; this does not claim OS-level isolation from unrestricted terminal access.

## Runtime adaptations

| Native guidance | Fork treatment |
| --- | --- |
| Skill installation, authoring, bundles and curator | Remove these instructions; durable owned work uses responsibility packages. Do not mechanically rename unsupported commands. |
| Standalone cron/webhook authoring | Route to responsibility authoring and its schedules/webhooks references. Retain native execution and inspection. |
| SOUL/personality and global personal memory | Use employee name/instructions, shared authored memory, per-person context and fixed Hindsight. Keep the dedicated memory reference. |
| Browser selection | Describe `browser_exec` and Browser Use Cloud in a dedicated reference. |
| Connection setup | Use native `hermes mcp`, service CLI authentication, configured secrets and native dashboard authorization. No hosted connection tool or mcporter dependency. |
| Other native functionality | Preserve the applicable references/templates, with actual tool availability and platform limits stated. |

User-requested service setup through native commands is allowed; it does not
permit arbitrary edits to employee settings, instructions or credentials.
General configuration remains administrator-owned. OAuth approval is completed
by the user; reusable secrets go through native administration or a configured
secret manager, never chat. Interactive/headless limitations must be stated,
not bypassed by hand-writing protected config or token files.

## Service manuals

Service manuals remain employee-owned knowledge under
`<profile-home>/connections/<service>/`, with optional references and scripts.
Removing the connections guide does not delete manuals or revoke access.
The prompt retains the frozen `Service manuals` listing and instruction to read
an existing manual before operating a service. The prompt and native guide route every connection operation through the
service-connections reference and existing manual. Every CLI/API/MCP/browser
connection is documented, including account, access method, credential location
(without values), verification result and useful references/scripts. Pending or
failed setup stays explicitly unverified. Connection changes update the manual.
A missing listing does not prove a manual or login is absent.

## Verification and upstream maintenance

Compare the native guide copy with `skills/autonomous-ai-agents/hermes-agent/`
on upstream sync. Preserve source structure and compatible improvements rather
than rewriting an independent overview. The checked-in reference diffs account
for adaptations. Verify prompt-to-guide-to-reference reads, template packaging,
removed-guide links, fixed surface exclusions, and unchanged warm-session prompt
bytes. Guide reads remain ordinary tool results; never rebuild an existing prompt
mid-conversation to refresh guide or manual discovery.
