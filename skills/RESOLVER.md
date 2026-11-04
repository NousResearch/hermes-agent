# Skill Resolver

Routing table for the skills in this directory. Each row maps trigger phrases
to a skill; on a match, load that skill's `skills/<name>/SKILL.md` before
acting. Rows use the compact form `- **skill-name**: trigger | trigger`.

Only skills that live directly in this directory are routed here. Skills nested
under a category directory are loaded on demand via the `skills_list` and
`skill_view` tools instead.

## Communication

- **email-drafter**: draft an email | compose an email | email this person | reply to this email | create a gmail draft | threaded email reply
- **redraft**: redraft this | make this readable | clean this up | format this properly | restructure this text | tidy up this message

## Workspace administration

- **gws-domain-admin**: act on another draas.com account | domain-wide delegation | read another employee mailbox | admin access to someone else workspace account

## Health and wellbeing

- **google-health**: my sleep data | my step count | my heart rate data | my nutrition log | my exercise history | google health api

## Data retrieval and research

- **monid**: discover a tool for this task | web scraping | structured data retrieval | enrichment lookup | social media data | company or people data | content monitoring endpoint

## Quality assurance

- **dogfood**: qa this web app | exploratory testing pass | find bugs in this site | browser based bug report

## Chat platforms

- **yuanbao**: yuanbao group | at-mention a group member | list yuanbao group members | 元宝 group
