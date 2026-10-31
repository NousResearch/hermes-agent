## Description:

AI Meeting Bot for Jitsi Meet and Google Meet via Habilis MCP Gateway that joins calls, transcribes with diarization, interacts via duplex voice, and generates structured minutes.

This skill is ready for commercial/non-commercial use.

## Publisher:

[rafacpti23](https://clawhub.ai/user/rafacpti23)

### License/Terms of Use:

MIT-0

## Use Case:

External users and teams use this skill to let an agent join Jitsi Meet or Google Meet calls, capture live transcripts with speaker identification, participate by voice when configured, and produce structured meeting minutes.

### Deployment Geography for Use:

Global

## Known Risks and Mitigations:

Risk: The Habilis MCP gateway can process meeting URLs, participant audio, chat, transcripts, and generated minutes.

Mitigation: Use the skill only for meetings where participants are notified and required consent is obtained; avoid confidential or regulated meetings unless the provider's controls are acceptable.

Risk: API tokens and meeting access may grant broad capability if reused across environments.

Mitigation: Use scoped or test tokens where possible and review the skill before installation and use.

## Reference(s):

- [ClawHub skill page](https://clawhub.ai/rafacpti23/skills/meeting-assistant-publish)
- [Habilis MCP Gateway](https://<your-domain>)

## Skill Output:

**Output Type(s):** [Text, Markdown, Shell commands, Configuration, Guidance]

**Output Format:** [Markdown instructions with inline shell and YAML snippets; agent-facing outputs include transcripts and structured meeting minutes.]

**Output Parameters:** [1D]

**Other Properties Related to Output:** [Requires Habilis, Gemini, and OpenAI API keys for the documented gateway and voice workflows.]

## Skill Version(s):

1.0.0 (source: server release metadata and SKILL.md frontmatter)

## Ethical Considerations:

Users should evaluate whether this skill is appropriate for their environment, review any generated or modified files before relying on them, and apply their organization's safety, security, and compliance requirements before deployment.
