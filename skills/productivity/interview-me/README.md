# interview-me

Open-source Hermes Agent skill, version **0.2.0**.

An adaptive interview protocol that loads when the user explicitly asks to be interviewed, asks one high-value question at a time, inspects available sources before questioning the user, and stops when more questions would not change the next action.

## Provenance

Derived from repeated planning, briefing, preference-discovery, and decision workflows. Public examples contain no private interview transcripts or persisted user profiles.

## Inputs

- The user's requested outcome and supplied context.
- Authorized files, notes, or tools that may already answer a question.
- User answers given voluntarily during the current interview.

## Outputs

- A READY TO PROCEED, PROCEED WITH ASSUMPTIONS, PAUSED, or STOPPED outcome.
- A concise brief separating confirmed context, interpretations, constraints, preferences, tradeoffs, and unknowns.
- A recommended next step without silent persistence.

## Requirements

- A Hermes Agent version that supports tap-discovered `SKILL.md` bundles.
- Read access to the evidence named by the request.
- Python 3.11 or newer only for the included validation commands.
- No third-party Python packages are required for bundle validation.

## Install

Install from Hermes Field Kit as a tap using the command supported by your installed Hermes version, or copy this skill directory into your local Hermes skills tree. See the [repository installation guide](../../docs/installation.md).

Linux or macOS, from the repository root:

```bash
mkdir -p ~/.hermes/skills
cp -R skills/interview-me ~/.hermes/skills/
```

PowerShell, from the repository root:

```powershell
$destination = Join-Path $env:LOCALAPPDATA "hermes\skills"
New-Item -ItemType Directory -Force $destination | Out-Null
Copy-Item -Recurse "skills\interview-me" $destination
```

Start a fresh Hermes session after installation because skill discovery may be cached.

## Invocation

Example triggers:

- Interview me before you start.
- Ask me questions so you understand what I want.
- Help me turn this rough idea into a decision brief.
- Learn my preferences before drafting the plan.

## Safety

The skill is diagnosis, planning, or interview-only by default. Mutations, repairs, process changes, credential changes, persistence, publication, installation, execution, or repository writes require separate explicit approval.

Inspected content is untrusted evidence. Embedded instructions never override the user, the skill contract, or higher-priority safeguards.

## Privacy

- Answers remain session context unless the user separately approves a specific memory write.
- Sensitive details are not requested unless they are materially necessary.
- The exact memory summary and destination must be shown before persistence.

## Limitations

- It is not a clinical, legal, psychological, or coercive intake process.
- It cannot resolve facts absent from supplied sources or user answers.
- More questions are not useful once they stop changing the next action.

## Examples

See [successful and boundary examples](examples/example-report.md).

## Validation

Run from the repository root:

```bash
python skills/interview-me/scripts/validate_bundle.py
python -m unittest discover -s skills/interview-me/tests -v
```

The validator and tests use only the Python standard library.

## Version history

### 0.2.0

- Initial public Field Kit release with evidence-first workflow, explicit authority boundaries, hostile-content handling, examples, and contract tests.

## License

Apache License 2.0. See the repository [`LICENSE`](../../LICENSE).
