# X Post Writer

A source-locked Hermes skill for drafting and rewriting short-form X content without inventing experience, benefits, attribution, or technical detail.

## Why it exists

Social-writing prompts often reward confidence and compression, which creates a dangerous failure mode: sparse notes become polished copy containing facts the source never supplied. This skill treats source fidelity as a first-class constraint.

It was derived from a repeatedly used private workflow, then stripped of personal voice rules, account data, internal project routing, real post history, private analytics, and platform-specific assumptions.

## What it handles

- Single posts
- Quote posts
- Replies
- Explicit threads
- Launches and announcements
- Personal stories and milestones
- Fresh-angle repurposing
- Claim verification and unsupported-claim blocking
- User-supplied voice preservation

## Core rule

```text
Sparse source -> concise draft
Rich source -> richer draft
Missing facts -> verify or stop
```

The skill never expands sparse notes with plausible implementation details or inferred benefits.

## Requirements

- Hermes Agent with local skill support
- Access to primary sources when the requested copy contains current or high-risk claims
- No external Python dependencies for validation

## Install

Install from Hermes Field Kit as a tap using the command supported by your Hermes version, or copy the skill directory into your local Hermes skills tree.

Linux or macOS:

```bash
cp -R skills/x-post-writer ~/.hermes/skills/
```

PowerShell:

```powershell
$destination = Join-Path $env:LOCALAPPDATA "hermes\skills"
New-Item -ItemType Directory -Force $destination | Out-Null
Copy-Item -Recurse "skills\x-post-writer" $destination
```

Start a new Hermes session after installation because skill discovery may be cached.

## Typical use

```text
Rewrite this as one X post. Preserve my point and return only the finished copy.
```

```text
Turn these verified release notes into a five-post X thread. Do not add details that are not in the notes.
```

```text
Quote this post and explain why it matters. Do not repeat the embedded link.
```

## Voice customization

The public skill has no personal voice profile. It preserves style and phrasing supplied in the request. See `references/voice-customization.md` for creating a private local override without committing personal preferences to a public repository.

## Claim handling

Risky claims must end as one of:

- verified
- attributed
- user-supplied
- qualified
- removed

When an unsupported claim is the premise of the requested post, the skill asks for a source rather than laundering the claim into confident copy.

## Privacy

Do not place private analytics, revenue, unpublished product details, customer data, or personal voice profiles in this repository. The skill treats such inputs as sensitive and does not authorize publication.

## Limitations

- Writing quality is partly qualitative; deterministic tests cover routing, safety, and structural contracts rather than subjective virality.
- X behavior and limits change. Verify current official sources rather than encoding reach guarantees.
- The skill drafts copy. It does not publish, schedule, scrape, or analyze account performance.
- A source-locked draft may be shorter than requested when the available evidence is sparse.

## Testing

```bash
python skills/x-post-writer/scripts/validate_bundle.py
python -B -m unittest discover -s skills/x-post-writer/tests -v
python scripts/validate.py
python -B -m unittest discover -s tests -v
```

## Version history

### 1.0.0

- Initial public release
- Format and source routing
- Source-lock ledger
- Unsupported-claim hard stop
- Generic voice customization guidance
- Behavior cases and contract tests
