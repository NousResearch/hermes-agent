# Chapter Generation Prompts

## Generation

```text
You are generating YouTube chapter markers from a timestamped transcript.

Rules:
- Return only chapter lines.
- First timestamp must be 00:00.
- Use concise titles.
- Use meaningful topic transitions.
- Do not create chapters for every sentence.
- Do not invent topics.
- Prefer 5-12 chapters for a normal 10-30 minute video.
- Use more chapters only for long videos with clear sections.
- If transcript quality is poor, say so clearly instead of inventing chapters.

Input transcript chunks:
{chunks}

Return:
00:00 ...
```

## Repair

```text
The chapter list below failed validation.

Validation errors:
{errors}

Chapter list:
{chapters}

Fix the chapter list while preserving only transcript-supported topics.
Return only valid chapter lines.
```
