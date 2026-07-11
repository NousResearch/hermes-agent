# Output Formats

## Transcript with Metadata Header

Default output includes a header block:

```markdown
# Event ROI Reality Check
URL: https://x.com/i/spaces/1yxBeMYdqgnJN
Duration: 34:51
Host: ipshi86

---

Full transcript text here...
```

## Clean Text (`--text-only`)

Strips metadata header and Whisper timestamps. Good for piping into other tools:

```
Full transcript text here, no headers, no timestamps.
```

## Suggested Summarization Template

After getting the transcript, structure the summary like this:

```markdown
## [Space Title] — Key Takeaways

**Host:** @username
**Guest:** @guestname (Title @ Company)
**Duration:** XX minutes

### Key Points

1. Point one — explanation
2. Point two — explanation
3. Point three — explanation

### Notable Quotes

> "Direct quote from speaker" — @speaker

### Context

2-3 sentences of background on who the speakers are and why this matters.

### Action Items

- What the listener should do with this information
- Related resources, links, or people to follow
```
