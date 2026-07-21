---
name: me-txt
description: Create, fetch, and validate me.txt personal identity files for AI agents
version: 1.0.0
author: me-txt
license: MIT
metadata:
  openclaw:
    emoji: "👤"
    homepage: "https://metxt.org"
  hermes:
    tags: [Identity, Personal, Profile, Web, Standards]
    homepage: "https://metxt.org"
---

# me.txt — Personal Identity File

me.txt is an open standard for personal identity files. It's a markdown file at a site root (`yoursite.com/me.txt`) that tells AI agents who someone is.

Spec: https://metxt.org/spec

## When to use this skill

- User asks to create a me.txt file
- User asks to set up their personal identity file for AI
- User wants to add a me.txt to their website
- User asks to fetch or look up someone's me.txt

## Creating a me.txt

Generate a `me.txt` file following this format:

```markdown
# Full Name

> One-line summary of who you are and what you do

![Full Name](https://example.com/avatar.jpg)  <!-- optional -->

## Now

- Current projects, focus areas, what you're working on

## Skills

- Core competencies and expertise

## Stack

- Technologies, tools, and languages

## Work

- [Project Name](url): Description
- Company Name: Role

## Writing

- [Post Title](url): Description

## Links

- [GitHub](https://github.com/username)
- [X](https://x.com/username)
- [Website](https://example.com)
- [Email](mailto:you@example.com)

## Preferences

- Communication: Async-first, email, GitHub issues, etc.
- Timezone: e.g. US Pacific (UTC-8)
- Response time: e.g. Within 24-48 hours
```

### Rules

1. The file MUST start with an H1 (`#`) containing the person's full name.
2. A blockquote summary (`>`) should follow immediately after the H1.
3. An optional markdown image for avatar goes after the summary, before sections.
4. Sections use H2 (`##`). Standard sections: Now, Skills, Stack, Work, Writing, Talks, Links, Preferences, Optional.
5. Links use markdown format: `- [Title](url): Optional description`
6. Keep it concise. Target: under 500 lines, under 2,000 tokens. Ideal: 100-200 lines.
7. Only include sections that have content. Skip empty sections.
8. Save the file as `me.txt` at the project or site root.

### Gathering info

Ask the user for:
- Their name and a one-line summary
- What they're currently working on (Now)
- Their key skills
- Notable projects or work
- Links (GitHub, X, website, email)
- Communication preferences and timezone

If the user has a GitHub profile, offer to pre-fill from it using: `npx create-me-txt --github username`

## Fetching a me.txt

To look up someone's me.txt, try these URLs in order:

1. `https://domain.com/me.txt`
2. `https://domain.com/.well-known/me.txt`
3. `https://metxt.org/api/lookup?domain=domain.com` (directory fallback)

Or use the CLI: `npx create-me-txt` then run `me-txt fetch domain.com`

## Validating a me.txt

Check that the file:
- Starts with an H1 name
- Has a blockquote summary
- Contains at least one H2 section
- Uses standard section names where possible
- Stays under 2,000 tokens

Or use the CLI: `me-txt lint me.txt`

## File placement

The me.txt should be served at one of:
- `https://yoursite.com/me.txt` (preferred)
- `https://yoursite.com/.well-known/me.txt` (alternative)

For static site generators, place it in the `public/` or `static/` directory.
