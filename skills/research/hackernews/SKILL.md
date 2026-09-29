---
name: hackernews
description: Read and search Hacker News stories, comments, and users.
version: 1.0.0
author: Yusuf Öncü (wolfgang1211)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Research, News, Hacker-News, Tech]
    related_skills: [arxiv, blogwatcher]
---

# Hacker News Skill

Read Hacker News front-page lists, individual items, user profiles, and the full searchable archive from the terminal. The helper script uses the public Firebase API for live HN data and the public Algolia HN Search API for archive search. It is read-only: it does not log in, vote, post, or comment.

The bundled helper is dependency-free and uses only the Python standard library, so it works in minimal Hermes installs and across Linux, macOS, and Windows.

## When to Use

- The user asks what is trending, new, best, discussed, or hiring on Hacker News.
- The user wants Show HN, Ask HN, or Who is Hiring/job posts.
- The user wants comments, score, author, URL, or discussion links for a specific HN item.
- The user wants to search old HN stories or comments by keyword, author tag, Show HN/Ask HN tag, or recency.

Don't use this for posting, voting, moderation, or private account data. The skill is intentionally read-only and unauthenticated.

## Prerequisites

None beyond network access. Both APIs are public and keyless:

- Firebase live API: `https://hacker-news.firebaseio.com/v0/`
- Algolia archive API: `https://hn.algolia.com/api/v1/`

## How to Run

Run the helper from the repository root:

```bash
python skills/research/hackernews/scripts/hn.py <command> [options]
```

Output is JSON on stdout. Network/API failures are reported as JSON on stderr with a non-zero exit code.

## Quick Reference

| Command | Purpose |
|---|---|
| `top -n 10` | Top front-page stories |
| `new -n 10` | Newest submissions |
| `best -n 10` | Best stories |
| `ask -n 10` | Ask HN posts |
| `show -n 10` | Show HN posts |
| `jobs -n 10` | Job posts |
| `item <id>` | One story/comment/poll/job by ID |
| `item <id> --comments` | Item plus top-level comments |
| `user <name>` | Public HN user profile |
| `search "<query>" -n 10` | Search stories/comments by relevance |
| `search "<query>" --by-date -n 10` | Search newest first |
| `search "<query>" --tags ask_hn` | Search with Algolia tag filters |

## Procedure

1. Decide whether the user needs a live listing, a specific item/user, or archive search.
2. Keep listing limits modest (`-n 10` to `-n 30`) because each live listing fetches item details one-by-one.
3. Run the matching command, for example:

   ```bash
   python skills/research/hackernews/scripts/hn.py top -n 5
   python skills/research/hackernews/scripts/hn.py search "rust async" --by-date -n 10
   python skills/research/hackernews/scripts/hn.py item 8863 --comments
   python skills/research/hackernews/scripts/hn.py user pg
   ```

4. Parse the JSON. Story-like rows include `id`, `title`, `by`, `score`, `time`, `url`, `hn_url`, and `comments` where available.
5. Summarize the result for the user and include `hn_url` links when discussion context matters.

## Pitfalls

- `hn_url` always points to the Hacker News discussion/item page.
- External story links are in `url`; self posts may not have an external URL.
- `text` is stripped to plain text for readability.
- `item --comments` includes top-level comments only, ordered as HN returns them.
- Search results come from Algolia and may include both stories and comments depending on `--tags`.

1. **Using huge limits.** Live listing commands fetch IDs first, then fetch each item. Keep `-n` low for interactive use.
2. **Mixing relevance and recency.** `search` sorts by relevance; add `--by-date` when the user asks for the latest mentions.
3. **Assuming every item has `url` or `title`.** Comments and some self posts lack one or both. Use `hn_url` as the stable link.
4. **Expecting deleted comments.** Deleted/dead comments are skipped in comment output.
5. **Using it for account actions.** The helper has no auth flow and cannot vote, post, reply, or edit.

## Verification

- [ ] `python skills/research/hackernews/scripts/hn.py top -n 1` returns a JSON array with one item.
- [ ] `python skills/research/hackernews/scripts/hn.py item 8863 --comments` returns an object with `hn_url` and `comments`.
- [ ] `python skills/research/hackernews/scripts/hn.py search "hermes" -n 1` returns a JSON array.
- [ ] Errors render as JSON and exit non-zero.
