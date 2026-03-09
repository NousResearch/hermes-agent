---
name: blogwatcher
description: Monitor blogs and RSS/Atom feeds for updates using the blogwatcher CLI. Add blogs, scan for new articles, and track what you've read.
version: 1.0.0
author: community
license: MIT
metadata:
  hermes:
    tags: [RSS, Blogs, Feed-Reader, Monitoring]
    homepage: https://github.com/Hyaxia/blogwatcher
---

# Blogwatcher

Track blog and RSS/Atom feed updates with the `blogwatcher` CLI.

## Prerequisites

- Go installed (`go version` to check)
- Install: `go install github.com/Hyaxia/blogwatcher/cmd/blogwatcher@latest`

## Common Commands

- Add a blog: `blogwatcher add "My Blog" https://example.com`
- List blogs: `blogwatcher blogs`
- Scan for updates: `blogwatcher scan`
- List articles: `blogwatcher articles`
- Mark an article read: `blogwatcher read 1`
- Mark all articles read: `blogwatcher read-all`
- Remove a blog: `blogwatcher remove "My Blog"`

## Example Output

```
$ blogwatcher blogs
Tracked blogs (1):

  xkcd
    URL: https://xkcd.com
```

```
$ blogwatcher scan
Scanning 1 blog(s)...

  xkcd
    Source: RSS | Found: 4 | New: 4

Found 4 new article(s) total!
```

## Notes

- Use `blogwatcher <command> --help` to discover flags and options.
