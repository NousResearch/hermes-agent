"""The release body GitHub accepts, shared by the stable and canary cuts.

GitHub refuses a release body longer than this with HTTP 422, and the refusal is
not atomic: in the 2026-09-28 canary run the release row was written anyway, so
the tag was already pushed and a draft carrying a stub body was left behind for
the next run to resume. The notes were the whole 12686-commit history, because a
repository with no earlier canary tag and no published stable release has no base
to bound the changelog range.

The stable path refuses an oversized body before it claims an attempt. The canary
path checks the same limit before it tags, for the same reason.
"""

# GitHub refuses a release body longer than this with HTTP 422.
GITHUB_BODY_LIMIT = 125_000
