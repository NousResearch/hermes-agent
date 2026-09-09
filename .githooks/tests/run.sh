#!/bin/sh
# .githooks/tests/run.sh — self-contained tests for content-scan.
# Creates throwaway git repos under a temp dir; no network, no effect on this repo.
# Exit 0 = all pass.
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
SCAN="$HERE/../content-scan"
TMP=$(mktemp -d 2>/dev/null || { d=/tmp/nfcs-tests.$$; mkdir -p "$d"; printf '%s' "$d"; })
trap 'rm -rf "$TMP"' EXIT

pass=0 fail=0 skipped=0
ok()   { pass=$((pass+1)); printf '  \033[32mPASS\033[0m %s\n' "$1"; }
bad()  { fail=$((fail+1)); printf '  \033[31mFAIL\033[0m %s\n' "$1"; }
skip() { skipped=$((skipped+1)); printf '  \033[33mSKIP\033[0m %s\n' "$1"; }

# Values that MATCH the tightened patterns but are obviously not real (mixed, no
# "EXAMPLE", no long single-char run). Stand-ins for a real leaked secret. Assembled
# from parts so no committed line contains a full credential shape — this file itself
# must pass both our content-scan AND GitHub's push-protection scanner. The full
# values only ever land in throwaway repos under a temp dir, never in a commit.
FAKE_AWS="AKIA""Z7QK9WPLM2ZR4NTX"
FAKE_GH="ghp_""A1b2C3d4E5f6G7h8I9j0K1L2m3N4o5P6q7R8"
FAKE_SLACK="xoxb-""2109876543-""2109876543-""Ab12Cd34Ef56Gh78Ij90Kl12"

newrepo() {
    d="$TMP/$1"; mkdir -p "$d"
    ( cd "$d" && git init -q && git config user.email t@t && git config user.name t && git config core.autocrlf false )
    printf '%s' "$d"
}

# ---------------------------------------------------------------------------
# 1. A real-shaped secret added then RENAMED then DELETED in later commits must
#    still fail a history-wide --commits scan of the range.
r=$(newrepo hist)
( cd "$r"
  printf 'aws_key = "%s"\n' "$FAKE_AWS" > creds.txt
  git add creds.txt && git commit -qm c1
  git mv creds.txt moved.txt && git commit -qm c2-rename
  git rm -q moved.txt && git commit -qm c3-delete
  printf 'nothing to see\n' > ok.txt && git add ok.txt && git commit -qm c4
)
root=$(cd "$r" && git rev-list --max-parents=0 HEAD)
if ( cd "$r" && "$SCAN" --commits "$root..HEAD" ) >/dev/null 2>&1; then
    bad "secret added-then-removed in the range is still caught by --commits"
else
    ok  "secret added-then-removed in the range is still caught by --commits"
fi
# the final TREE alone is clean (the file is gone at HEAD)
if ( cd "$r" && "$SCAN" --tree HEAD ) >/dev/null 2>&1; then
    ok  "--tree HEAD is clean once the file is deleted"
else
    bad "--tree HEAD is clean once the file is deleted"
fi

# ---------------------------------------------------------------------------
# 2. An allowlisted fixture value passes; the same value without the marker fails.
r=$(newrepo allow)
( cd "$r"
  printf 'FIXTURE_KEY = "%s"   # nf-scan: allow  synthetic value, not a real key\n' "$FAKE_AWS" > fixture.py
  git add fixture.py && git commit -qm fixture
)
if ( cd "$r" && "$SCAN" --tree HEAD ) >/dev/null 2>&1; then
    ok  "inline 'nf-scan: allow' marker lets a confirmed fixture value pass"
else
    bad "inline 'nf-scan: allow' marker lets a confirmed fixture value pass"
fi
r=$(newrepo allow_missing)
( cd "$r"
  printf 'FIXTURE_KEY = "%s"\n' "$FAKE_AWS" > fixture.py
  git add fixture.py && git commit -qm fixture
)
if ( cd "$r" && "$SCAN" --tree HEAD ) >/dev/null 2>&1; then
    bad "the same value WITHOUT the marker is blocked"
else
    ok  "the same value WITHOUT the marker is blocked"
fi

# ---------------------------------------------------------------------------
# 3. GitHub and Slack shapes are caught; placeholders / prose are not.
r=$(newrepo shapes)
( cd "$r"
  printf 'gh=%s\nslack=%s\n' "$FAKE_GH" "$FAKE_SLACK" > t.txt
  git add t.txt && git commit -qm c
)
if ( cd "$r" && "$SCAN" --tree HEAD ) >/dev/null 2>&1; then
    bad "GitHub ghp_ and structured Slack xoxb- tokens are caught"
else
    ok  "GitHub ghp_ and structured Slack xoxb- tokens are caught"
fi
r=$(newrepo placeholders)
( cd "$r"
  {
    printf 'SLACK_BOT_TOKEN=xoxb-your-bot-token-here\n'
    printf 'aws = "AKIAIOSFODNN7EXAMPLE"   # AWS documented example\n'
    printf 'gh  = "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"\n'
    printf 'The notes mention a secret handshake, a token of thanks, and the keys to the office.\n'
  } > doc.md
  git add doc.md && git commit -qm c
)
if ( cd "$r" && "$SCAN" --tree HEAD ) >/dev/null 2>&1; then
    ok  "docs placeholders (xoxb-…-here, AKIA…EXAMPLE, ghp_xxxx) and prose are not flagged"
else
    bad "docs placeholders (xoxb-…-here, AKIA…EXAMPLE, ghp_xxxx) and prose are not flagged"
fi

# ---------------------------------------------------------------------------
# 4. Codex F-03 / ERR-2026-09-07-005 regression. The old --commits scan captured
#    changed paths as newline text and expanded them UNQUOTED, so a path with a
#    space (tab, leading dash, ...) word-split into non-existent pathspecs and its
#    content was never scanned. A planted GitHub-token-shaped value in
#    "dir/file name.txt" passed `--commits` clean. These must now be CAUGHT.

# 4a — THE regression: a secret in a filename containing a space.
r=$(newrepo f03_space)
( cd "$r"
  printf 'base\n' > base.txt && git add base.txt && git commit -qm c0
  mkdir -p d
  printf 'gh = "%s"\n' "$FAKE_GH" > "d/file name.txt"
  git add -A && git commit -qm c1
)
if ( cd "$r" && "$SCAN" --commits "HEAD~1..HEAD" ) >/dev/null 2>&1; then
    bad "F-03: secret in a space-containing filename is caught by --commits"
else
    ok  "F-03: secret in a space-containing filename is caught by --commits"
fi

# 4b — control: a space-containing filename with NO secret is not falsely flagged.
r=$(newrepo f03_space_clean)
( cd "$r"
  printf 'base\n' > base.txt && git add base.txt && git commit -qm c0
  mkdir -p d
  printf 'just some text, no credentials\n' > "d/file name.txt"
  git add -A && git commit -qm c1
)
if ( cd "$r" && "$SCAN" --commits "HEAD~1..HEAD" ) >/dev/null 2>&1; then
    ok  "a space-containing filename with no secret is not falsely flagged"
else
    bad "a space-containing filename with no secret is not falsely flagged"
fi

# 4c — filename that starts with a dash.
r=$(newrepo f03_dash)
( cd "$r"
  printf 'base\n' > base.txt && git add base.txt && git commit -qm c0
  printf 'aws = "%s"\n' "$FAKE_AWS" > -leading-dash.txt
  git add -A && git commit -qm c1
)
if ( cd "$r" && "$SCAN" --commits "HEAD~1..HEAD" ) >/dev/null 2>&1; then
    bad "secret in a filename that starts with '-' is caught"
else
    ok  "secret in a filename that starts with '-' is caught"
fi

# 4d — filename with non-ASCII characters (probe: some filesystems/locales mangle it).
r=$(newrepo f03_unicode)
uni="café-π-Ω.txt"
if ( cd "$r"
     printf 'base\n' > base.txt && git add base.txt && git commit -qm c0
     printf 'gh = "%s"\n' "$FAKE_GH" > "$uni" 2>/dev/null && [ -f "$uni" ] &&
     git add -A && git commit -qm c1 ) 2>/dev/null; then
    if ( cd "$r" && "$SCAN" --commits "HEAD~1..HEAD" ) >/dev/null 2>&1; then
        bad "secret in a filename with non-ASCII characters is caught"
    else
        ok  "secret in a filename with non-ASCII characters is caught"
    fi
else
    skip "non-ASCII filename case (filesystem/locale would not create the name)"
fi

# 4e — tab in a filename (probe: NTFS/Explorer discourage it; CI's ext4 is fine).
r=$(newrepo f03_tab)
tabname=$(printf 'has\ttab.txt')
if ( cd "$r"
     printf 'base\n' > base.txt && git add base.txt && git commit -qm c0
     printf 'aws = "%s"\n' "$FAKE_AWS" > "$tabname" 2>/dev/null && [ -f "$tabname" ] &&
     git add -A && git commit -qm c1 ) 2>/dev/null; then
    if ( cd "$r" && "$SCAN" --commits "HEAD~1..HEAD" ) >/dev/null 2>&1; then
        bad "secret in a filename containing a tab is caught"
    else
        ok  "secret in a filename containing a tab is caught"
    fi
else
    skip "tab-in-filename case (this filesystem will not create the name)"
fi

# 4f — a secret introduced while RENAMING a file INTO a space-containing path,
#      all in one commit: the added post-image must still be scanned.
r=$(newrepo f03_rename)
( cd "$r"
  printf 'base\n' > base.txt && git add base.txt && git commit -qm c0
  printf 'placeholder\n' > moveme.txt && git add moveme.txt && git commit -qm c1
  mkdir -p "dir with space"
  git mv moveme.txt "dir with space/renamed file.txt"
  printf 'gh = "%s"\n' "$FAKE_GH" >> "dir with space/renamed file.txt"
  git add -A && git commit -qm c2
)
if ( cd "$r" && "$SCAN" --commits "HEAD~1..HEAD" ) >/dev/null 2>&1; then
    bad "secret added while renaming into a spaced path is caught"
else
    ok  "secret added while renaming into a spaced path is caught"
fi

# 4g — add-then-delete across the range, adding path has a space: caught at the add.
r=$(newrepo f03_add_delete)
( cd "$r"
  printf 'base\n' > base.txt && git add base.txt && git commit -qm c0
  mkdir -p d
  printf 'slack = "%s"\n' "$FAKE_SLACK" > "d/temp name.txt"
  git add -A && git commit -qm c1-add
  git rm -q "d/temp name.txt" && git commit -qm c2-delete
  printf 'unrelated\n' > later.txt && git add later.txt && git commit -qm c3
)
root=$(cd "$r" && git rev-list --max-parents=0 HEAD)
if ( cd "$r" && "$SCAN" --commits "$root..HEAD" ) >/dev/null 2>&1; then
    bad "secret added (spaced path) then deleted later in the range is still caught"
else
    ok  "secret added (spaced path) then deleted later in the range is still caught"
fi

# ---------------------------------------------------------------------------
if [ "$skipped" -gt 0 ]; then
    printf '\n%s passed, %s failed, %s skipped\n' "$pass" "$fail" "$skipped"
else
    printf '\n%s passed, %s failed\n' "$pass" "$fail"
fi
[ "$fail" -eq 0 ]
