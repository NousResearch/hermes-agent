#!/bin/sh
# .githooks/tests/run.sh — self-contained tests for content-scan.
# Creates throwaway git repos under a temp dir; no network, no effect on this repo.
# Exit 0 = all pass.
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
SCAN="$HERE/../content-scan"
TMP=$(mktemp -d 2>/dev/null || { d=/tmp/nfcs-tests.$$; mkdir -p "$d"; printf '%s' "$d"; })
trap 'rm -rf "$TMP"' EXIT

pass=0 fail=0
ok()   { pass=$((pass+1)); printf '  \033[32mPASS\033[0m %s\n' "$1"; }
bad()  { fail=$((fail+1)); printf '  \033[31mFAIL\033[0m %s\n' "$1"; }

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
printf '\n%s passed, %s failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
