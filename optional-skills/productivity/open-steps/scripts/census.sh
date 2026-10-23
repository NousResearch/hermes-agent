#!/usr/bin/env bash
# The measured half of os-big-picture. Nothing here is a guess: every number
# comes out of git. Prose, stages and the queue stay the skill's job.
#
#   bash scripts/census.sh [repo-path]
#
# It prints three kinds of line, tab separated:
#
#   MEASURED  <date>
#   AGE       <days> <young|ok> <date liveness starts meaning something>
#   PART      <path> <last worked on> <commits in the window> <wired|orphan> <signal>
#
# The signal is one of `active`, `stable`, `unused - N months`. It is the
# combination of two things and never the date alone: quiet code that is still
# reached by something is finished, not dead.
#
# OS_QUIET_MONTHS moves the window; the tests use it, people should not.
set -u

cd "${1:-.}" 2>/dev/null || { echo "no such directory: ${1:-.}" >&2; exit 1; }
git rev-parse --git-dir >/dev/null 2>&1 || { echo "not a git repository" >&2; exit 1; }

MONTHS="${OS_QUIET_MONTHS:-6}"
SINCE="$MONTHS months ago"
NOW="$(date +%s)"
# 30-day months and a 183-day half-year, on purpose: this is a staleness
# signal, not an anniversary. Exactness here would buy nothing.
WINDOW=$((MONTHS * 2592000))

# A copied-in dependency has one commit, years old, and is not ours to retire.
# Build output is not a feature either. Both come off the map before anything
# is measured, so they can never appear as a retire candidate.
SKIP='^(node_modules|vendor|third_party|third-party|dist|build|target|\.git|\.github)$'

# date(1) splits down the middle here: BSD wants -r, GNU wants -d @.
on_day() { date -r "$1" +%Y-%m-%d 2>/dev/null || date -d "@$1" +%Y-%m-%d; }

printf 'MEASURED\t%s\n' "$(on_day "$NOW")"

# How old the repository itself is. A project younger than the quiet window has
# no quiet code yet - by definition, not by luck - so the liveness column
# cannot mean anything until it is old enough, and the map has to say so.
BORN="$(git log --max-parents=0 --format=%ct 2>/dev/null | sort -n | head -1)"
if [ -n "$BORN" ]; then
  DAYS=$(((NOW - BORN) / 86400))
  if [ "$((NOW - BORN))" -lt "$WINDOW" ]; then
    printf 'AGE\t%s days\tyoung\t%s\n' "$DAYS" "$(on_day "$((BORN + WINDOW))")"
  else
    printf 'AGE\t%s days\tok\t-\n' "$DAYS"
  fi
fi

# Enumerated, never typed from memory: a hand-written list of directories is
# the most common way this skill produces a confident, wrong map.
parts() {
  git ls-files | awk -F/ 'NF>1 {print $1}' | sort -u
  git ls-files | awk -F/ 'NF==1 {print}' | sort
}

parts | while IFS= read -r p; do
  [ -n "$p" ] || continue
  printf '%s' "$p" | grep -Eq "$SKIP" && continue
  # Repository housekeeping is not a feature: a licence, a readme, an ignore
  # file. Nothing names them, so they would read as unused on any repository
  # older than the window, and "retire .gitignore" is the confident, wrong map.
  case "$p" in
    .*|LICENSE*|LICENCE*|README*|CHANGELOG*|CONTRIBUTING*|CODE_OF_CONDUCT*|SECURITY*)
      [ -f "$p" ] && continue ;;
  esac

  last="$(git log -1 --format=%ad --date=short -- "$p" 2>/dev/null)"
  [ -n "$last" ] || continue
  lastct="$(git log -1 --format=%ct -- "$p" 2>/dev/null)"
  busy="$(git log --oneline --since="$SINCE" -- "$p" 2>/dev/null | grep -c . || true)"
  : "${busy:=0}"

  # Wiring: does anything OUTSIDE this part name it. One search covers both
  # kinds of part - code somebody imports, and a standalone thing nothing
  # imports but something builds, ships or documents. A directory is searched
  # by its name, a loose file by its name without the extension.
  name="${p%/}"
  case "$name" in */*) name="${name##*/}" ;; esac
  if [ -f "$p" ]; then base="${name%.*}"; [ -n "$base" ] && name="$base"; fi
  wired=orphan
  if [ -n "$name" ] && git grep -l -F -e "$name" -- . ":(exclude)$p" >/dev/null 2>&1; then
    wired=wired
  fi

  if [ "$busy" -gt 0 ]; then
    signal="active"
  elif [ "$wired" = wired ]; then
    signal="stable"
  else
    signal="unused - $(((NOW - lastct) / 2592000)) months"
  fi

  printf 'PART\t%s\t%s\t%s\t%s\t%s\n' "$p" "$last" "$busy" "$wired" "$signal"
done
