#!/usr/bin/env bash
set -euo pipefail

PROFILE_NAME="${1:-claudio-lab}"

if ! command -v hermes >/dev/null 2>&1; then
  echo "hermes not found on PATH" >&2
  exit 1
fi

hermes profile create "$PROFILE_NAME" || true
PROFILE_HOME="$HOME/.hermes/profiles/$PROFILE_NAME"
mkdir -p "$PROFILE_HOME"
cp SOUL.md "$PROFILE_HOME/SOUL.md"

PROFILE_SKILLS="$PROFILE_HOME/skills/claudio-authored"
REPO_SKILLS_DIR="$(pwd)/skills/yallaplay"
if [[ -d "$REPO_SKILLS_DIR" ]]; then
  mkdir -p "$PROFILE_SKILLS"
  for skill_dir in "$REPO_SKILLS_DIR"/*; do
    [[ -d "$skill_dir" ]] || continue
    skill_name="$(basename "$skill_dir")"
    target="$PROFILE_SKILLS/$skill_name"
    if [[ -e "$target" || -L "$target" ]]; then
      rm -rf "$target"
    fi
    ln -s "$skill_dir" "$target"
  done

  # Remove old repo-owned YallaPlay symlinks from the former category so
  # `/skills` has one clear visual group for Claudio-authored skills.
  OLD_PROFILE_SKILLS="$PROFILE_HOME/skills/yallaplay"
  if [[ -d "$OLD_PROFILE_SKILLS" ]]; then
    for old_target in "$OLD_PROFILE_SKILLS"/*; do
      [[ -L "$old_target" ]] || continue
      resolved="$(readlink -f "$old_target")"
      case "$resolved" in
        "$REPO_SKILLS_DIR"/*) rm "$old_target" ;;
      esac
    done
    rmdir "$OLD_PROFILE_SKILLS" 2>/dev/null || true
  fi
fi

echo "Created/updated Hermes profile: $PROFILE_NAME"
echo "Linked repo Claudio/YallaPlay skills into: $PROFILE_SKILLS"
echo "Next: $PROFILE_NAME setup"
echo "Then: $PROFILE_NAME chat"
