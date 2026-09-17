#!/usr/bin/env bash
# Bootstrap a project with GitHub Spec Kit (constitution -> specify -> plan
# -> tasks -> implement -> converge), installed as real Claude-Code-compatible
# skills, translated into Hermes-native skills automatically, then layered
# with this skill's worldview/constitution/EARS additions. One call does the
# whole setup -- idempotent: safe to re-run against an already-initialized
# project (translation re-runs too, so upgrading specify-cli and re-running
# this script re-syncs the Hermes-format skill copies).
#
# Usage: init_project.sh <project-name> [--here]
#   <project-name>  Directory to create (or --here to init in cwd).
set -euo pipefail

PROJECT_NAME="${1:-}"
HERE_FLAG="${2:-}"
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "$PROJECT_NAME" && "$HERE_FLAG" != "--here" ]]; then
  echo "Usage: init_project.sh <project-name> | init_project.sh . --here" >&2
  exit 1
fi

if ! command -v specify >/dev/null 2>&1; then
  echo "== Installing specify-cli via uv =="
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install uv first: https://docs.astral.sh/uv/" >&2
    exit 1
  fi
  uv tool install specify-cli
fi

echo "== Running specify init =="
if [[ "$HERE_FLAG" == "--here" ]]; then
  specify init --here --integration claude --non-interactive
  TARGET_DIR="$(pwd)"
else
  specify init "$PROJECT_NAME" --integration claude --non-interactive
  TARGET_DIR="$(pwd)/$PROJECT_NAME"
fi

echo "== Seeding constitution defaults =="
CONST_FILE="$TARGET_DIR/.specify/memory/constitution.md"
if [[ -f "$CONST_FILE" ]]; then
  cp "$SKILL_DIR/templates/constitution-defaults.md" "$CONST_FILE"
  echo "Copied constitution-defaults.md over the blank template."
  echo "Next: run /speckit-constitution in Claude Code to refine per-project specifics."
else
  echo "WARNING: $CONST_FILE not found; specify init may have changed layout." >&2
fi

echo "== Seeding worldview.md (only if this is the first project using it) =="
DOCS_DIR="$TARGET_DIR/docs"
mkdir -p "$DOCS_DIR"
if [[ ! -f "$DOCS_DIR/worldview.md" ]]; then
  cp "$SKILL_DIR/templates/worldview.md" "$DOCS_DIR/worldview.md"
  echo "Copied worldview.md skeleton to docs/worldview.md -- fill in gradually via conversation, do not rush it."
else
  echo "docs/worldview.md already exists, leaving it alone."
fi

echo "== Seeding philosophical-preamble.md (per-project, always seeded if absent) =="
if [[ ! -f "$DOCS_DIR/philosophical-preamble.md" ]]; then
  cp "$SKILL_DIR/templates/philosophical-preamble.md" "$DOCS_DIR/philosophical-preamble.md"
  echo "Copied philosophical-preamble.md skeleton to docs/philosophical-preamble.md -- answer why THIS project exists."
else
  echo "docs/philosophical-preamble.md already exists, leaving it alone."
fi

CLAUDE_MD="$TARGET_DIR/CLAUDE.md"
for IMPORT_LINE in "@docs/worldview.md" "@docs/philosophical-preamble.md"; do
  if [[ -f "$CLAUDE_MD" ]]; then
    if ! grep -qF "$IMPORT_LINE" "$CLAUDE_MD"; then
      printf '\n%s\n' "$IMPORT_LINE" >> "$CLAUDE_MD"
      echo "Appended $IMPORT_LINE to CLAUDE.md."
    else
      echo "CLAUDE.md already imports $IMPORT_LINE."
    fi
  else
    printf '%s\n' "$IMPORT_LINE" > "$CLAUDE_MD"
    echo "Created CLAUDE.md with $IMPORT_LINE."
  fi
done

echo "== Copying EARS reference material =="
mkdir -p "$TARGET_DIR/docs/spec-driven"
cp "$SKILL_DIR/references/ears-syntax.md" "$TARGET_DIR/docs/spec-driven/ears-syntax.md"
cp "$SKILL_DIR/references/ears-schema.json" "$TARGET_DIR/docs/spec-driven/ears-schema.json"

cp "$SKILL_DIR/references/contract-testing.md" "$TARGET_DIR/docs/spec-driven/contract-testing.md"

echo "== Translating speckit-* skills into Hermes frontmatter =="
python3 "$SKILL_DIR/scripts/translate_speckit_skills.py" "$TARGET_DIR"

echo ""
echo "Done. Next steps in Claude Code, one at a time:"
echo "  0. docs/philosophical-preamble.md -- answer why THIS project exists before ratifying the constitution"
echo "  1. /speckit-constitution -- refine the pre-seeded principles for this project"
echo "  2. /speckit-specify -- describe what to build (EARS lines: see docs/spec-driven/ears-syntax.md)"
echo "  3. /speckit-clarify, /speckit-plan, /speckit-tasks, /speckit-implement, /speckit-converge"
echo "  Route every open question through scripts/spec_decision_gate.py first -- see this skill's SKILL.md."
echo "  4. Once requirements are agreed, promote EARS lines to JSON (docs/spec-driven/ears-schema.json)"
echo "     and run scripts/generate_property_tests.py to scaffold property-based tests."
echo "  5. If this project has a client/server or multi-consumer boundary, see"
echo "     docs/spec-driven/contract-testing.md and scripts/generate_mock_server.py."
