#!/usr/bin/env bash
# Detect common repository manifests and print a review plan. This script never executes project commands.

set -u
set -o pipefail

PROJECT_ROOT="${1:-$PWD}"

if [ ! -d "$PROJECT_ROOT" ]; then
  printf 'Error: project root does not exist: %s\n' "$PROJECT_ROOT" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"

find_manifest() {
  local filename="$1"
  find "$PROJECT_ROOT" \
    -type d \( -name .git -o -name node_modules -o -name vendor -o -name target -o -name dist -o -name build \) -prune -o \
    -type f -name "$filename" -print
}

print_rows() {
  local toolchain="$1"
  local filename="$2"
  local workflow="$3"
  local manifests
  local manifest
  local relative

  manifests="$(find_manifest "$filename")"
  [ -n "$manifests" ] || return 0

  while IFS= read -r manifest; do
    relative="${manifest#"$PROJECT_ROOT"/}"
    printf '| %s | `%s` | %s |\n' "$toolchain" "$relative" "$workflow"
    DETECTED=1
  done <<EOF
$manifests
EOF
}

DETECTED=0

printf '# Project Review Detection\n\n'
printf -- '- **Root:** `%s`\n' "$PROJECT_ROOT"
if [ -d "$PROJECT_ROOT/.git" ]; then
  printf -- '- **Git repository:** yes\n'
else
  printf -- '- **Git repository:** no\n'
fi

printf '\n## Detected Toolchains\n\n'
printf '| Toolchain | Evidence | Recommended workflow |\n'
printf '| --- | --- | --- |\n'

print_rows 'Foundry / Solidity' 'foundry.toml' 'Use an installed Foundry-specific review workflow.'
print_rows 'Node.js / TypeScript' 'package.json' 'Inspect package scripts and lockfiles; run declared checks.'
print_rows 'Python' 'pyproject.toml' 'Inspect test/lint configuration; run available local tools.'
print_rows 'Python' 'requirements.txt' 'Inspect dependency and test configuration before choosing checks.'
print_rows 'Rust' 'Cargo.toml' 'Run Cargo test/clippy and installed audit tooling.'
print_rows 'Go' 'go.mod' 'Run Go tests and installed vulnerability tooling.'
print_rows 'Java / Kotlin' 'pom.xml' 'Inspect Maven configuration and run declared tests.'
print_rows 'Java / Kotlin' 'build.gradle' 'Inspect Gradle configuration and run declared tests.'
print_rows 'Ruby' 'Gemfile' 'Inspect Bundler configuration and run declared tests.'
print_rows 'PHP' 'composer.json' 'Inspect Composer configuration and run declared tests.'

if [ "$DETECTED" -eq 0 ]; then
  printf '| Unknown | No supported manifest found | Read the README, CI files, and build scripts before reviewing manually. |\n'
fi

printf '\n## Safe Next Steps\n\n'
printf -- '- Read the README, manifests, lockfiles, and CI configuration before running commands.\n'
printf -- '- Use only commands compatible with a detected toolchain and available on `PATH`.\n'
printf -- '- Do not install dependencies, contact production services, or run deployment scripts without approval.\n'
printf -- '- Record unrun checks and missing tools in the final review.\n'
