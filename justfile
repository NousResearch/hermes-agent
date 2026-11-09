set default-list
set shell := ["bash", "-euo", "pipefail", "-c"]
set windows-shell := ["bash", "-euo", "pipefail", "-c"]

# Provision the PM tools, Python runtime and test environment
[group('setup')]
[unix]
setup:
    bash setup-hermes.sh --runtime-only --test-environment

# Provision the PM tools, Python runtime and test environment
[group('setup')]
[windows]
setup:
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File setup-hermes.ps1 -RuntimeOnly

# Install the JS workspaces from pnpm-lock.yaml
[group('setup')]
js-install:
    pnpm install --frozen-lockfile

# Run every blocking lint check CI runs (`just check --only health`)
[group('check')]
[positional-arguments]
check *args:
    scripts/run-in-hermes-env python scripts/check "$@"

# Apply ruff and JS workspace autofixes to the tree
[group('check')]
fix:
    scripts/run-in-hermes-env python -m scripts.code_health.ruff_runner check --fix .
    pnpm -r fix

# Run every JS workspace check
[group('check')]
check-js:
    pnpm -r check

# Run the named test files or directories (pytest flags pass through)
[group('test')]
[positional-arguments]
test +args:
    scripts/run_tests.sh --require-paths "$@"

# Run the full Python test suite (about 34k tests)
[group('test')]
test-all:
    scripts/run_tests.sh

# Run every JS workspace's tests (builds @hermes/ink first, as the TUI's check does)
[group('test')]
test-js:
    pnpm --dir ui-tui run build:ink
    pnpm -r test

# Start the desktop app in development mode
[group('dev')]
desktop:
    pnpm --dir apps/desktop run dev

# Start the TUI in watch mode
[group('dev')]
tui:
    pnpm --dir ui-tui run dev

# Run a command against a throwaway HERMES_HOME (scripts/dev-sandbox.sh)
[group('dev')]
[positional-arguments]
sandbox *args:
    scripts/dev-sandbox.sh "$@"

# Regenerate logos and icons from assets/
[group('dev')]
icons:
    node scripts/generate-icons.mjs

# Install the website's npm dependencies
[group('docs')]
docs-install:
    npm ci --prefix website

# Build the English docs site
[group('docs')]
docs-build:
    npm run build:fast --prefix website

# Refresh uv.lock after a pyproject.toml change, then re-sync the environment
[group('deps')]
lock:
    scripts/relock.sh

# Pre-commit: the pinned ruff with ruff.pre-commit.toml on the given files
[group('hooks')]
[positional-arguments]
ruff +files:
    scripts/run-in-hermes-env python -m scripts.code_health.ruff_runner check --config ruff.pre-commit.toml "$@"

# Pre-commit: eslint on the given files, from the calling workspace
[group('hooks')]
[no-cd]
[positional-arguments]
eslint +files:
    pnpm exec eslint "$@"
