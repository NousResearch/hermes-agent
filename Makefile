# ============================================================================
# Hermes Agent — Developer Task Runner
# ============================================================================
# Usage:  make <target>
#         make help       # list all targets
#
# This Makefile provides shortcuts for common development tasks.
# For frontend development (TUI / Web Dashboard), use npm scripts directly:
#   cd ui-tui && npm run dev
#   cd web && npm run dev
# ============================================================================

.DEFAULT_GOAL := help

# ── Detect venv (.venv preferred, venv as fallback) ─────────────────────────
VENV_DIR := $(shell [ -d .venv ] && echo .venv || echo venv)
HERMES   := $(VENV_DIR)/bin/hermes

# ── Targets ─────────────────────────────────────────────────────────────────

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

.PHONY: setup
setup: ## First-time setup (install uv, create .venv, install deps)
	./setup-hermes.sh

.PHONY: dev
dev: ## Start interactive chat
	$(HERMES)

.PHONY: test
test: ## Run full test suite
	scripts/run_tests.sh

.PHONY: test-f
test-f: ## Run specific test file/dir (make test-f F=tests/agent/test_foo.py)
	@if [ -z "$(F)" ]; then echo "Usage: make test-f F=tests/agent/test_foo.py"; exit 1; fi
	scripts/run_tests.sh $(F)

.PHONY: lint
lint: ## Run ruff linter
	$(VENV_DIR)/bin/ruff check .

.PHONY: lint-fix
lint-fix: ## Run ruff linter with auto-fix
	$(VENV_DIR)/bin/ruff check . --fix

.PHONY: build
build: build-tui build-web ## Build TUI + Web Dashboard

.PHONY: build-tui
build-tui: ## Build terminal UI
	cd ui-tui && npm run build

.PHONY: build-web
build-web: ## Build web dashboard
	cd web && npm run build

.PHONY: tui-dev
tui-dev: ## Start TUI in dev mode (watch + rebuild)
	cd ui-tui && npm run dev

.PHONY: gateway
gateway: ## Start messaging gateway
	$(HERMES) gateway run

.PHONY: doctor
doctor: ## Diagnose environment issues
	$(HERMES) doctor

.PHONY: clean
clean: ## Remove .venv, __pycache__, build artifacts
	rm -rf .venv venv
	find . -type d -name __pycache__ -not -path '*/node_modules/*' -exec rm -rf {} + 2>/dev/null || true
	rm -rf ui-tui/dist web/dist
