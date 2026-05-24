# Makefile for hermes-agent developmental automation
# Sets up uniform local shortcuts for formatting, linting, and running tests.

# Shell configuration
SHELL := /bin/bash

# Directories
PROJECT_ROOT := $(shell pwd)
VENV_BIN := $(PROJECT_ROOT)/venv/bin

# Commands
PYTHON := $(VENV_BIN)/python
PIP := $(VENV_BIN)/pip
RUFF := $(VENV_BIN)/ruff
PYTEST := $(VENV_BIN)/pytest
DEV_REVIEW := $(shell which dev-review 2>/dev/null || echo "$(HOME)/.local/bin/dev-review")
COMPACT_TRACE := $(PROJECT_ROOT)/scripts/compact_trace.py
RGT := /opt/homebrew/bin/rgt

# Default command: show help instructions
.PHONY: help
help:
	@echo "========================================================================="
	@echo "                 HERMES-AGENT AGENTIC LABS AUTOMATION"
	@echo "========================================================================="
	@echo "Available commands:"
	@echo "  make help         - Display this help banner"
	@echo "  make install      - Install local dependencies in editable mode"
	@echo "  make lint         - Check syntax, style, and structural anomalies (ruff)"
	@echo "  make format       - Autofix styling layout and imports (ruff format)"
	@echo "  make review       - Execute local CodeRabbit review on uncommitted delta changes"
	@echo "  make review-all   - Execute local CodeRabbit review on committed + uncommitted deltas"
	@echo "  make test         - Run the entire automated test suite (isolated parallel)"
	@echo "  make test-smoke   - Run a rapid, self-contained unit test (smoke verification)"
	@echo "  make debug-test   - Run full tests with express-assert, clarity-diffS, and clean trace compression"
	@echo "  make debug-smoke  - Run smoke tests with express-assert, clarity-diffS, and clean trace compression"
	@echo "  make rgt-log      - Display Regent VCS session and transaction step logs"
	@echo "  make rgt-status   - Display Regent VCS workspace audit states"
	@echo "  make clean         - Flush caches, build artifacts, and intermediate residues"
	@echo "========================================================================="

.PHONY: install
install:
	@echo "⚙️ Installing dependencies in editable dev-mode..."
	$(PIP) install -e .

.PHONY: lint
lint:
	@echo "🔍 Inspecting syntax and patterns with Ruff..."
	$(RUFF) check .

.PHONY: format
format:
	@echo "🎨 Beautifying styles and layout with Ruff..."
	$(RUFF) format .
	$(RUFF) check --fix .

.PHONY: test
test:
	@echo "🚀 Initiating full parallel testing workspace via run_tests.sh..."
	bash scripts/run_tests.sh

.PHONY: test-smoke
test-smoke:
	@echo "🧪 Running fast smoke-test (acp method suppression test)..."
	$(PYTEST) tests/acp/test_ping_suppression.py -v --tb=short

.PHONY: review
review:
	@echo "🐇 Fetching CodeRabbit review on local changes..."
	$(DEV_REVIEW) review --plain --type uncommitted

.PHONY: review-all
review-all:
	@echo "🐇 Fetching CodeRabbit review on all uncommitted + committed branch deltas..."
	$(DEV_REVIEW) review --plain --type all

.PHONY: debug-test
debug-test:
	@echo "🚀 Initiating parallel testing stack with immediate failure tracking and traceback pruning..."
	bash scripts/run_tests.sh -- --instafail --diff-width=120 | $(PYTHON) $(COMPACT_TRACE)

.PHONY: debug-smoke
debug-smoke:
	@echo "🧪 Running unit debugging on smoke target with immediate assertion visualization..."
	$(PYTEST) tests/acp/test_ping_suppression.py -vv --instafail --diff-width=120 | $(PYTHON) $(COMPACT_TRACE)

.PHONY: rgt-log
rgt-log:
	@echo "🛡️ Fetching Regent VCS agent conversation and implementation transaction logs..."
	HOME=/Users/felipelamartine $(RGT) log

.PHONY: rgt-status
rgt-status:
	@echo "🛡️ Auditing Regent VCS repository workspace status..."
	HOME=/Users/felipelamartine $(RGT) status

.PHONY: rgt-blame
rgt-blame:
	@if [ -z "$(FILE)" ]; then \
		echo "❌ Please specify a file target to blame, e.g.: make rgt-blame FILE=src/main.py"; \
		exit 1; \
	fi
	@echo "🛡️ Fetching line-by-line line-provenances for $(FILE)..."
	HOME=/Users/felipelamartine $(RGT) blame $(FILE)

.PHONY: clean
clean:
	@echo "🧹 Flushing cache files and compile residues..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "✨ Cleaned."
