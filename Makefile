.PHONY: setup test lint build clean install help

VENV ?= .venv
PYTHON ?= python3

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Install all dependencies
	pip install uv
	$(PYTHON) -m uv sync --locked --extra all --extra dev
	cd ui-tui && npm install
	cd website && npm install

test: ## Run test suite (pass args via ARGS, e.g. make test ARGS="tests/agent/ -v")
	scripts/run_tests.sh $(ARGS)

lint: ## Run linting checks
	ruff check .
	ruff format --check .

lint-fix: ## Auto-fix linting issues
	ruff check --fix .
	ruff format .

typecheck: ## Run type checking
	ty check hermes_cli/ agent/ tools/

build: ## Build package
	uv build

clean: ## Clean build artifacts
	rm -rf dist/ build/ *.egg-info .pytest_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

install: ## Install in editable mode
	pip install -e ".[all]"

dev: setup install ## Full dev setup
	@echo "Hermes Agent dev environment ready. Run 'make test' to verify."

.PRECIOUS: $(VENV)
