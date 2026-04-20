PYTHON ?= python3
LOG_PATH ?= logs/router-decisions.jsonl
CONFIG ?= router_config.yaml

.PHONY: install install-dev test validate-config check route analyze analyze-json suggest suggest-segments propose-patch propose-patch-full demo review

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e .[dev]

test:
	pytest -q

validate-config:
	validate-router-config $(CONFIG)

check: validate-config test

route:
	route-model --task-type coding --mode execute --priority high --has-code --json

analyze:
	analyze-router-telemetry $(LOG_PATH)

analyze-json:
	analyze-router-telemetry $(LOG_PATH) --json

suggest:
	suggest-router-changes $(LOG_PATH)

suggest-segments:
	suggest-router-segments $(LOG_PATH)

propose-patch:
	propose-router-patch $(LOG_PATH) --config $(CONFIG)

propose-patch-full:
	propose-router-patch $(LOG_PATH) --config $(CONFIG) --full-config

demo:
	$(PYTHON) scripts/demo_workflow.py

review:
	@echo "== validate config =="
	@validate-router-config $(CONFIG)
	@echo
	@echo "== telemetry summary =="
	@analyze-router-telemetry $(LOG_PATH)
	@echo
	@echo "== general suggestions =="
	@suggest-router-changes $(LOG_PATH)
	@echo
	@echo "== segment suggestions =="
	@suggest-router-segments $(LOG_PATH)
	@echo
	@echo "== proposed patch =="
	@propose-router-patch $(LOG_PATH) --config $(CONFIG)
