.PHONY: setup-ai test-ai clean-ai
setup-ai:
	@bash scripts/setup-litellm.sh
test-ai:
	@source venv/bin/activate && python ai_router.py
clean-ai:
	@rm -rf venv __pycache__ .litellm_costs.csv
