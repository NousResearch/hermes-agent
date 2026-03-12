# Regression Policy v1

Promotion gate:
- Run smoke suite on every major change.
- Run full suite before promotion.
- Reject promotion on holdout regression.
- Require pre/post diff for run_agent.py, model_tools.py, agent/, tools/ edits.

Flake control:
- Same-commit rerun required when pass rate delta > 5%.
- Prefer deterministic graders; model-graded tasks must include rationale artifacts.
