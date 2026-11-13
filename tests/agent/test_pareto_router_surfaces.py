"""``openrouter.min_coding_score`` reaches every AIAgent construction surface (#153086).

A surface that builds an agent with the provider-routing kwargs but not the Pareto floor sends
``openrouter/pareto-code`` requests without the ``pareto-router`` plugin, and OpenRouter then
routes to its top (most expensive) tier.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {"tests", "evals", ".venv", "node_modules", ".worktrees", "website", "apps", "ui-tui"}
# batch_runner's CLI passes only command-line routing flags (no config read; no score flag).
EXEMPT = {"batch_runner.py"}


def _routing_call_sites():
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if rel.parts[0] in SKIP_DIRS or str(rel) in EXEMPT:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                names = {kw.arg for kw in node.keywords if kw.arg}
                if "provider_sort" in names:
                    yield f"{rel}:{node.lineno}", names


# Surfaces that build agent kwargs as a dict literal (no provider_routing of their own) but still read
# the profile config: the floor must be one of the literal's keys.
DICT_SURFACES = ("gateway/platforms/api_server.py", "acp_adapter/session.py")


def _dict_literal_keys(rel):
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            yield {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}


def test_every_routing_agent_construction_passes_the_pareto_floor():
    sites = list(_routing_call_sites())
    assert len(sites) >= 6, sites  # CLI, TUI, gateway turn, /bg, cold review, TUI side, cron
    missing = sorted(site for site, names in sites if "openrouter_min_coding_score" not in names)
    for rel in DICT_SURFACES:  # API server / ACP build their kwargs as a dict literal
        agent_dicts = [keys for keys in _dict_literal_keys(rel) if {"platform", "session_id"} <= keys]
        assert agent_dicts, rel
        if not any("openrouter_min_coding_score" in keys for keys in agent_dicts):
            missing.append(rel)
    assert not missing, (
        "AIAgent construction(s) pass provider_routing but drop openrouter.min_coding_score "
        f"(use agent.pareto_router.min_coding_score): {missing}")
