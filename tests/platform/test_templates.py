"""Business templates: each one produces a bundle that loads, applies and is budgeted.

A template that ships broken is worse than none — the first thing a new client sees is an
error. So every template in the library is filled in and loaded here as a real bundle,
then applied to a runtime, on every test run.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from nova import templates
from nova.errors import NovaError, SpecError

REPO = Path(__file__).resolve().parents[2]
TEMPLATE_IDS = [t.id for t in templates.catalogue()]
GOOD = {"tenant_id": "acme-shop", "company": "Acme Shop", "support_email": "help@acme.example"}


@pytest.fixture(autouse=True)
def catalogue(tmp_path_factory, monkeypatch):
    """The MCP catalogue as the image composes it, so HubSpot and Sheets grants validate."""
    composed = tmp_path_factory.mktemp("catalogue")
    shutil.copytree(REPO / "optional-mcps", composed, dirs_exist_ok=True)
    shutil.copytree(REPO / "nova" / "mcp-catalog", composed, dirs_exist_ok=True)
    monkeypatch.setenv("HERMES_OPTIONAL_MCPS", str(composed))


def test_there_are_templates():
    assert set(TEMPLATE_IDS) >= {"ecommerce-support", "real-estate-leads", "invoice-followup"}


@pytest.mark.parametrize("template_id", TEMPLATE_IDS)
def test_a_template_becomes_a_bundle_that_loads(template_id, tmp_path):
    bundle = templates.new_bundle(template_id, tmp_path / "b", {**GOOD, "company": "Acme & Sons (UK)"})
    assert bundle.tenant_id == "acme-shop"
    assert bundle.organization.legal_name == "Acme & Sons (UK)"
    assert len(bundle.agents) >= 2
    leftovers = [p for p in (tmp_path / "b").rglob("*") if p.is_file() and "{{" in p.read_text()]
    assert not leftovers, f"unfilled placeholders in {leftovers}"


@pytest.mark.parametrize("template_id", TEMPLATE_IDS)
def test_every_template_is_budgeted_and_governed(template_id, tmp_path):
    bundle = templates.new_bundle(template_id, tmp_path / "b", {**GOOD, "monthly_budget": "90"})
    assert bundle.deployment.monthly_budget_usd == 90.0
    for agent in bundle.agents:
        assert agent.limits.monthly_budget_usd, f"{agent.id} has no monthly budget"
        assert "terminal" in agent.tools.deny and "execute_code" in agent.tools.deny
    assert bundle.policy.unlisted_tool == "deny"
    assert any(a.requires_approval for a in bundle.policy.actions.values()), "risky actions need approval"
    assert bundle.knowledge.sources and bundle.automations and bundle.objectives


@pytest.mark.parametrize("template_id", TEMPLATE_IDS)
def test_a_template_bundle_applies_to_a_runtime(template_id, tmp_path, runtime, audit):
    from nova.apply import apply_bundle

    bundle = templates.new_bundle(template_id, tmp_path / "b", GOOD)
    report = apply_bundle(bundle, runtime, audit=audit)
    assert {r.agent_id for r in report.agents} >= {a.id for a in bundle.agents}


@pytest.mark.parametrize("template_id", TEMPLATE_IDS)
def test_templates_carry_no_secrets(template_id):
    text = "\n".join(p.read_text() for p in templates.get(template_id).path.rglob("*") if p.is_file())
    for pattern in (r"xox[bpa]-", r"ghp_", r"sk-", r"AKIA[0-9A-Z]{16}", r"\d{8,10}:[A-Za-z0-9_-]{30,}"):
        assert not re.search(pattern, text), pattern


@pytest.mark.parametrize("field, value, words", [
    ("company", 'Acme: "Ltd"', "company"),
    ("company", "Acme\nevil: true", "company"),
    ("tenant_id", "Acme_Shop", "tenant_id"),
    ("support_email", "not-an-email", "email"),
    ("timezone", "Mars/Olympus", "time zone"),
    ("monthly_budget", "0", "more than zero"),
    ("region", "london", "region"),
])
def test_a_bad_value_is_refused_and_nothing_is_left_behind(field, value, words, tmp_path):
    destination = tmp_path / "b"
    with pytest.raises(SpecError, match=words):
        templates.new_bundle("ecommerce-support", destination, {**GOOD, field: value})
    assert not destination.exists()


def test_a_non_empty_destination_is_refused(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "keep.txt").write_text("mine")
    with pytest.raises(NovaError, match="not empty"):
        templates.new_bundle("ecommerce-support", tmp_path / "b", GOOD)
    assert (tmp_path / "b" / "keep.txt").read_text() == "mine"


def test_an_unknown_template_names_the_ones_there_are(tmp_path):
    with pytest.raises(NovaError, match="ecommerce-support"):
        templates.new_bundle("pizza-shop", tmp_path / "b", GOOD)


def test_the_cli_writes_a_validated_bundle(tmp_path, capsys):
    from nova.cli import main

    assert main(["template", "list"]) == 0
    assert "real-estate-leads" in capsys.readouterr().out
    assert main(["template", "new", "invoice-followup", str(tmp_path / "b"), "--tenant", "acme-ar",
                 "--company", "Acme Accounts", "--email", "ar@acme.example", "--budget", "60"]) == 0
    out = capsys.readouterr().out
    assert "validated" in out and "GOOGLE_SHEETS_SERVICE_ACCOUNT_B64" in out


def test_the_image_keeps_the_templates_markdown():
    """.dockerignore drops *.md from the build context; a template's prompts and knowledge are
    Markdown, so without the exception `nova template new` in the image wrote bundles whose
    prompt files were missing."""
    lines = [line.strip() for line in (REPO / ".dockerignore").read_text().splitlines()]
    assert "*.md" in lines
    assert lines.index("!nova/templates/library/**") > lines.index("*.md"), (
        "the exception must come after *.md — .dockerignore applies the last matching rule"
    )
