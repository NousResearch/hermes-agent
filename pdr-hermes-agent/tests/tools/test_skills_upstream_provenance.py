"""Optional-skill stubs that pull live third-party content name their upstream repository wherever
they are listed or installed, and never claim Nous Research maintenance."""

from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

SAFE_SKILL_MD = "---\nname: brag\ndescription: demo\n---\n\n# Brag\n\nWrite a short brag document.\n"
UPSTREAM_STUB_MD = (
    "---\nname: brag\ndescription: Brag docs\nmetadata:\n  hermes:\n"
    "    upstream:\n      repo: example-owner/brag\n      path: skills/brag\n---\n\nStub.\n"
)


def _upstream_bundle(monkeypatch, skill_md=SAFE_SKILL_MD):
    from tools.skills_hub_models import SkillBundle
    from tools.skills_hub_official import OptionalSkillSource

    fetched = SkillBundle(
        name="brag", files={"SKILL.md": skill_md}, source="github",
        identifier="example-owner/brag/skills/brag", trust_level="community",
        metadata={"source_url": "https://github.com/example-owner/brag/tree/abc123/skills/brag"})
    source = OptionalSkillSource()
    monkeypatch.setattr(source, "_get_github", lambda: SimpleNamespace(fetch=lambda _ident: fetched))
    bundle = source._fetch_from_upstream({"repo": "example-owner/brag", "path": "skills/brag"}, "creative/brag")
    assert bundle.source == "official" and bundle.trust_level == "trusted"
    return bundle


@pytest.mark.parametrize("upstream", [True, False])
def test_install_panel_names_upstream_repo_and_drops_nous_wording(monkeypatch, upstream):
    from hermes_cli import skills_hub as hub_cli
    from tools.skills_hub_models import SkillBundle

    bundle = (_upstream_bundle(monkeypatch) if upstream else SkillBundle(
        name="brag", files={}, source="official", identifier="official/creative/brag", trust_level="builtin"))
    out = StringIO()
    monkeypatch.setattr(hub_cli, "_confirm_or_cancel", lambda *_a, **_k: False)
    hub_cli._confirm_install(Console(file=out, width=200), bundle, "creative", True)
    text = out.getvalue()

    assert ("https://github.com/example-owner/brag" in text) is upstream
    assert ("official optional skill maintained by Nous Research" in text) is not upstream


@pytest.mark.parametrize(("force", "skip_confirm"), [(True, False), (False, True)])
def test_unprompted_install_still_discloses_upstream_repo(monkeypatch, tmp_path, force, skip_confirm):
    from hermes_cli import skills_hub as hub_cli

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    bundle = _upstream_bundle(monkeypatch, SAFE_SKILL_MD)
    monkeypatch.setattr("tools.skills_sync_bundled_ops.bundled_skill_for_install", lambda _ident: None)
    monkeypatch.setattr(hub_cli, "_pinned_sources", lambda *_a: [])
    monkeypatch.setattr(hub_cli, "_full_identifier", lambda ident, *_a: ident)
    from tools.skills_hub_official import OptionalSkillSource
    monkeypatch.setattr(hub_cli, "_resolve_source_meta_and_bundle",
                        lambda *_a: (None, bundle, OptionalSkillSource()))
    monkeypatch.setattr(hub_cli, "_confirm_install", lambda *_a: pytest.fail("prompt must be skipped"))
    out = StringIO()

    assert hub_cli.do_install("official/creative/brag", category="creative", force=force,
                              skip_confirm=skip_confirm, console=Console(file=out, width=200),
                              invalidate_cache=False) is True
    text = out.getvalue()
    assert "https://github.com/example-owner/brag" in text and "not maintained by Nous Research" in text
    # Last, after the file list: the dashboard's action log shows only the tail of a long install.
    assert text.index("Third-party content") > text.index("Files:")


def test_catalog_lists_upstream_stub_with_its_upstream_repo(monkeypatch, tmp_path):
    from tools.skills_hub_official import OptionalSkillSource

    root = tmp_path / "optional-skills"
    for rel, body in (("creative/brag", UPSTREAM_STUB_MD), ("creative/vendored", SAFE_SKILL_MD)):
        (root / rel).mkdir(parents=True)
        (root / rel / "SKILL.md").write_text(body, encoding="utf-8")
    source = OptionalSkillSource()
    monkeypatch.setattr(source, "_optional_dir", root)
    metas = {m.identifier: m for m in source._scan_all()}

    upstream, vendored = metas["official/creative/brag"], metas["official/creative/vendored"]
    assert (upstream.trust_level, upstream.repo, upstream.path) == ("trusted", "example-owner/brag", "skills/brag")
    assert (vendored.trust_level, vendored.repo) == ("builtin", OptionalSkillSource.OFFICIAL_REPO)


def test_index_fallback_bundle_keeps_upstream_identity(monkeypatch):
    from tools.skills_hub_github import GitHubAuth
    from tools.skills_hub_models import SkillBundle
    from tools.skills_hub_official import HermesIndexSource

    fetched = SkillBundle(name="brag", files={"SKILL.md": SAFE_SKILL_MD}, source="github",
                          identifier="example-owner/brag/skills/brag", trust_level="community")
    src = HermesIndexSource(auth=GitHubAuth())
    src._index, src._loaded = {"skills": [
        {"name": "brag", "source": "official", "identifier": "official/creative/brag",
         "repo": "example-owner/brag", "path": "skills/brag"},
        {"name": "arxiv", "source": "official", "identifier": "official/research/arxiv",
         "repo": "NousResearch/hermes-agent", "path": "optional-skills/research/arxiv"},
    ]}, True
    monkeypatch.setattr(src, "_get_github", lambda: SimpleNamespace(fetch=lambda _ident: SkillBundle(
        **{**fetched.__dict__, "metadata": {}})))

    upstream = src.fetch("official/creative/brag")
    assert (upstream.metadata["upstream_repo"], upstream.metadata["upstream_path"]) == (
        "example-owner/brag", "skills/brag")
    assert "upstream_repo" not in src.fetch("official/research/arxiv").metadata


def test_headless_catalog_install_reports_upstream_repo(monkeypatch, tmp_path):
    from tools.connectors.catalog import HostInstaller, _installed_row
    from tools.skills_hub import HubLockFile

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))

    def do_install(identifier, **_):
        HubLockFile().record_install(
            name="brag", source="official", identifier="official/creative/brag", trust_level="trusted",
            scan_verdict="safe", skill_hash="0" * 64, install_path="creative/brag", files=["SKILL.md"],
            metadata={"upstream_repo": "example-owner/brag", "upstream_path": "skills/brag"})
        return True

    monkeypatch.setattr("hermes_cli.skills_hub.do_install", do_install)
    outcome = HostInstaller().install_skill("official/creative/brag", force=False, source="official")
    assert outcome == {"name": "brag", "already_installed": False, "upstream_repo": "example-owner/brag"}

    row, detail = _installed_row(SimpleNamespace(kind="skill", extra={}), outcome)
    assert row["upstream_repo"] == "example-owner/brag"
    assert "https://github.com/example-owner/brag" in detail and "not maintained by Nous Research" in detail

    # The row rides the gateway's connection.update / operation.status payloads, which forbid unknown keys.
    from tui_gateway.contracts.connectors_operation import ConnectionOperationTarget
    ConnectionOperationTarget.model_validate(
        {"name": "brag", "kind": "skill", "action": "install", "state": "connected", **row, "detail": detail})
