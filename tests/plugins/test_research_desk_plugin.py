from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "research-desk"


def load_plugin():
    package_name = "research_desk_test_plugin"
    for name in list(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        package_name,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


class Context:
    def __init__(self, home: Path, workspace: Path, openmanus_workspace: Path, *, profile: str = "default"):
        self.profile_name = profile
        self.home = home
        self.workspace = workspace
        self.openmanus_workspace = openmanus_workspace
        self.tools = []
        self.commands = []
        self.cli_commands = {}
        self.calls = []

        class Llm:
            def complete_structured(self, **kwargs):
                class Result:
                    parsed = {"findings": [{"claim": "A supported observation", "source_refs": ["source-01"], "confidence": "high", "caveat": ""}]}
                    text = json.dumps(parsed)
                    provider = "test"
                    model = "test-model"

                return Result()

        self.llm = Llm()

    def register_tool(self, **kwargs):
        self.tools.append(kwargs)

    def register_command(self, *args, **kwargs):
        self.commands.append((args, kwargs))

    def register_cli_command(self, name, **kwargs):
        self.cli_commands[name] = kwargs

    def dispatch_tool(self, name, args, **kwargs):
        self.calls.append((name, args))
        if name == "web_search":
            return json.dumps({"success": True, "data": {"web": [{"url": "https://example.com/report", "title": "Public report", "description": "Public source"}]}})
        if name == "web_extract":
            return json.dumps({"results": [{"url": args["urls"][0], "title": "Public report", "content": "Public evidence content."}]})
        if name == "openmanus_wide_research":
            return json.dumps({"ok": True, "status": "completed", "items": [{"status": "completed", "stdout": json.dumps({"findings": [{"claim": "A supported observation", "source_refs": ["source-01"], "confidence": "high", "caveat": ""}]})}]})
        raise AssertionError(f"unexpected tool: {name}")


def configure(module, monkeypatch, ctx: Context):
    monkeypatch.setattr(module.core, "get_hermes_home", lambda: ctx.home)
    monkeypatch.setattr(module.core, "_all_entries", lambda: {
        "research-desk": {"workspace_root": str(ctx.workspace), "allowed_domains": ["example.com"], "profile_name": ctx.profile_name},
        "openmanus": {"workspace_root": str(ctx.openmanus_workspace), "allow_llm_network": True},
    })
    monkeypatch.setattr(module.core, "_entry", lambda: {"workspace_root": str(ctx.workspace), "allowed_domains": ["example.com"], "profile_name": ctx.profile_name})
    monkeypatch.setattr(module.core, "_source_revision", lambda: "test-revision")


def test_register_exposes_tools_cli_and_requirement():
    module = load_plugin()
    ctx = Context(Path("C:/home"), Path("C:/workspace"), Path("C:/workspace"))
    module.register(ctx)
    assert {tool["name"] for tool in ctx.tools} == {
        "research_desk_status",
        "research_desk_plan",
        "research_desk_run",
        "research_desk_export",
    }
    assert "research-desk" in ctx.cli_commands
    manifest = (PLUGIN_DIR / "plugin.yaml").read_text(encoding="utf-8")
    assert "requires_plugins:" in manifest
    assert "  - openmanus" in manifest


def test_plan_is_dry_and_does_not_dispatch(tmp_path, monkeypatch):
    module = load_plugin()
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    home.mkdir()
    workspace.mkdir()
    ctx = Context(home, workspace, workspace)
    configure(module, monkeypatch, ctx)
    result = module.core.plan(ctx, {"topic": "public market", "source_domains": ["example.com"]})
    assert result["ok"] is True
    assert result["external_communication"] is False
    assert result["openmanus_started"] is False
    assert ctx.calls == []
    assert Path(result["plan_path"]).is_file()


def test_plan_rejects_profile_mismatch_and_bad_domain(tmp_path, monkeypatch):
    module = load_plugin()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ctx = Context(tmp_path / "home", workspace, workspace, profile="customer-a")
    configure(module, monkeypatch, ctx)
    bad_profile = module.core.plan(ctx, {"topic": "topic", "profile_name": "customer-b", "source_domains": ["example.com"]})
    assert bad_profile["ok"] is False
    assert "profile_name" in bad_profile["error"]
    bad_domain = module.core.plan(ctx, {"topic": "topic", "source_domains": ["not-allowlisted.example"]})
    assert bad_domain["ok"] is False


def test_url_policy_rejects_private_and_unallowlisted_urls():
    module = load_plugin()
    assert module.core._domain_allowed("https://example.com/a", ["example.com"])
    assert not module.core._domain_allowed("http://127.0.0.1/a", ["example.com"])
    assert not module.core._domain_allowed("https://not-example.com/a", ["example.com"])
    assert not module.core._domain_allowed("https://example.com:8443/a", ["example.com"])


def test_run_collects_evidence_runs_no_secret_worker_and_writes_receipt(tmp_path, monkeypatch):
    module = load_plugin()
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    home.mkdir()
    workspace.mkdir()
    ctx = Context(home, workspace, workspace)
    configure(module, monkeypatch, ctx)
    planned = module.core.plan(ctx, {"topic": "public market", "source_domains": ["example.com"], "worker_count": 1})
    result = module.core.run(ctx, {"plan_id": planned["plan_id"], "approved": True, "acknowledge_side_effects": True})
    assert result["ok"] is True
    assert any(name == "openmanus_wide_research" for name, _ in ctx.calls)
    worker_call = next(args for name, args in ctx.calls if name == "openmanus_wide_research")
    assert worker_call["allow_network"] is True
    assert worker_call["network_scope"] == "llm_only"
    assert worker_call["no_secret_env"] is True
    receipt = json.loads(Path(result["receipt_path"]).read_text(encoding="utf-8"))
    assert receipt["raw_customer_content_included"] is False
    assert receipt["openmanus_revision"] == "test-revision"
    assert "Public evidence content" not in json.dumps(receipt)
    report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
    assert report["findings"][0]["source_refs"] == ["source-01"]


def test_sensitive_text_is_redacted_before_worker_and_receipt():
    module = load_plugin()
    text = "Contact alice@example.com or +81 90-1234-5678 from C:/Users/Alice/Documents."
    redacted = module.core._redact_sensitive_text(text)
    assert "alice@example.com" not in redacted
    assert "90-1234-5678" not in redacted
    assert "C:/Users/Alice" not in redacted
    assert "[REDACTED_EMAIL]" in redacted


def test_export_requires_approval(tmp_path, monkeypatch):
    module = load_plugin()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ctx = Context(tmp_path / "home", workspace, workspace)
    configure(module, monkeypatch, ctx)
    assert module.core.export(ctx, {"run_id": "run-invalid", "format": "json", "approved": False})["ok"] is False
