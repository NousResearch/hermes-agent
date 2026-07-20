from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = ROOT / "skills" / "autonomous-ai-agents" / "ebbinghaus-memory" / "SKILL.md"
PLUGIN_MANIFEST = ROOT / "plugins" / "memory" / "ebbinghaus" / "plugin.yaml"
PLUGIN_INIT = ROOT / "plugins" / "memory" / "ebbinghaus" / "__init__.py"


def _frontmatter_and_body(path: Path) -> tuple[dict[str, object], str]:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    end = text.index("\n---\n", 4)
    raw = text[4:end]
    body = text[end + len("\n---\n"):]
    data: dict[str, object] = {}
    for line in raw.splitlines():
        if not line or line.startswith("  ") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"')
    return data, body


def _load_ebbinghaus_module():
    spec = importlib.util.spec_from_file_location(
        "plugins.memory.ebbinghaus", PLUGIN_INIT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ebbinghaus_memory_skill_links_to_plugin_and_tool():
    metadata, body = _frontmatter_and_body(SKILL_PATH)

    assert metadata["name"] == "ebbinghaus-memory"
    description = str(metadata["description"])
    assert description.endswith(".")
    assert len(description) <= 60

    text = SKILL_PATH.read_text(encoding="utf-8")
    assert "plugin: plugins/memory/ebbinghaus" in text
    assert "tools: [ebbinghaus_memory]" in text
    assert "memory.provider` is set to another provider" in body
    for action in ("remember", "recall", "rehearse", "decay", "sleep", "forget"):
        assert action in body


def test_ebbinghaus_plugin_manifest_points_at_skill():
    manifest = PLUGIN_MANIFEST.read_text(encoding="utf-8")

    assert "skills:" in manifest
    assert "skills/autonomous-ai-agents/ebbinghaus-memory/SKILL.md" in manifest


def test_ebbinghaus_register_exposes_provider_and_skill(monkeypatch):
    module = _load_ebbinghaus_module()

    class Collector:
        def __init__(self):
            self.provider = None
            self.skills = []

        def register_memory_provider(self, provider):
            self.provider = provider

        def register_skill(self, name, path, description=""):
            self.skills.append((name, Path(path), description))

    monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(ROOT / "skills"))
    collector = Collector()
    module.register(collector)

    assert collector.provider is not None
    assert collector.provider.name == "ebbinghaus"
    assert collector.skills == [
        (
            "ebbinghaus-memory",
            SKILL_PATH,
            "Use Ebbinghaus memory sleep, recall, dream, and decay.",
        )
    ]
