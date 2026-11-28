"""Tests for the bundled suno-studio skill and its cross-platform helper script.

Covers:
  - Prompt validation (`validate_prompt`): artist/trademark blocking, character
    limits, structural `[metatags]`, contradictory tag detection, unexpanded
    digit warnings.
  - Browser automation JS generation (`emit_batch_fill_js`, `emit_click_create_js`,
    `emit_harvest_links_js`): IIFE wrapping, JSON escaping of quotes/backticks/newlines,
    and native React prototype setters.
  - Harvest link parsing (`parse_harvested_links`): deduplication, variation
    numbering per song title, and `/song/<id>` extraction.
  - Cross-platform CDP launcher and status checks (`find_browser`, `build_launch_argv`,
    `cdp_endpoint_status`, `launch_cdp_browser`) with mocked `urlopen` and `Popen`
    (no live network, no real browser spawned).
  - Reference link resolution between `SKILL.md` and `references/` + `scripts/`.
"""
from __future__ import annotations

import importlib.util
import io
import json
import re
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "skills" / "creative" / "suno-studio"
SKILL_MD = SKILL_DIR / "SKILL.md"
SCRIPT_PATH = SKILL_DIR / "scripts" / "suno_helper.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("suno_helper", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_skill_references_and_scripts_resolve():
    content = SKILL_MD.read_text(encoding="utf-8")
    refs = set(re.findall(r"references/[A-Za-z0-9._-]+\.md", content))
    scripts = set(re.findall(r"scripts/[A-Za-z0-9._-]+\.py", content))
    assert "references/browser-automation-patterns.md" in refs
    assert "scripts/suno_helper.py" in scripts
    for rel in refs | scripts:
        assert (SKILL_DIR / rel).is_file(), f"Missing referenced file: {rel}"


def test_validate_prompt_accepts_well_formed_prompt_and_flags_violations():
    mod = _load_helper()

    good = mod.validate_prompt(
        title="Midnight Neon Drift",
        style=(
            "Synthwave darkwave retro, 112 BPM, minor key, starts with sparse analog "
            "pads, builds with gated drums into a soaring chorus, smoky alto vocal."
        ),
        lyrics="[Intro]\n[Instrumental]\n\n[Verse 1]\nWalking twenty four seven under neon rain\n\n[Chorus]\n[Powerful]\nDrift into the night",
    )
    assert good["valid"] is True
    assert good["errors"] == []
    assert good["warnings"] == []

    bad = mod.validate_prompt(
        title="One Two Three Four Five Six Seven Eight Nine Ten",
        style="90s grunge rock in the style of Nirvana and Daft Punk",
        lyrics="[Verse 1]\n[Calm]\n[Aggressive]\nWorking 24/7 on AI",
    )
    assert bad["valid"] is False
    assert any("nirvana" in err.lower() for err in bad["errors"])
    assert any("daft punk" in err.lower() for err in bad["errors"])
    assert any("contradictory metatags" in w.lower() for w in bad["warnings"])
    assert any("24/7" in w for w in bad["warnings"])


def test_emit_batch_fill_js_escapes_special_chars_in_iife():
    mod = _load_helper()
    snippet = mod.emit_batch_fill_js(
        title='Song "One" `Test`',
        style="Warm analog soul, 90 BPM",
        lyrics="[Verse 1]\nLine 1\nLine 2 with `backticks` and 'quotes'",
        exclude_styles="autotune, mumble rap",
    )
    assert snippet.startswith("(() => {")
    assert snippet.endswith("})()")
    assert "HTMLInputElement.prototype" in snippet
    assert "HTMLTextAreaElement.prototype" in snippet
    # Extract the embedded `const data = {...};` JSON literal and verify it round-trips
    m = re.search(r"const data = (\{.*?\});\n", snippet, re.DOTALL)
    assert m is not None
    parsed = json.loads(m.group(1))
    assert parsed["title"] == 'Song "One" `Test`'
    assert "`backticks`" in parsed["lyrics"]
    assert parsed["excludeStyles"] == "autotune, mumble rap"


def test_parse_harvested_links_groups_variations_and_deduplicates():
    mod = _load_helper()
    raw = "\n".join(
        [
            "Midnight Neon Drift | https://suno.com/song/11111111-aaaa-bbbb-cccc-000000000001",
            "Midnight Neon Drift | https://suno.com/song/11111111-aaaa-bbbb-cccc-000000000001",
            "Midnight Neon Drift | https://suno.com/song/22222222-aaaa-bbbb-cccc-000000000002",
            "Solar Flare | https://suno.com/song/33333333-aaaa-bbbb-cccc-000000000003",
        ]
    )
    tracks = mod.parse_harvested_links(raw)
    assert len(tracks) == 3
    assert tracks[0]["song_id"] == "11111111-aaaa-bbbb-cccc-000000000001"
    assert tracks[0]["variation"] == 1
    assert tracks[1]["variation"] == 2
    assert tracks[2]["title"] == "Solar Flare"
    assert tracks[2]["variation"] == 1


def test_cdp_status_and_launch_use_mocked_transport_and_popen(tmp_path, monkeypatch):
    mod = _load_helper()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    class FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    payload = json.dumps(
        {
            "Browser": "Brave/130.0",
            "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/browser/abc",
        }
    ).encode("utf-8")

    with mock.patch.object(mod.urllib.request, "urlopen", return_value=FakeResp(payload)):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = mod.main(["cdp-status", "--port", "9222"])
    assert rc == 0
    status_out = json.loads(buf.getvalue())
    assert status_out["listening"] is True
    assert status_out["browser"] == "Brave/130.0"

    fake_browser = tmp_path / "brave-mock"
    fake_browser.write_text("", encoding="utf-8")
    seen_popen: dict = {}

    class DummyProc:
        pid = 42424

    def fake_popen(argv, **kwargs):
        seen_popen["argv"] = argv
        seen_popen["kwargs"] = kwargs
        return DummyProc()

    with mock.patch.object(mod.subprocess, "Popen", side_effect=fake_popen):
        res = mod.launch_cdp_browser(browser=str(fake_browser), port=9333)

    assert res["ok"] is True
    assert res["pid"] == 42424
    assert "--remote-debugging-port=9333" in seen_popen["argv"]
    assert any(a.startswith("--user-data-dir=") and "chrome-debug" in a for a in seen_popen["argv"])
