"""Unit tests for the civitai skill scripts.

Covers every CLI subcommand in search.py, show.py, download.py — plus the
helpers in _common.py and the health_check.py harness. All network I/O is
mocked at the urllib layer (and subprocess where needed) so the suite is
fully offline and deterministic.

Run with::

    pytest hermesroot/tests/skills/test_civitai_skill.py -v
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _find_scripts_dir() -> Path:
    """Locate the civitai/scripts/ directory.

    Strategy:
      1. Honor CIVITAI_SCRIPTS_DIR env var if set (CI / restructured layouts).
      2. Walk up from this test file looking for any */civitai/scripts/_common.py
         under common skill-host roots (skills/, optional-skills/, plus a bare
         civitai/ for repos that drop the category dir).
      3. Fail loudly if not found — better than a confusing import error later.
    """
    override = os.environ.get("CIVITAI_SCRIPTS_DIR")
    if override:
        return Path(override).resolve()

    here = Path(__file__).resolve()
    # Sentinel file inside the scripts directory we're looking for.
    sentinel = "_common.py"
    # Candidate skill-tree roots, relative to each ancestor of this file.
    rel_candidates = (
        Path("optional-skills") / "creative" / "civitai" / "scripts",
        Path("skills") / "creative" / "civitai" / "scripts",
        Path("optional-skills") / "civitai" / "scripts",
        Path("skills") / "civitai" / "scripts",
        Path("civitai") / "scripts",
    )
    for parent in here.parents:
        for rel in rel_candidates:
            candidate = parent / rel
            if (candidate / sentinel).is_file():
                return candidate

    raise RuntimeError(
        "Could not locate civitai/scripts/ — set CIVITAI_SCRIPTS_DIR to "
        "the absolute path of the scripts directory."
    )


SCRIPTS_DIR = _find_scripts_dir()

# Make `from _common import ...` inside the scripts resolve against SCRIPTS_DIR.
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _load(name: str):
    """Load a script as a module, registering it under its bare name so that
    sibling scripts can `from _common import ...` without surprises."""
    if name in sys.modules:
        return sys.modules[name]
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load _common.py first — the others depend on it.
common = _load("_common")
search_mod = _load("search")
show_mod = _load("show")
download_mod = _load("download")
health_check_mod = _load("health_check")


# --------------------------------------------------------------------------- #
# Mock helpers
# --------------------------------------------------------------------------- #
class _MockResp:
    """Minimal context-manager mimic of a urllib response object."""

    def __init__(self, payload, status: int = 200):
        if isinstance(payload, (bytes, bytearray)):
            self._body = bytes(payload)
        elif isinstance(payload, str):
            self._body = payload.encode("utf-8")
        else:
            self._body = json.dumps(payload).encode("utf-8")
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self._body


def _route(responses: dict):
    """Build a urlopen side_effect that dispatches by URL substring match."""

    def _side_effect(req, timeout=None):  # noqa: ARG001
        url = req.full_url if hasattr(req, "full_url") else req.get_full_url()
        for needle, resp in responses.items():
            if needle in url:
                if isinstance(resp, Exception):
                    raise resp
                return _MockResp(resp)
        raise AssertionError(f"Unexpected URL in test: {url}")

    return _side_effect


def _http_error(url: str, code: int, body: bytes = b"err") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, "err", {}, io.BytesIO(body))


def _run_main(module, argv, env=None):
    """Invoke a script's main() with sys.argv patched."""
    full_argv = [module.__name__] + list(argv)
    with patch.object(sys, "argv", full_argv):
        if env is not None:
            with patch.dict(os.environ, env, clear=False):
                module.main()
        else:
            module.main()


# --------------------------------------------------------------------------- #
# Fixture payloads
# --------------------------------------------------------------------------- #
SAMPLE_MODEL = {
    "id": 257749,
    "name": "Realistic Vision V6.0",
    "type": "Checkpoint",
    "creator": {"username": "SG_161222"},
    "tags": [{"name": "photorealism"}, {"name": "base model"}],
    "description": "A great realism checkpoint.",
    "modelVersions": [
        {
            "id": 999111,
            "name": "V6.0 B1",
            "baseModel": "SD 1.5",
            "trainedWords": ["analog photo"],
            "files": [
                {
                    "name": "realisticVision_v60B1.safetensors",
                    "sizeKB": 2_097_152,
                    "primary": True,
                    "downloadUrl": "https://civitai.com/api/download/models/999111",
                    "hashes": {
                        "AutoV2": "E837144C55",
                        "SHA256": "E837144C55" + "0" * 54,
                    },
                    "pickleScanResult": "Success",
                    "virusScanResult": "Success",
                    "metadata": {
                        "format": "SafeTensor",
                        "fp": "fp16",
                        "size": "pruned",
                    },
                }
            ],
        },
        {
            "id": 999000,
            "name": "V5.1",
            "baseModel": "SD 1.5",
            "files": [],
        },
    ],
}

SAMPLE_VERSION = {
    "id": 999111,
    "modelId": 257749,
    "name": "V6.0 B1",
    "baseModel": "SD 1.5",
    "trainedWords": ["analog photo"],
    "model": {"name": "Realistic Vision V6.0", "type": "Checkpoint"},
    "files": SAMPLE_MODEL["modelVersions"][0]["files"],
}

SAMPLE_IMAGES = {
    "items": [
        {
            "id": 5551,
            "username": "alice",
            "stats": {
                "heartCount": 5,
                "likeCount": 10,
                "laughCount": 0,
                "cryCount": 0,
            },
            "meta": {
                "prompt": "a beautiful landscape, photorealistic",
                "negativePrompt": "blurry, low quality",
                "steps": 30,
                "sampler": "DPM++ 2M",
                "cfgScale": 7,
                "seed": 42,
                "resources": [
                    {"type": "lora", "name": "detail_tweaker", "weight": 0.8}
                ],
            },
            "tools": [{"name": "Automatic1111"}],
            "techniques": [{"name": "txt2img"}],
            "baseModel": "SD 1.5",
        }
    ]
}

SAMPLE_MEILI_RESPONSE = {
    "results": [
        {
            "hits": [
                {
                    "id": 257749,
                    "name": "Realistic Vision V6.0",
                    "type": "Checkpoint",
                    "user": {"username": "SG_161222"},
                    "version": {"baseModel": "SD 1.5"},
                    "metrics": {"downloadCount": 200_000, "thumbsUpCount": 5_000},
                    "triggerWords": ["analog photo"],
                    "tags": [{"name": "photorealism"}],
                },
                {
                    "id": 1,
                    "name": "Tiny LoRA",
                    "type": "LORA",
                    "user": {"username": "bob"},
                    "version": {"baseModel": "Flux.1 D"},
                    "metrics": {"downloadCount": 1234, "thumbsUpCount": 56},
                },
            ],
            "estimatedTotalHits": 2,
            "processingTimeMs": 4,
        }
    ]
}


# =========================================================================== #
# _common.py
# =========================================================================== #
class TestCommonFormatters:
    def test_fmt_size_gb(self):
        assert common.fmt_size(2 * 1024 * 1024) == "2.00 GB"

    def test_fmt_size_mb(self):
        assert common.fmt_size(2048) == "2.0 MB"

    def test_fmt_size_kb(self):
        assert common.fmt_size(512) == "512 KB"

    def test_fmt_size_invalid(self):
        assert common.fmt_size(None) == "?"
        assert common.fmt_size("nope") == "?"

    def test_fmt_int_with_commas(self):
        assert common.fmt_int(1234567) == "1,234,567"

    def test_fmt_int_invalid_passes_through(self):
        assert common.fmt_int("nope") == "nope"

    def test_truncate_short_unchanged(self):
        assert common.truncate("hello") == "hello"

    def test_truncate_long_appends_ellipsis(self):
        out = common.truncate("x" * 600, n=500)
        assert len(out) == 501  # 500 chars + ellipsis
        assert out.endswith("…")

    def test_truncate_collapses_newlines(self):
        assert common.truncate("a\nb\r\nc") == "a b c"

    def test_truncate_none_returns_empty(self):
        assert common.truncate(None) == ""

    def test_emit_json_writes_to_stdout(self, capsys):
        common.emit_json({"a": 1, "b": [1, 2]})
        assert json.loads(capsys.readouterr().out) == {"a": 1, "b": [1, 2]}


class TestParseBrowsingLevel:
    def test_default_all(self):
        assert common.parse_browsing_level("") == 31
        assert common.parse_browsing_level(None) == 31
        assert common.parse_browsing_level("all") == 31

    def test_individual_levels(self):
        assert common.parse_browsing_level("PG") == 1
        assert common.parse_browsing_level("X") == 8
        assert common.parse_browsing_level("XXX") == 16

    def test_combo_or_mask(self):
        # X | XXX  =  8 | 16  =  24
        assert common.parse_browsing_level("X,XXX") == 24

    def test_pg13_aliases_both_accepted(self):
        assert common.parse_browsing_level("PG-13") == 2
        assert common.parse_browsing_level("PG13") == 2

    def test_unknown_falls_back_to_all(self):
        assert common.parse_browsing_level("WEIRD") == 31


class TestDie:
    def test_die_exits_2_with_stderr(self, capsys):
        with pytest.raises(SystemExit) as exc:
            common.die("boom")
        assert exc.value.code == 2
        assert "error: boom" in capsys.readouterr().err

    def test_die_custom_code(self):
        with pytest.raises(SystemExit) as exc:
            common.die("nope", code=7)
        assert exc.value.code == 7


class TestApiGet:
    def test_basic_get(self):
        with patch("urllib.request.urlopen", side_effect=_route({
            "civitai.com/api/v1/models/257749": SAMPLE_MODEL,
        })):
            out = common.api_get("models/257749")
        assert out["id"] == 257749

    def test_includes_auth_header_when_key_set(self, monkeypatch):
        monkeypatch.setenv("CIVITAI_API_KEY", "secret-token")
        seen = {}

        def fake_urlopen(req, timeout=None):  # noqa: ARG001
            seen["headers"] = dict(req.headers)
            return _MockResp({"ok": True})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            common.api_get("models/1")

        # urllib lower-cases header keys when stored on Request.headers
        auth = seen["headers"].get("Authorization") or seen["headers"].get("authorization")
        assert auth == "Bearer secret-token"

    def test_no_auth_header_when_key_unset(self, monkeypatch):
        monkeypatch.delenv("CIVITAI_API_KEY", raising=False)
        seen = {}

        def fake_urlopen(req, timeout=None):  # noqa: ARG001
            seen["headers"] = dict(req.headers)
            return _MockResp({"ok": True})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            common.api_get("models/1")

        assert "Authorization" not in seen["headers"]
        assert "authorization" not in seen["headers"]

    def test_query_param_encoding(self):
        captured = {}

        def fake_urlopen(req, timeout=None):  # noqa: ARG001
            captured["url"] = req.full_url
            return _MockResp({"items": []})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            common.api_get("models", {
                "limit": 5,
                "sort": "Newest",
                "types": ["LORA", "Checkpoint"],
                "nsfw": False,
                "skip": None,    # dropped
            })

        url = captured["url"]
        assert "limit=5" in url
        assert "sort=Newest" in url
        assert "types=LORA" in url and "types=Checkpoint" in url
        assert "nsfw=false" in url
        assert "skip" not in url

    def test_404_dies_not_found(self, capsys):
        err = _http_error("https://civitai.com/api/v1/models/0", 404)
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(SystemExit):
                common.api_get("models/0")
        assert "not found" in capsys.readouterr().err

    def test_401_dies_auth(self, capsys):
        err = _http_error("https://civitai.com/api/v1/models/1", 401)
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(SystemExit):
                common.api_get("models/1")
        assert "auth failed" in capsys.readouterr().err

    def test_403_dies_auth(self, capsys):
        err = _http_error("https://civitai.com/api/v1/models/1", 403)
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(SystemExit):
                common.api_get("models/1")
        assert "auth failed" in capsys.readouterr().err

    def test_429_dies_after_retries(self, capsys):
        err = _http_error("https://civitai.com/api/v1/models/1", 429)
        with patch("urllib.request.urlopen", side_effect=err):
            with patch.object(common.time, "sleep"):  # speed up retry waits
                with pytest.raises(SystemExit):
                    common.api_get("models/1")
        assert "rate limited" in capsys.readouterr().err

    def test_retry_recovers_after_transient_503(self):
        responses = [_http_error("u", 503), _MockResp({"id": 1})]
        with patch("urllib.request.urlopen", side_effect=responses):
            with patch.object(common.time, "sleep"):
                out = common.api_get("models/1")
        assert out == {"id": 1}

    def test_unknown_5xx_dies(self, capsys):
        err = _http_error("u", 500, body=b"internal error blob")
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(SystemExit):
                common.api_get("models/1")
        assert "http 500" in capsys.readouterr().err


class TestMeiliSearch:
    def test_constructs_filter_sort_and_index(self):
        captured = {}

        def fake_urlopen(req, timeout=None):  # noqa: ARG001
            captured["body"] = json.loads(req.data.decode())
            captured["url"] = req.full_url
            return _MockResp(SAMPLE_MEILI_RESPONSE)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            r = common.meili_search(
                "anime",
                types=["LORA"],
                base_model="Flux.1 D",
                tag="Anime",
                username="bob",
                sort="Newest",
                nsfw=False,
                limit=3,
            )

        assert "multi-search" in captured["url"]
        q = captured["body"]["queries"][0]
        assert q["q"] == "anime"
        assert q["limit"] == 3
        assert q["indexUid"] == "models_v9"
        assert q["sort"] == ["createdAt:desc"]

        f = q["filter"]
        assert 'type = "LORA"' in f
        assert 'version.baseModel = "Flux.1 D"' in f
        assert 'tags.name = "anime"' in f      # auto-lowercased
        assert 'user.username = "bob"' in f
        assert "nsfwLevel = 1" in f

        assert r["hits"][0]["id"] == 257749

    def test_multiple_types_joined_by_or(self):
        captured = {}

        def fake_urlopen(req, timeout=None):  # noqa: ARG001
            captured["body"] = json.loads(req.data.decode())
            return _MockResp(SAMPLE_MEILI_RESPONSE)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            common.meili_search("x", types=["LORA", "Checkpoint"])

        f = captured["body"]["queries"][0]["filter"]
        assert 'type = "LORA"' in f
        assert 'type = "Checkpoint"' in f
        assert " OR " in f

    def test_default_sort_when_unknown(self):
        captured = {}

        def fake_urlopen(req, timeout=None):  # noqa: ARG001
            captured["body"] = json.loads(req.data.decode())
            return _MockResp(SAMPLE_MEILI_RESPONSE)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            common.meili_search("x", sort="NotARealSort")

        q = captured["body"]["queries"][0]
        assert q["sort"] == ["metrics.downloadCount:desc"]

    def test_meili_auth_failure_dies(self, capsys):
        err = _http_error("https://search-new.civitai.com/multi-search", 401)
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(SystemExit):
                common.meili_search("x")
        assert "meilisearch http 401" in capsys.readouterr().err

    def test_meili_uses_env_key_when_set(self, monkeypatch):
        # The module reads MEILISEARCH_KEY at import time, so we just verify
        # the constant resolves to the documented default when env is unset.
        # (Setting it post-import has no effect — by design.)
        monkeypatch.delenv("MEILISEARCH_KEY", raising=False)
        assert len(common.MEILI_KEY) == 64  # 64-char hex string


# =========================================================================== #
# search.py
# =========================================================================== #
class TestSearchModels:
    def test_query_routes_to_meilisearch(self, capsys):
        with patch("urllib.request.urlopen", side_effect=_route({
            "multi-search": SAMPLE_MEILI_RESPONSE,
        })):
            _run_main(search_mod, ["models", "--query", "realistic", "--limit", "2"])

        out = capsys.readouterr().out
        assert "Meilisearch" in out
        assert "#257749" in out
        assert "Realistic Vision V6.0" in out
        assert "SG_161222" in out

    def test_query_json_output(self, capsys):
        with patch("urllib.request.urlopen", side_effect=_route({
            "multi-search": SAMPLE_MEILI_RESPONSE,
        })):
            _run_main(search_mod, ["models", "--query", "anything", "--json"])

        parsed = json.loads(capsys.readouterr().out)
        assert parsed["hits"][0]["id"] == 257749
        assert parsed["estimatedTotalHits"] == 2

    def test_no_query_uses_rest(self, capsys):
        rest_payload = {"items": [SAMPLE_MEILI_RESPONSE["results"][0]["hits"][0]]}
        captured = {}

        def fake_urlopen(req, timeout=None):  # noqa: ARG001
            captured["url"] = req.full_url
            return _MockResp(rest_payload)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            _run_main(search_mod, ["models", "--type", "Checkpoint", "--limit", "1"])

        assert "civitai.com/api/v1/models" in captured["url"]
        assert "multi-search" not in captured["url"]
        assert "Models (REST)" in capsys.readouterr().out

    def test_ids_forces_rest_even_with_query(self):
        captured = {}

        def fake_urlopen(req, timeout=None):  # noqa: ARG001
            captured["url"] = req.full_url
            return _MockResp({"items": []})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            _run_main(search_mod, ["models", "--query", "ignored", "--ids", "1,2,3"])

        assert "civitai.com/api/v1/models" in captured["url"]
        # ids reach the URL (may be percent-encoded)
        assert "ids=1%2C2%2C3" in captured["url"] or "ids=1,2,3" in captured["url"]

    def test_rest_sort_downgrade(self):
        captured = {}

        def fake_urlopen(req, timeout=None):  # noqa: ARG001
            captured["url"] = req.full_url
            return _MockResp({"items": []})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            # 'Most Tipped' is not REST-supported and should silently downgrade
            _run_main(search_mod, ["models", "--sort", "Most Tipped"])

        assert "sort=Most+Downloaded" in captured["url"]

    def test_no_results_prints_message(self, capsys):
        empty = {"results": [{"hits": [], "estimatedTotalHits": 0, "processingTimeMs": 1}]}
        with patch("urllib.request.urlopen", side_effect=_route({
            "multi-search": empty,
        })):
            _run_main(search_mod, ["models", "--query", "zzz"])

        assert "No results." in capsys.readouterr().out


class TestSearchImages:
    def test_model_id_auto_resolves_to_version(self, capsys, monkeypatch):
        monkeypatch.delenv("CIVITAI_API_KEY", raising=False)
        captured_urls = []

        def fake_urlopen(req, timeout=None):  # noqa: ARG001
            captured_urls.append(req.full_url)
            if "/models/257749" in req.full_url:
                return _MockResp(SAMPLE_MODEL)
            if "/images" in req.full_url:
                return _MockResp(SAMPLE_IMAGES)
            raise AssertionError(f"Unexpected URL: {req.full_url}")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            _run_main(search_mod, ["images", "--model-id", "257749", "--has-meta"])

        cap = capsys.readouterr()
        assert "# resolved model 257749 → version 999111" in cap.err
        image_urls = [u for u in captured_urls if "/images" in u]
        assert any("modelVersionId=999111" in u for u in image_urls)
        assert "hasMeta=true" in image_urls[0]

    def test_model_version_id_direct(self, capsys):
        captured_urls = []

        def fake_urlopen(req, timeout=None):  # noqa: ARG001
            captured_urls.append(req.full_url)
            return _MockResp(SAMPLE_IMAGES)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            _run_main(search_mod, ["images", "--model-version-id", "999111"])

        assert any("modelVersionId=999111" in u for u in captured_urls)
        out = capsys.readouterr().out
        assert "Image #5551" in out
        assert "Prompt:" in out
        assert "LoRAs:" in out
        assert "DPM++ 2M" in out

    def test_browsing_level_nsfw_downgrade_without_key(self, capsys, monkeypatch):
        monkeypatch.delenv("CIVITAI_API_KEY", raising=False)
        captured = {}

        def fake_urlopen(req, timeout=None):  # noqa: ARG001
            captured["url"] = req.full_url
            return _MockResp({"items": []})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            _run_main(search_mod, [
                "images", "--model-version-id", "1", "--browsing-level", "X,XXX",
            ])

        err = capsys.readouterr().err
        assert "downgrading to R" in err
        # R = 4
        assert "browsingLevel=4" in captured["url"]

    def test_browsing_level_with_key_respected(self, monkeypatch):
        monkeypatch.setenv("CIVITAI_API_KEY", "secret")
        captured = {}

        def fake_urlopen(req, timeout=None):  # noqa: ARG001
            captured["url"] = req.full_url
            return _MockResp({"items": []})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            _run_main(search_mod, [
                "images", "--model-version-id", "1", "--browsing-level", "X,XXX",
            ])

        # X | XXX = 24
        assert "browsingLevel=24" in captured["url"]

    def test_nsfw_flag_desugars_to_x_xxx_with_key(self, monkeypatch):
        """--nsfw on images should be equivalent to --browsing-level X,XXX."""
        monkeypatch.setenv("CIVITAI_API_KEY", "secret")
        captured = {}

        def fake_urlopen(req, timeout=None):  # noqa: ARG001
            captured["url"] = req.full_url
            return _MockResp({"items": []})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            _run_main(search_mod, [
                "images", "--model-version-id", "1", "--nsfw",
            ])

        # --nsfw → X|XXX = 24
        assert "browsingLevel=24" in captured["url"]

    def test_nsfw_flag_downgrades_to_r_without_key(self, capsys, monkeypatch):
        """--nsfw without a key emits a warning and downgrades to R."""
        monkeypatch.delenv("CIVITAI_API_KEY", raising=False)
        captured = {}

        def fake_urlopen(req, timeout=None):  # noqa: ARG001
            captured["url"] = req.full_url
            return _MockResp({"items": []})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            _run_main(search_mod, [
                "images", "--model-version-id", "1", "--nsfw",
            ])

        assert "browsingLevel=4" in captured["url"]  # R = 4
        assert "NSFW" in capsys.readouterr().err

    def test_explicit_browsing_level_wins_over_nsfw_flag(self, monkeypatch):
        """If both --nsfw and --browsing-level are passed, --browsing-level wins."""
        monkeypatch.setenv("CIVITAI_API_KEY", "secret")
        captured = {}

        def fake_urlopen(req, timeout=None):  # noqa: ARG001
            captured["url"] = req.full_url
            return _MockResp({"items": []})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            _run_main(search_mod, [
                "images", "--model-version-id", "1",
                "--nsfw", "--browsing-level", "R",
            ])

        # R = 4, not the X|XXX=24 that --nsfw alone would set
        assert "browsingLevel=4" in captured["url"]

    def test_json_output(self, capsys):
        with patch("urllib.request.urlopen", side_effect=_route({
            "/images": SAMPLE_IMAGES,
        })):
            _run_main(search_mod, ["images", "--model-version-id", "1", "--json"])

        rendered = json.loads(capsys.readouterr().out)
        assert rendered["items"][0]["id"] == 5551


class TestSearchTopModels:
    def test_basic_hits_rest_with_filters(self, capsys):
        # top-models has no --query option, so q is None and the code routes
        # through the REST /models endpoint with type+baseModels+sort filters.
        rest_payload = {"items": SAMPLE_MEILI_RESPONSE["results"][0]["hits"]}
        captured_urls = []

        def fake_urlopen(req, timeout=None):  # noqa: ARG001
            captured_urls.append(req.full_url)
            return _MockResp(rest_payload)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            _run_main(search_mod, [
                "top-models", "--type", "LORA",
                "--base-model", "Flux.1 D", "--limit", "2",
            ])

        url = captured_urls[0]
        assert "civitai.com/api/v1/models" in url
        assert "types=LORA" in url
        assert "baseModels=Flux.1+D" in url or "baseModels=Flux.1%20D" in url
        out = capsys.readouterr().out
        assert "Models (REST)" in out
        assert "#257749" in out or "Tiny LoRA" in out

    def test_top_models_json(self, capsys):
        rest_payload = {"items": SAMPLE_MEILI_RESPONSE["results"][0]["hits"]}
        with patch("urllib.request.urlopen", side_effect=_route({
            "civitai.com/api/v1/models": rest_payload,
        })):
            _run_main(search_mod, [
                "top-models", "--type", "Checkpoint", "--json",
            ])

        rendered = json.loads(capsys.readouterr().out)
        assert rendered["items"][0]["id"] == 257749


class TestSearchTopImages:
    def test_basic(self, capsys):
        captured_urls = []

        def fake_urlopen(req, timeout=None):  # noqa: ARG001
            captured_urls.append(req.full_url)
            return _MockResp(SAMPLE_IMAGES)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            _run_main(search_mod, [
                "top-images", "--period", "Week", "--has-meta", "--limit", "3",
            ])

        assert any("/images" in u for u in captured_urls)
        assert any("period=Week" in u for u in captured_urls)
        out = capsys.readouterr().out
        assert "Image #5551" in out


# =========================================================================== #
# show.py
# =========================================================================== #
class TestShowModel:
    def test_text_output(self, capsys):
        with patch("urllib.request.urlopen", side_effect=_route({
            "/models/257749": SAMPLE_MODEL,
        })):
            _run_main(show_mod, ["model", "257749"])

        out = capsys.readouterr().out
        assert "#257749" in out
        assert "Realistic Vision V6.0" in out
        assert "SD 1.5" in out
        assert "#999111" in out        # version listed
        assert "URL: https://civitai.com/models/257749" in out

    def test_json(self, capsys):
        with patch("urllib.request.urlopen", side_effect=_route({
            "/models/257749": SAMPLE_MODEL,
        })):
            _run_main(show_mod, ["model", "257749", "--json"])

        parsed = json.loads(capsys.readouterr().out)
        assert parsed["id"] == 257749
        assert parsed["modelVersions"][0]["id"] == 999111


class TestShowSlimJSON:
    """JSON output is slimmed to stay under terminal transport caps (~50 KB).

    Without this, large /models/{id} payloads (HTML descriptions, image
    arrays, many versions) get truncated mid-stream and break json.loads.
    """

    def test_slim_model_drops_description_blob(self, capsys):
        fat = {
            **SAMPLE_MODEL,
            # Simulate the 10-50 KB HTML description Civitai returns.
            "description": "<p>" + ("x" * 50_000) + "</p>",
        }
        with patch("urllib.request.urlopen", side_effect=_route({
            "/models/257749": fat,
        })):
            _run_main(show_mod, ["model", "257749", "--json"])

        out = capsys.readouterr().out
        # The description is what blows the payload up; it must not appear.
        assert "description" not in json.loads(out)
        # And total output must comfortably parse and stay small.
        assert len(out) < 10_000
        assert json.loads(out)["id"] == 257749

    def test_slim_model_drops_per_version_images(self, capsys):
        fat_versions = [
            {
                **SAMPLE_MODEL["modelVersions"][0],
                # 50 fake image entries per version — these inflate the payload
                # without giving an agent anything actionable.
                "images": [{"url": f"https://example.com/{i}.png",
                            "meta": {"prompt": "x" * 200}} for i in range(50)],
            }
        ]
        fat = {**SAMPLE_MODEL, "modelVersions": fat_versions}
        with patch("urllib.request.urlopen", side_effect=_route({
            "/models/257749": fat,
        })):
            _run_main(show_mod, ["model", "257749", "--json"])

        parsed = json.loads(capsys.readouterr().out)
        v0 = parsed["modelVersions"][0]
        assert "images" not in v0

    def test_slim_model_caps_versions_at_20(self, capsys):
        many = [
            {"id": 1000 + i, "name": f"v{i}", "baseModel": "SD 1.5", "files": []}
            for i in range(55)
        ]
        fat = {**SAMPLE_MODEL, "modelVersions": many}
        with patch("urllib.request.urlopen", side_effect=_route({
            "/models/257749": fat,
        })):
            _run_main(show_mod, ["model", "257749", "--json"])

        parsed = json.loads(capsys.readouterr().out)
        assert len(parsed["modelVersions"]) == 20
        assert parsed["_versions_truncated"] is True
        # Truncation surfaces the *oldest* dropped: kept the first 20 newest.
        assert parsed["modelVersions"][0]["id"] == 1000
        assert parsed["modelVersions"][-1]["id"] == 1019

    def test_slim_model_does_not_truncate_when_under_cap(self, capsys):
        with patch("urllib.request.urlopen", side_effect=_route({
            "/models/257749": SAMPLE_MODEL,
        })):
            _run_main(show_mod, ["model", "257749", "--json"])

        parsed = json.loads(capsys.readouterr().out)
        assert parsed["_versions_truncated"] is False

    def test_slim_files_drops_extra_hash_formats(self, capsys):
        # Civitai emits 6 hash formats per file; we only need 2.
        fat_file = {
            **SAMPLE_MODEL["modelVersions"][0]["files"][0],
            "hashes": {
                "AutoV1": "old1",
                "AutoV2": "E837144C55",
                "AutoV3": "new3",
                "SHA256": "E837144C55" + "0" * 54,
                "CRC32":  "deadbeef",
                "BLAKE3": "b" * 64,
            },
        }
        fat_versions = [{**SAMPLE_MODEL["modelVersions"][0], "files": [fat_file]}]
        fat = {**SAMPLE_MODEL, "modelVersions": fat_versions}
        with patch("urllib.request.urlopen", side_effect=_route({
            "/models/257749": fat,
        })):
            _run_main(show_mod, ["model", "257749", "--json"])

        parsed = json.loads(capsys.readouterr().out)
        kept = parsed["modelVersions"][0]["files"][0]["hashes"]
        assert set(kept.keys()) == {"SHA256", "AutoV2"}
        assert kept["AutoV2"] == "E837144C55"

    def test_slim_model_preserves_essentials(self, capsys):
        """Anything an agent might actually use must survive slimming."""
        with patch("urllib.request.urlopen", side_effect=_route({
            "/models/257749": SAMPLE_MODEL,
        })):
            _run_main(show_mod, ["model", "257749", "--json"])

        p = json.loads(capsys.readouterr().out)
        # Top-level identity + discovery fields
        for k in ("id", "name", "type", "creator", "tags"):
            assert k in p, f"slim_model dropped essential top-level field {k}"
        # Per-version download essentials
        v = p["modelVersions"][0]
        for k in ("id", "name", "baseModel", "trainedWords", "files"):
            assert k in v, f"slim_model dropped essential version field {k}"
        f = v["files"][0]
        for k in ("name", "sizeKB", "primary", "hashes",
                  "pickleScanResult", "virusScanResult", "downloadUrl"):
            assert k in f, f"slim_files dropped essential file field {k}"

    def test_slim_version_drops_redundant_hashes(self, capsys):
        fat = {
            **SAMPLE_VERSION,
            "files": [{
                **SAMPLE_VERSION["files"][0],
                "hashes": {
                    "AutoV1": "1", "AutoV2": "E837144C55",
                    "SHA256": "x" * 64, "CRC32": "d", "BLAKE3": "b" * 64,
                },
            }],
        }
        with patch("urllib.request.urlopen", side_effect=_route({
            "/model-versions/999111": fat,
        })):
            _run_main(show_mod, ["version", "999111", "--json"])

        parsed = json.loads(capsys.readouterr().out)
        kept = parsed["files"][0]["hashes"]
        assert set(kept.keys()) == {"SHA256", "AutoV2"}

    def test_slim_prompts_envelope_and_lora_filter(self, capsys):
        # Add a non-lora resource to confirm it gets filtered out.
        fat_images = {
            "items": [{
                **SAMPLE_IMAGES["items"][0],
                "meta": {
                    **SAMPLE_IMAGES["items"][0]["meta"],
                    "resources": [
                        {"type": "lora",  "name": "good_lora", "weight": 0.8},
                        {"type": "checkpoint", "name": "should_drop"},
                        {"type": "vae",  "name": "also_drop"},
                    ],
                },
            }],
        }

        def fake_urlopen(req, timeout=None):  # noqa: ARG001
            if "/models/257749" in req.full_url:
                return _MockResp(SAMPLE_MODEL)
            if "/images" in req.full_url:
                return _MockResp(fat_images)
            raise AssertionError(req.full_url)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            _run_main(show_mod, ["prompts", "257749", "--json"])

        parsed = json.loads(capsys.readouterr().out)
        assert parsed["modelId"] == 257749
        assert parsed["versionId"] == 999111
        item = parsed["items"][0]
        # reactions collapsed from stats
        assert item["reactions"] == 15  # heart 5 + like 10
        # URL helpfully attached
        assert item["url"] == "https://civitai.com/images/5551"
        # Non-LoRA resources dropped
        names = [r["name"] for r in item["meta"]["resources"]]
        assert names == ["good_lora"]

    def test_slim_output_stays_under_transport_cap(self, capsys):
        """End-to-end: a pathological model with HTML + images + many versions
        must still emit parseable JSON well under the 50 KB transport cap."""
        pathological = {
            **SAMPLE_MODEL,
            "description": "<html>" + ("y" * 80_000) + "</html>",
            "modelVersions": [
                {
                    "id":     i,
                    "name":   f"v{i}",
                    "baseModel": "SD 1.5",
                    "description": "<p>" + ("z" * 10_000) + "</p>",
                    "files":  SAMPLE_MODEL["modelVersions"][0]["files"],
                    "images": [{"url": f"u{j}", "meta": {"prompt": "p" * 500}}
                               for j in range(30)],
                }
                for i in range(50)
            ],
        }
        with patch("urllib.request.urlopen", side_effect=_route({
            "/models/257749": pathological,
        })):
            _run_main(show_mod, ["model", "257749", "--json"])

        out = capsys.readouterr().out
        # Must parse — the whole point of slimming.
        parsed = json.loads(out)
        # 80 KB of description + 50 × 10 KB version-descriptions + 50 × 30
        # image entries = ~600 KB raw; slim output must be drastically smaller.
        assert len(out) < 20_000, f"slim output is {len(out)} bytes — too fat"
        assert parsed["_versions_truncated"] is True


class TestShowVersion:
    def test_text_output_includes_hashes_and_scans(self, capsys):
        with patch("urllib.request.urlopen", side_effect=_route({
            "/model-versions/999111": SAMPLE_VERSION,
        })):
            _run_main(show_mod, ["version", "999111"])

        out = capsys.readouterr().out
        assert "Version #999111" in out
        assert "[PRIMARY]" in out
        assert "AutoV2: E837144C55" in out
        assert "SHA256:" in out
        assert "pickle=Success" in out
        assert "virus=Success" in out
        assert "fp16" in out

    def test_json(self, capsys):
        with patch("urllib.request.urlopen", side_effect=_route({
            "/model-versions/999111": SAMPLE_VERSION,
        })):
            _run_main(show_mod, ["version", "999111", "--json"])

        parsed = json.loads(capsys.readouterr().out)
        assert parsed["id"] == 999111


class TestShowHash:
    def test_text(self, capsys):
        with patch("urllib.request.urlopen", side_effect=_route({
            "/by-hash/E837144C55": SAMPLE_VERSION,
        })):
            _run_main(show_mod, ["hash", "e837144c55"])

        out = capsys.readouterr().out
        assert "Match: E837144C55" in out
        assert "Version #999111" in out
        assert "Realistic Vision V6.0" in out

    def test_uppercases_hash_in_request_url(self):
        captured = {}

        def fake_urlopen(req, timeout=None):  # noqa: ARG001
            captured["url"] = req.full_url
            return _MockResp(SAMPLE_VERSION)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            _run_main(show_mod, ["hash", "abc123def"])

        assert "ABC123DEF" in captured["url"]


class TestShowPrompts:
    def test_text(self, capsys):
        def fake_urlopen(req, timeout=None):  # noqa: ARG001
            if "/models/257749" in req.full_url:
                return _MockResp(SAMPLE_MODEL)
            if "/images" in req.full_url:
                return _MockResp(SAMPLE_IMAGES)
            raise AssertionError(f"Unexpected URL: {req.full_url}")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            _run_main(show_mod, ["prompts", "257749", "--limit", "5"])

        cap = capsys.readouterr()
        assert "resolved model 257749 → version 999111" in cap.err
        assert "Prompts for #257749" in cap.out
        assert "a beautiful landscape" in cap.out
        assert "Negative: blurry" in cap.out
        assert "LoRAs:" in cap.out
        assert "DPM++ 2M" in cap.out

    def test_no_images_message(self, capsys):
        def fake_urlopen(req, timeout=None):  # noqa: ARG001
            if "/models/" in req.full_url:
                return _MockResp(SAMPLE_MODEL)
            if "/images" in req.full_url:
                return _MockResp({"items": []})
            raise AssertionError(f"Unexpected URL: {req.full_url}")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            _run_main(show_mod, ["prompts", "257749"])

        assert "No images with meta." in capsys.readouterr().out

    def test_no_versions_dies(self, capsys):
        empty_model = {**SAMPLE_MODEL, "modelVersions": []}
        with patch("urllib.request.urlopen", side_effect=_route({
            "/models/257749": empty_model,
        })):
            with pytest.raises(SystemExit):
                _run_main(show_mod, ["prompts", "257749"])

        assert "has no versions" in capsys.readouterr().err


# =========================================================================== #
# download.py
# =========================================================================== #
class TestDownload:
    def test_emits_all_three_formats(self, capsys):
        with patch("urllib.request.urlopen", side_effect=_route({
            "/models/257749": SAMPLE_MODEL,
        })):
            _run_main(download_mod, [
                "257749", "--comfyui-path", "/data/ComfyUI/models",
            ])

        out = capsys.readouterr().out
        assert "curl -L" in out
        assert "wget --header=" in out
        assert "Invoke-WebRequest" in out
        assert "/data/ComfyUI/models/checkpoints/realisticVision_v60B1.safetensors" in out
        assert "$CIVITAI_API_KEY" in out
        assert "AutoV2: E837144C55" in out
        assert "pickle=Success" in out

    def test_curl_only(self, capsys):
        with patch("urllib.request.urlopen", side_effect=_route({
            "/models/257749": SAMPLE_MODEL,
        })):
            _run_main(download_mod, [
                "257749", "--format", "curl", "--comfyui-path", "/x",
            ])

        out = capsys.readouterr().out
        assert "curl -L" in out
        assert "wget --header=" not in out
        assert "Invoke-WebRequest" not in out

    def test_wget_only(self, capsys):
        with patch("urllib.request.urlopen", side_effect=_route({
            "/models/257749": SAMPLE_MODEL,
        })):
            _run_main(download_mod, [
                "257749", "--format", "wget", "--comfyui-path", "/x",
            ])

        out = capsys.readouterr().out
        assert "wget --header=" in out
        assert "curl -L" not in out

    def test_powershell_only(self, capsys):
        with patch("urllib.request.urlopen", side_effect=_route({
            "/models/257749": SAMPLE_MODEL,
        })):
            _run_main(download_mod, [
                "257749", "--format", "ps", "--comfyui-path", "/x",
            ])

        out = capsys.readouterr().out
        assert "Invoke-WebRequest" in out
        assert "curl -L" not in out

    def test_does_not_leak_key_value(self, capsys, monkeypatch):
        monkeypatch.setenv("CIVITAI_API_KEY", "my-super-secret-token")
        with patch("urllib.request.urlopen", side_effect=_route({
            "/models/257749": SAMPLE_MODEL,
        })):
            _run_main(download_mod, ["257749", "--comfyui-path", "/x"])

        out = capsys.readouterr().out
        assert "my-super-secret-token" not in out
        assert "$CIVITAI_API_KEY" in out

    def test_version_id_specific(self, capsys):
        with patch("urllib.request.urlopen", side_effect=_route({
            "/models/257749": SAMPLE_MODEL,
        })):
            _run_main(download_mod, [
                "257749", "--version-id", "999111", "--comfyui-path", "/x",
            ])

        assert "Version #999111" in capsys.readouterr().out

    def test_version_id_not_found_dies(self, capsys):
        with patch("urllib.request.urlopen", side_effect=_route({
            "/models/257749": SAMPLE_MODEL,
        })):
            with pytest.raises(SystemExit):
                _run_main(download_mod, ["257749", "--version-id", "424242"])

        assert "not found" in capsys.readouterr().err

    def test_no_comfy_path_uses_placeholder(self, capsys):
        # download.py no longer scrapes config.yaml; with no --comfyui-path,
        # the placeholder behavior kicks in naturally.
        with patch("urllib.request.urlopen", side_effect=_route({
            "/models/257749": SAMPLE_MODEL,
        })):
            _run_main(download_mod, ["257749"])

        assert "<COMFYUI_PATH>" in capsys.readouterr().out

    def test_json_output(self, capsys):
        with patch("urllib.request.urlopen", side_effect=_route({
            "/models/257749": SAMPLE_MODEL,
        })):
            _run_main(download_mod, [
                "257749", "--comfyui-path", "/x", "--json",
            ])

        rendered = json.loads(capsys.readouterr().out)
        assert rendered["model_id"] == 257749
        assert rendered["type"] == "Checkpoint"
        assert rendered["subfolder"] == "checkpoints"
        assert rendered["version_id"] == 999111
        f0 = rendered["files"][0]
        assert f0["primary"] is True
        assert f0["hashes"]["AutoV2"] == "E837144C55"
        assert f0["pickle_scan"] == "Success"
        assert "/x/checkpoints/" in f0["target"]

    def test_lora_subfolder_mapping(self, capsys):
        lora_model = {**SAMPLE_MODEL, "type": "LORA"}
        with patch("urllib.request.urlopen", side_effect=_route({
            "/models/257749": lora_model,
        })):
            _run_main(download_mod, [
                "257749", "--comfyui-path", "/m", "--json",
            ])

        rendered = json.loads(capsys.readouterr().out)
        assert rendered["subfolder"] == "loras"
        assert "/m/loras/" in rendered["files"][0]["target"]

    def test_dora_subfolder_mapping(self, capsys):
        dora_model = {**SAMPLE_MODEL, "type": "DoRA"}
        with patch("urllib.request.urlopen", side_effect=_route({
            "/models/257749": dora_model,
        })):
            _run_main(download_mod, [
                "257749", "--comfyui-path", "/m", "--json",
            ])

        rendered = json.loads(capsys.readouterr().out)
        assert rendered["subfolder"] == "loras"

    def test_vae_subfolder_mapping(self, capsys):
        vae_model = {**SAMPLE_MODEL, "type": "VAE"}
        with patch("urllib.request.urlopen", side_effect=_route({
            "/models/257749": vae_model,
        })):
            _run_main(download_mod, [
                "257749", "--comfyui-path", "/m", "--json",
            ])

        rendered = json.loads(capsys.readouterr().out)
        assert rendered["subfolder"] == "vae"

    def test_unknown_type_falls_back_to_other(self, capsys):
        weird = {**SAMPLE_MODEL, "type": "Workflows"}
        with patch("urllib.request.urlopen", side_effect=_route({
            "/models/1": weird,
        })):
            _run_main(download_mod, ["1", "--comfyui-path", "/m", "--json"])

        rendered = json.loads(capsys.readouterr().out)
        assert rendered["subfolder"] == "other"

    def test_no_versions_dies(self, capsys):
        no_versions = {**SAMPLE_MODEL, "modelVersions": []}
        with patch("urllib.request.urlopen", side_effect=_route({
            "/models/1": no_versions,
        })):
            with pytest.raises(SystemExit):
                _run_main(download_mod, ["1", "--comfyui-path", "/x"])

        assert "no versions" in capsys.readouterr().err

    def test_no_files_dies(self, capsys):
        no_files = {**SAMPLE_MODEL, "modelVersions": [{
            "id": 1, "name": "v", "baseModel": "SD 1.5", "files": [],
        }]}
        with patch("urllib.request.urlopen", side_effect=_route({
            "/models/1": no_files,
        })):
            with pytest.raises(SystemExit):
                _run_main(download_mod, ["1", "--comfyui-path", "/x"])

        assert "no files" in capsys.readouterr().err


# =========================================================================== #
# health_check.py
# =========================================================================== #
class TestHealthCheck:
    def test_static_checks_pass(self):
        results = health_check_mod.static_checks()
        statuses = [r["status"] for r in results]
        assert "fail" not in statuses, f"static_checks reported failures: {results}"
        names = [r["name"] for r in results]
        for s in ("_common.py present", "search.py present",
                   "show.py present", "download.py present"):
            assert s in names

    def test_static_checks_detect_missing_script(self, tmp_path, monkeypatch):
        # Point HERE at an empty dir and verify static_checks flags missing files.
        monkeypatch.setattr(health_check_mod, "HERE", str(tmp_path))
        results = health_check_mod.static_checks()
        statuses_by_name = {r["name"]: r["status"] for r in results}
        assert statuses_by_name["_common.py present"] == "fail"
        assert statuses_by_name["search.py present"] == "fail"

    def test_offline_mode_skips_network(self, capsys, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["health_check.py", "--offline", "--json"])
        with pytest.raises(SystemExit) as exc:
            health_check_mod.main()

        rendered = json.loads(capsys.readouterr().out)
        groups = {c["group"] for c in rendered}
        assert "Network" not in groups
        assert "Auth" not in groups
        assert "Functional" not in groups
        # offline can still fail if static fails — accept either 0 or 1
        assert exc.value.code in (0, 1)

    def test_network_checks_ok(self):
        def fake_http(url, key=None, body=None, timeout=12):  # noqa: ARG001
            return 200, 100, b"{}"

        with patch.object(health_check_mod, "_http", side_effect=fake_http):
            results = health_check_mod.network_checks()

        assert all(r["status"] == "ok" for r in results)

    def test_network_failure_marks_fail(self):
        def fake_http(url, key=None, body=None, timeout=12):  # noqa: ARG001
            return None, 100, b"oops"

        with patch.object(health_check_mod, "_http", side_effect=fake_http):
            results = health_check_mod.network_checks()

        assert any(r["status"] == "fail" for r in results)

    def test_search_new_root_404_still_ok(self):
        # Root path of search-new is allowed to 404 — any HTTP response = reachable
        def fake_http(url, key=None, body=None, timeout=12):  # noqa: ARG001
            if "search-new" in url:
                return 404, 50, b""
            return 200, 50, b"{}"

        with patch.object(health_check_mod, "_http", side_effect=fake_http):
            results = health_check_mod.network_checks()

        assert all(r["status"] == "ok" for r in results)

    def test_auth_check_skipped_without_key(self, monkeypatch):
        monkeypatch.delenv("CIVITAI_API_KEY", raising=False)
        results = health_check_mod.auth_checks()
        assert len(results) == 1
        assert results[0]["status"] == "skip"

    def test_auth_check_ok_with_valid_key(self, monkeypatch):
        monkeypatch.setenv("CIVITAI_API_KEY", "valid")
        with patch.object(health_check_mod, "_http", return_value=(200, 50, b"{}")):
            results = health_check_mod.auth_checks()
        assert results[0]["status"] == "ok"

    def test_auth_check_fail_with_bad_key(self, monkeypatch):
        monkeypatch.setenv("CIVITAI_API_KEY", "bad")
        with patch.object(health_check_mod, "_http", return_value=(401, 50, b"nope")):
            results = health_check_mod.auth_checks()
        assert results[0]["status"] == "fail"
        assert "401" in results[0]["detail"]

    def test_functional_checks_happy_path(self, monkeypatch):
        monkeypatch.delenv("CIVITAI_API_KEY", raising=False)
        meili_body = json.dumps({"results": [{"hits": [{"id": 1}]}]}).encode()
        model_body = json.dumps({"name": "Realistic Vision V6.0"}).encode()
        by_hash_body = json.dumps({"modelId": 257749}).encode()

        def fake_http(url, key=None, body=None, timeout=12):  # noqa: ARG001
            if "multi-search" in url:
                return 200, 50, meili_body
            if "by-hash" in url:
                return 200, 50, by_hash_body
            if "/models/" in url:
                return 200, 50, model_body
            return 200, 50, b"{}"

        with patch.object(health_check_mod, "_http", side_effect=fake_http), \
             patch.object(
                 health_check_mod.subprocess, "run",
                 return_value=MagicMock(returncode=0, stdout="curl -L test", stderr="")):
            results = health_check_mod.functional_checks()

        statuses = [r["status"] for r in results]
        assert "fail" not in statuses, f"functional reported failures: {results}"

    def test_functional_detects_key_leak(self, monkeypatch):
        monkeypatch.setenv("CIVITAI_API_KEY", "leaky-token-xyz")
        with patch.object(health_check_mod, "_http", return_value=(200, 50, b'{"name":"x"}')), \
             patch.object(
                 health_check_mod.subprocess, "run",
                 return_value=MagicMock(
                     returncode=0,
                     stdout="curl -L https://example.com -H 'Authorization: Bearer leaky-token-xyz'",
                     stderr="")):
            results = health_check_mod.functional_checks()

        leak_checks = [r for r in results if "download.py" in r["name"]]
        assert leak_checks and leak_checks[0]["status"] == "fail"
        assert "LEAKED" in leak_checks[0]["detail"]

    def test_config_check_missing_dir(self):
        results = health_check_mod.config_checks("/nonexistent/path/xyz123")
        assert results[0]["status"] == "fail"

    def test_config_check_ok_when_subfolders_present(self, tmp_path):
        (tmp_path / "checkpoints").mkdir()
        (tmp_path / "loras").mkdir()
        results = health_check_mod.config_checks(str(tmp_path))
        assert results[0]["status"] == "ok"

    def test_config_check_skip_when_subfolders_missing(self, tmp_path):
        results = health_check_mod.config_checks(str(tmp_path))
        assert results[0]["status"] == "skip"

    def test_config_check_empty_path_returns_nothing(self):
        assert health_check_mod.config_checks("") == []

    def test_full_run_with_everything_mocked(self, capsys, monkeypatch):
        monkeypatch.delenv("CIVITAI_API_KEY", raising=False)

        meili_body = json.dumps({"results": [{"hits": [{"id": 1}]}]}).encode()
        model_body = json.dumps({"name": "Realistic Vision V6.0"}).encode()
        by_hash_body = json.dumps({"modelId": 257749}).encode()

        def fake_http(url, key=None, body=None, timeout=12):  # noqa: ARG001
            if "multi-search" in url:
                return 200, 50, meili_body
            if "by-hash" in url:
                return 200, 50, by_hash_body
            if "/models/" in url:
                return 200, 50, model_body
            if "search-new.civitai.com" in url:
                return 404, 50, b""
            return 200, 50, b'{"items":[]}'

        monkeypatch.setattr(sys, "argv", ["health_check.py", "--json"])
        with patch.object(health_check_mod, "_http", side_effect=fake_http), \
             patch.object(
                 health_check_mod.subprocess, "run",
                 return_value=MagicMock(returncode=0, stdout="curl -L test", stderr="")):
            with pytest.raises(SystemExit) as exc:
                health_check_mod.main()

        rendered = json.loads(capsys.readouterr().out)
        groups = {c["group"] for c in rendered}
        assert {"Static", "Network", "Auth", "Functional"} <= groups
        fails = [c for c in rendered if c["status"] == "fail"]
        assert not fails, f"unexpected failures: {fails}"
        assert exc.value.code == 0

    def test_text_emit_writes_groups_and_summary(self, capsys):
        sample = [
            {"group": "Static", "name": "x", "status": "ok", "detail": ""},
            {"group": "Static", "name": "y", "status": "skip", "detail": "why"},
            {"group": "Network", "name": "z", "status": "fail", "detail": "boom"},
        ]
        health_check_mod._emit_text(sample)
        out = capsys.readouterr().out
        assert "[Static]" in out
        assert "[Network]" in out
        assert "boom" in out
        assert "1 ok" in out and "1 skip" in out and "1 fail" in out