"""Hermetic tests for the seerr skill.

Stdlib + pytest + unittest.mock only; no live network. Every HTTP call is
intercepted at urllib.request.urlopen and asserted on.

    scripts/run_tests.sh tests/skills/test_seerr_skill.py -q
"""

from __future__ import annotations

import io
import json
import re
import sys
import urllib.error
from pathlib import Path
from unittest import mock

import pytest

_HERE = Path(__file__).resolve()
SKILL_DIR = _HERE.parent.parent.parent / "skills" / "media" / "seerr"
SKILL_MD = SKILL_DIR / "SKILL.md"
sys.path.insert(0, str(SKILL_DIR / "scripts"))

import seerr  # noqa: E402


ENV = {"SEERR_URL": "http://seerr.test:5055", "SEERR_API_KEY": "k3y"}


class _Response(io.BytesIO):
    """Minimal stand-in for the urlopen context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_urlopen(payload, captured: list):
    def _open(request, timeout=None):
        captured.append(request)
        return _Response(json.dumps(payload).encode("utf-8"))

    return _open


def _run(argv, payload=None, env=None):
    """Run the CLI with urlopen mocked. Returns (exit_code, stdout, requests)."""
    captured: list = []
    stdout = io.StringIO()
    with mock.patch.dict("os.environ", env if env is not None else ENV, clear=True), \
         mock.patch("urllib.request.urlopen", _fake_urlopen(payload or {}, captured)), \
         mock.patch("sys.stdout", stdout):
        code = seerr.main(argv)
    return code, stdout.getvalue(), captured


# ── Frontmatter: AGENTS.md skill authoring standards ──────────────────────


def _frontmatter() -> dict:
    text = SKILL_MD.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, "SKILL.md must open with a YAML frontmatter block"
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(match.group(1))


def test_description_meets_authoring_standard():
    description = _frontmatter()["description"]
    assert len(description) <= 60, f"description is {len(description)} chars, max 60"
    assert description.endswith("."), "description must end with a period"
    assert description.count(".") == 1, "description must be a single sentence"


def test_required_frontmatter_metadata_present():
    front = _frontmatter()
    for field in ("name", "version", "author", "license", "platforms"):
        assert front.get(field), f"SKILL.md frontmatter is missing `{field}`"
    assert front["name"] == "seerr"


def test_every_yaml_block_in_skill_md_is_valid_yaml():
    """Regression: the v1 skill documented dotenv KEY=value inside a ```yaml fence.

    A YAML parse failure in config.yaml makes Hermes drop *every* user override,
    so a copy-pasteable but invalid block is actively harmful.
    """
    yaml = pytest.importorskip("yaml")
    blocks = re.findall(r"```yaml\n(.*?)```", SKILL_MD.read_text(encoding="utf-8"), re.DOTALL)
    assert blocks, "expected at least one yaml block (the config.yaml example)"
    for block in blocks:
        yaml.safe_load(block)  # raises if the block is dotenv or otherwise invalid
        assert not re.search(r"^[A-Z_]+=", block, re.MULTILINE), (
            "dotenv KEY=value lines must not appear inside a ```yaml fence"
        )


# ── Query encoding ────────────────────────────────────────────────────────


def test_search_percent_encodes_reserved_characters():
    """Regression: replacing spaces alone leaves '&' live and rewrites the query."""
    _, _, requests = _run(["search", "Tom & Jerry"], {"results": []})
    url = requests[0].full_url

    assert "Tom%20%26%20Jerry" in url
    assert "+" not in url, "space must encode as %20, not '+'"
    query = url.split("?", 1)[1]
    assert query.count("&") == 1, "the only '&' may be the separator before page="


def test_encode_params_escapes_every_reserved_character():
    encoded = seerr._encode_params({"query": "a&b=c d/e?f#g"})
    assert encoded == "query=a%26b%3Dc%20d%2Fe%3Ff%23g"


# ── Requests ──────────────────────────────────────────────────────────────


def _body(request) -> dict:
    return json.loads(request.data.decode("utf-8"))


def test_movie_request_uses_rootFolder_not_rootFolderPath():
    """Seerr's field is `rootFolder`; an unknown key is silently ignored."""
    _, _, requests = _run(
        ["request", "--type", "movie", "--tmdb-id", "27205", "--root-folder", "/movies"],
        {"id": 7, "media": {"status": 2}},
    )
    body = _body(requests[0])

    assert body["rootFolder"] == "/movies"
    assert "rootFolderPath" not in body
    assert body["mediaType"] == "movie"
    assert body["mediaId"] == 27205
    assert requests[0].method == "POST"


def test_tv_request_parses_seasons_and_accepts_all():
    _, _, requests = _run(
        ["request", "--type", "tv", "--tmdb-id", "136315", "--seasons", "1,3"],
        {"id": 8, "media": {}},
    )
    assert _body(requests[0])["seasons"] == [1, 3]

    _, _, requests = _run(
        ["request", "--type", "tv", "--tmdb-id", "136315", "--seasons", "all"],
        {"id": 9, "media": {}},
    )
    assert _body(requests[0])["seasons"] == "all"


def test_tv_request_without_seasons_is_rejected():
    code, _, requests = _run(["request", "--type", "tv", "--tmdb-id", "1"], {})
    assert code == 1
    assert not requests, "must fail before issuing an HTTP request"


def test_optional_request_fields_are_omitted_when_unset():
    _, _, requests = _run(["request", "--type", "movie", "--tmdb-id", "1"], {"id": 1, "media": {}})
    body = _body(requests[0])
    for field in ("rootFolder", "serverId", "profileId"):
        assert field not in body


# ── Auth, discovery, parsing ──────────────────────────────────────────────


def test_api_key_is_sent_as_header():
    _, _, requests = _run(["search", "dune"], {"results": []})
    assert requests[0].get_header("X-api-key") == "k3y"


def test_servers_reports_root_folders_from_seerr():
    """The supported root-folder discovery path: Seerr reads them from Radarr."""
    routes = {
        "/api/v1/service/radarr": [{"id": 0, "name": "Radarr", "isDefault": True}],
        "/api/v1/service/radarr/0": {
            "rootFolders": [{"id": 1, "path": "/data/movies"}],
            "profiles": [{"id": 4, "name": "HD-1080p"}],
        },
    }

    def _open(request, timeout=None):
        path = request.full_url.split("5055", 1)[1].split("?", 1)[0]
        return _Response(json.dumps(routes[path]).encode("utf-8"))

    stdout = io.StringIO()
    with mock.patch.dict("os.environ", ENV, clear=True), \
         mock.patch("urllib.request.urlopen", _open), \
         mock.patch("sys.stdout", stdout):
        code = seerr.main(["servers", "radarr"])

    out = stdout.getvalue()
    assert code == 0
    assert "server 0: Radarr (default)" in out
    assert "root folder: /data/movies" in out
    assert "profile 4: HD-1080p" in out


def test_seasons_excludes_specials():
    payload = {
        "name": "The Bear",
        "firstAirDate": "2022-06-23",
        "seasons": [
            {"seasonNumber": 0, "episodeCount": 3},
            {"seasonNumber": 1, "episodeCount": 8},
        ],
    }
    _, out, _ = _run(["seasons", "136315"], payload)
    assert "season 1" in out
    assert "season 0" not in out


def test_search_ignores_people_and_flags_existing_library_items():
    payload = {
        "results": [
            {"id": 1, "mediaType": "person", "name": "Nobody"},
            {"id": 2, "mediaType": "movie", "title": "Dune", "releaseDate": "2021-09-15",
             "mediaInfo": {"status": 5}},
        ]
    }
    _, out, _ = _run(["search", "dune"], payload)
    assert "Nobody" not in out
    assert "Dune (2021)" in out
    assert "[already in library]" in out


# ── Failure modes ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("missing", ["SEERR_URL", "SEERR_API_KEY"])
def test_missing_config_reports_a_clean_error(missing):
    env = {k: v for k, v in ENV.items() if k != missing}
    code, _, requests = _run(["search", "dune"], {}, env=env)
    assert code == 1
    assert not requests


def test_rejected_api_key_reports_a_clean_error():
    def _raise(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, io.BytesIO(b""))

    with mock.patch.dict("os.environ", ENV, clear=True), \
         mock.patch("urllib.request.urlopen", _raise):
        with pytest.raises(seerr.SeerrError, match="API key"):
            seerr.api("search", params={"query": "x"})
