"""Hermetic tests for the seerr skill.

Stdlib + pytest + unittest.mock only; no live network. Every HTTP call is
intercepted at urllib.request.urlopen and asserted on. HOME is redirected to a
temp dir so the script's ~/.hermes/.env fallback cannot read the real one.

    scripts/run_tests.sh tests/skills/test_seerr_skill.py -q
"""

from __future__ import annotations

import io
import json
import pathlib
import re
import sys
import urllib.error
import urllib.parse
from email.message import Message
from unittest import mock

import pytest

_HERE = pathlib.Path(__file__).resolve()
SKILL_DIR = _HERE.parent.parent.parent / "skills" / "media" / "seerr"
SKILL_MD = SKILL_DIR / "SKILL.md"
sys.path.insert(0, str(SKILL_DIR / "scripts"))

import seerr  # noqa: E402


ENV = {"SEERR_URL": "http://seerr.test:5055", "SEERR_API_KEY": "k3y"}
ARR_ENV = {"RADARR_URL": "http://radarr.test:7878", "RADARR_API_KEY": "r4d"}


@pytest.fixture(autouse=True)
def isolate_home(tmp_path, monkeypatch):
    """Never let the real ~/.hermes/.env satisfy a test that should fail."""
    monkeypatch.setattr(seerr.pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    seerr._ENV_LOADED = False
    yield tmp_path
    seerr._ENV_LOADED = False


class _Response:
    """Minimal stand-in for the urlopen context manager."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _run(argv, payload=None, env=None, routes=None):
    """Run the CLI with urlopen mocked. Returns (exit_code, stdout, requests)."""
    captured: list = []

    def _open(request, timeout=None):
        captured.append(request)
        if routes is not None:
            path = urllib.parse.urlsplit(request.full_url).path
            return _Response(json.dumps(routes[path]).encode("utf-8"))
        return _Response(json.dumps(payload if payload is not None else {}).encode("utf-8"))

    stdout = io.StringIO()
    with mock.patch.dict("os.environ", ENV if env is None else env, clear=True), \
         mock.patch("urllib.request.urlopen", _open), \
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


def test_only_seerr_credentials_are_mandatory():
    """Radarr/Sonarr are optional: a Seerr-only user must not be prompted for them."""
    names = {v["name"] for v in _frontmatter()["required_environment_variables"]}
    assert names == {"SEERR_URL", "SEERR_API_KEY"}


def test_every_yaml_block_in_skill_md_is_valid_yaml():
    """Regression: the v1 skill documented dotenv KEY=value inside a ```yaml fence.

    A YAML parse failure in config.yaml makes Hermes drop *every* user override,
    so a copy-pasteable but invalid block is actively harmful.
    """
    yaml = pytest.importorskip("yaml")
    blocks = re.findall(r"```yaml\n(.*?)```", SKILL_MD.read_text(encoding="utf-8"), re.DOTALL)
    assert blocks, "expected at least one yaml block (the config.yaml example)"
    for block in blocks:
        yaml.safe_load(block)
        assert not re.search(r"^[A-Z_]+=", block, re.MULTILINE), (
            "dotenv KEY=value lines must not appear inside a ```yaml fence"
        )


def test_skill_md_hardcodes_no_root_folder_paths():
    """The tables of real paths went stale twice; discovery replaced them."""
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "rootFolderPath" not in text.replace("`rootFolder`, not `rootFolderPath`", "")
    assert "/data/media/" not in text, "no install-specific paths in the skill"


# ── Query encoding ────────────────────────────────────────────────────────


def test_search_percent_encodes_reserved_characters():
    """Regression: replacing spaces alone leaves '&' live and rewrites the query."""
    _, _, requests = _run(["search", "Tom & Jerry"], {"results": []})
    url = requests[0].full_url

    assert "Tom%20%26%20Jerry" in url
    assert "+" not in url, "space must encode as %20, not '+'"
    assert url.split("?", 1)[1].count("&") == 1, "only the page= separator may be a bare '&'"


def test_encode_params_escapes_every_reserved_character():
    assert seerr._encode_params({"query": "a&b=c d/e?f#g"}) == "query=a%26b%3Dc%20d%2Fe%3Ff%23g"


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
        ["request", "--type", "tv", "--tmdb-id", "136315", "--seasons", "1,3"], {"id": 8, "media": {}}
    )
    assert _body(requests[0])["seasons"] == [1, 3]

    _, _, requests = _run(
        ["request", "--type", "tv", "--tmdb-id", "136315", "--seasons", "all"], {"id": 9, "media": {}}
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


# ── Discovery, auth, parsing ──────────────────────────────────────────────


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
    code, out, _ = _run(["servers", "radarr"], routes=routes)

    assert code == 0
    assert "server 0: Radarr (default)" in out
    assert "root folder: /data/movies" in out
    assert "profile 4: HD-1080p" in out


def test_seasons_excludes_specials():
    payload = {
        "name": "The Bear",
        "firstAirDate": "2022-06-23",
        "seasons": [{"seasonNumber": 0, "episodeCount": 3}, {"seasonNumber": 1, "episodeCount": 8}],
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


# ── Optional Radarr/Sonarr commands ───────────────────────────────────────


def test_diskspace_formats_mounts_and_hits_radarr():
    payload = [{"path": "/data", "freeSpace": 549_800_000_000, "totalSpace": 3_936_800_000_000}]
    _, out, requests = _run(["diskspace"], payload, env={**ENV, **ARR_ENV})

    assert "/api/v3/diskspace" in requests[0].full_url
    assert requests[0].get_header("X-api-key") == "r4d", "must use the Radarr key, not Seerr's"
    assert "549.8 GB free" in out


def test_optional_arr_commands_fail_cleanly_without_credentials():
    """A Seerr-only user must get a clear message, not a traceback."""
    code, _, requests = _run(["diskspace"], env=ENV)  # no RADARR_* set
    assert code == 1
    assert not requests


def test_queue_reports_progress():
    payload = {"records": [{"title": "Dune", "size": 100, "sizeleft": 25, "status": "downloading"}]}
    _, out, _ = _run(["queue"], payload, env={**ENV, **ARR_ENV})
    assert "75%" in out


# ── Configuration fallback ────────────────────────────────────────────────


def test_dotenv_fallback_supplies_vars_hermes_did_not_export(isolate_home):
    """On a real Hermes box SEERR_URL is absent from the shell; ~/.hermes/.env has it."""
    hermes = isolate_home / ".hermes"
    hermes.mkdir()
    (hermes / ".env").write_text(
        "# comment\nSEERR_URL=http://from-dotenv:5055\nexport SEERR_API_KEY='dotenv-key'\n",
        encoding="utf-8",
    )

    _, _, requests = _run(["search", "dune"], {"results": []}, env={})

    assert requests[0].full_url.startswith("http://from-dotenv:5055")
    assert requests[0].get_header("X-api-key") == "dotenv-key"


def test_real_environment_wins_over_dotenv(isolate_home):
    hermes = isolate_home / ".hermes"
    hermes.mkdir()
    (hermes / ".env").write_text("SEERR_URL=http://stale:5055\n", encoding="utf-8")

    _, _, requests = _run(["search", "dune"], {"results": []})  # ENV has seerr.test
    assert requests[0].full_url.startswith("http://seerr.test:5055")


@pytest.mark.parametrize("missing", ["SEERR_URL", "SEERR_API_KEY"])
def test_missing_config_reports_a_clean_error(missing):
    env = {k: v for k, v in ENV.items() if k != missing}
    code, _, requests = _run(["search", "dune"], {}, env=env)
    assert code == 1
    assert not requests


def test_rejected_api_key_reports_a_clean_error():
    def _raise(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", Message(), io.BytesIO(b""))

    with mock.patch.dict("os.environ", ENV, clear=True), \
         mock.patch("urllib.request.urlopen", _raise), \
         mock.patch("sys.stderr", io.StringIO()):
        assert seerr.main(["search", "dune"]) == 1


def test_already_requested_is_reported_as_such():
    def _raise(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 409, "Conflict", Message(), io.BytesIO(b""))

    stderr = io.StringIO()
    with mock.patch.dict("os.environ", ENV, clear=True), \
         mock.patch("urllib.request.urlopen", _raise), \
         mock.patch("sys.stderr", stderr):
        assert seerr.main(["request", "--type", "movie", "--tmdb-id", "1"]) == 1
    assert "Already requested" in stderr.getvalue()
