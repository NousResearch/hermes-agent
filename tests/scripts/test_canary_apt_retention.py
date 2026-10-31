"""Termux's live APT head outlives canaries that select only desktop lanes."""
from datetime import UTC, datetime
import json

import pytest

from scripts.releases import lanes, r2
from scripts.termux import stage_apt_repo
from tests.scripts.test_release_r2 import r2_server
from tests.termux_fixtures import build_deb


ROOT = "releases/termux/canary/"
OLD = "canary.20261001T064100Z-v1.2.3"
NEW = "canary.20261020T064100Z-v1.2.3"
NOW = datetime(2026, 10, 28, tzinfo=UTC).timestamp()


def test_prune_keeps_actual_apt_pool_after_desktop_only_canaries(request, tmp_path, capsys):
    server = request.getfixturevalue("r2_server")
    pool, staged = tmp_path / "pool", tmp_path / "apt"
    pool.mkdir()
    filename = "hermes-agent_1.2.3~canary.20261001064100-1_aarch64.deb"
    build_deb(pool / filename, {"Package": "hermes-agent", "Version": "1.2.3~canary.20261001064100-1",
                               "Architecture": "aarch64"})
    assert stage_apt_repo.stage(pool, staged, "hermes-canary", None, OLD) == 3
    for path in staged.rglob("*"):
        if path.is_file():
            server.store[ROOT + path.relative_to(staged).as_posix()] = (path.read_bytes(), '"apt"')
    live_pool = {key for key in server.store if key.startswith(ROOT + "pool/")}
    assert live_pool == {f"{ROOT}pool/{OLD}/h/{filename}"}
    assert lanes.jobs_to_run(lanes.CANARY - {"termux"}, "canary")["publish-canary"]
    server.store["releases/canary/release-candidates.json"] = (json.dumps({"archive": NEW}).encode(), '"head"')
    current = f"releases/tag/{NEW}/desktop.zip"
    stale = {f"releases/tag/{OLD}/desktop.zip", f"{ROOT}pool/{OLD}/h/unreferenced.deb"}
    stable = "releases/termux/stable/pool/rc.1-v1.2.3/h/hermes-agent.deb"
    for key in {current, stable, *stale}:
        server.store[key] = (b"artifact", '"object"')
    before = dict(server.store)
    r2.prune(14, dry_run=True, now_epoch=NOW)
    deleted = {line.removeprefix("(dry-run) would delete r2:") for line in capsys.readouterr().out.splitlines()
               if line.startswith("(dry-run)")}
    assert deleted == stale
    assert server.store == before
    r2.prune(14, now_epoch=NOW)
    assert set(server.store) == set(before) - stale
    assert live_pool <= set(server.store)
    assert server.store[stable] == before[stable]


@pytest.mark.parametrize("text", [
    None, "", "<html>upstream error</html>", "Package: hermes-agent\nVersion: 1.2.3\n",
    "Package: hermes-agent\nFilename: ../stable/pool/h/hermes.deb\n",
    "Package: hermes-agent\nFilename: pool/h/hermes.deb\nFilename: pool/h/other.deb\n",
    "Package: hermes-agent\nFilename: pool/h/hermes.deb\n\nPackage: missing-filename\n",
    "Package: hermes-agent\nFilename: pool/h/hermes.deb\n malformed continuation\n",
])
def test_prune_refuses_unreadable_or_ambiguous_live_apt_indices(request, text):
    server = request.getfixturevalue("r2_server")
    index = ROOT + "dists/hermes-canary/main/binary-aarch64/Packages"
    # Packages.gz proves a live index exists even if its uncompressed peer cannot be read.
    server.store[index + ".gz"] = (b"compressed index marker", '"gzip"')
    if text is not None:
        server.store[index] = (text.encode(), '"index"')
    server.store[f"{ROOT}pool/{OLD}/h/hermes.deb"] = (b"live package", '"pool"')
    server.store[f"releases/tag/{OLD}/desktop.zip"] = (b"expired archive", '"archive"')
    before = dict(server.store)
    with pytest.raises((RuntimeError, r2.R2RequestError), match="APT.*refusing to prune|R2 GET.*Packages"):
        r2.prune(14, now_epoch=NOW)
    assert server.store == before
    assert not any(method == "DELETE" for method, _, _ in server.requests)


def test_missing_apt_indices_keep_the_pool_until_references_can_be_established(request):
    server = request.getfixturevalue("r2_server")
    pool = f"{ROOT}pool/{OLD}/h/hermes.deb"
    archive = f"releases/tag/{OLD}/desktop.zip"
    server.store.update({pool: (b"package", '"pool"'), archive: (b"expired archive", '"archive"')})
    r2.prune(14, now_epoch=NOW)
    assert pool in server.store
    assert archive not in server.store
