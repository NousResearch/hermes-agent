"""Release-channel records: the updater follows the stable record exactly, or refuses.

A source install subscribed to ``stable`` resolves its target from the channel record at
``https://hermes-assets.nousresearch.com/releases/channels/stable.json`` and the build manifest
it names (digest-pinned). The edge serves those objects (a fake of the R2 bucket, behind the
TLS-inspecting proxy that is the namespace's only egress) and the git server behind the same
proxy serves the source. In every cell origin/main has moved past what stable pins, so "the
updater silently followed main" is visible as a HEAD that landed on the main tip.

Classes: the record is valid (land exactly on its commit), unavailable (404 / 503 / tunnel cut),
malformed (not JSON, manifest digest mismatch, another repository's record), or names a commit
the forge does not have. #124309 is the missing live record; this suite pins that the updater
then refuses stable with a clear message instead of moving to main.
"""

from __future__ import annotations

import json
import shutil

import pytest

from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I
from tests.e2e.core.upgrade.network import _netedge as N
from tests.e2e.core.upgrade.network import _seed as S

pytestmark = [
    pytest.mark.platforms("linux"),
    # Every `hermes update` here targets a throwaway sandboxed install, never the real checkout.
    pytest.mark.live_system_guard_bypass,
    pytest.mark.skipif(H.sandbox_required_reason() is not None, reason=str(H.sandbox_required_reason())),
    pytest.mark.skipif(N.netns_required_reason() is not None, reason=str(N.netns_required_reason())),
    pytest.mark.skipif(shutil.which("git") is None, reason="git required"),
    pytest.mark.skipif(I.real_uv() is None, reason="uv required"),
]

STABLE = "/releases/channels/stable.json"


@pytest.fixture(scope="module")
def inst(tmp_path_factory):
    inst = S.seed_install(tmp_path_factory.mktemp("net-channels"))
    r = inst.hermes("update", "--set-channel", "stable", edge=None, timeout=120)
    assert r.rc == 0, "could not subscribe the install to stable\n" + r.report(inst)
    return inst


def _release_and_main_tip(inst: S.Installed, tag: str) -> tuple[str, str]:
    """A stable candidate commit on top of the install, then a newer main tip on top of it."""
    release = inst.publish(f"{tag}-release")
    tip = I.publish_commit(inst.origin, inst.root, f"e2e network: {tag} main tip",
                           {f"E2E_NETWORK_{tag}_TIP.txt": "unreleased\n"})
    return release, tip


def _update(inst: S.Installed, assets: N.App | None = None, **edge_kw) -> S.Result:
    edge = inst.edge(assets=assets, **edge_kw)
    try:
        return inst.hermes("update", "--yes", edge=edge, timeout=600)
    finally:
        edge.close()


def test_valid_stable_record_lands_exactly_on_its_commit(inst):
    release, tip = _release_and_main_tip(inst, "valid")
    r = _update(inst, N.static_app(S.stable_objects(release)))
    assert r.rc == 0, "update to a valid stable record failed\n" + r.report(inst)
    head = inst.head()
    assert head != tip, "stable subscriber was moved to the unreleased main tip\n" + r.report(inst)
    assert head == release, f"stable update landed on {head}, not the released {release}\n" + r.report(inst)
    reads = [h.path for h in r.edge.proxy.requests(S.ASSETS)] if r.edge else []
    assert STABLE in reads, f"the stable record was not read through the proxy: {reads}\n" + r.report(inst)


def _refused(inst: S.Installed, r: S.Result, before: dict, what: str) -> None:
    S.assert_nothing_changed(inst, before, r, what)
    assert "stable" in r.out and "No update was applied" in r.out, (
        f"{what}: the refusal does not name the stable channel and say nothing was applied\n" + r.report(inst))


UNAVAILABLE = {
    "not-published-404": dict(assets=N.static_app({})),
    "outage-503": dict(assets=N.static_app({}, always={STABLE: N.Response(503, b"down\n")})),
    "tunnel-cut": dict(eof_hosts=[S.ASSETS]),
}


@pytest.mark.parametrize("fault", sorted(UNAVAILABLE))
def test_unavailable_stable_record_refuses_and_never_moves_to_main(inst, fault):
    _release_and_main_tip(inst, f"unavailable-{fault}")
    before = inst.state()
    r = _update(inst, **UNAVAILABLE[fault])
    _refused(inst, r, before, f"stable record {fault}")


def _malformed(release: str) -> dict[str, dict[str, bytes]]:
    good = S.stable_objects(release)
    manifest_key = next(k for k in good if k != STABLE)
    tampered = dict(good)
    manifest = json.loads(good[manifest_key])
    manifest["request"]["commit"] = "f" * 40  # a different build than the record's digest pins
    tampered[manifest_key] = S.canonical(manifest)
    return {
        "not-json": {STABLE: b"<html>502 Bad Gateway</html>\n"},
        "manifest-digest-mismatch": tampered,
        "other-repository": S.stable_objects(release, repository="someone-else/hermes-agent"),
    }


@pytest.mark.parametrize("fault", ["manifest-digest-mismatch", "not-json", "other-repository"])
def test_malformed_stable_record_refuses(inst, fault):
    release, _ = _release_and_main_tip(inst, f"malformed-{fault}")
    before = inst.state()
    r = _update(inst, N.static_app(_malformed(release)[fault]))
    _refused(inst, r, before, f"stable record {fault}")


def test_stable_record_naming_a_commit_the_forge_does_not_have(inst):
    """The record was published for a build whose commit never reached the forge (or was
    force-pushed away): the fetch of that exact commit fails and nothing may change."""
    _release_and_main_tip(inst, "unpublished-commit")
    before = inst.state()
    r = _update(inst, N.static_app(S.stable_objects("0123456789abcdef0123456789abcdef01234567")))
    S.assert_nothing_changed(inst, before, r, "stable record naming an unpublished commit")
