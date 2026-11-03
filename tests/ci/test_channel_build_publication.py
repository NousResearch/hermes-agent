"""A channel claim's publication: Stable Release builds the claim, then advances its channel."""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import hermes_yaml
import pytest

from hermes_cli.release_channels import ChannelError, canonical_json
from scripts.releases import channel_claims
from tests.ci.desktop_release_roles import (
    DOWNLOADABLE_DISPATCHES, admitted, commit_summary, gate, native_builds, needs_of, stage_step,
)
from tests.ci.test_desktop_release_tag_admission import _workflow

CLAIM = "channel.20261028T120000Z-v1.2.3"
SHA = "a" * 40
REPO = Path(__file__).resolve().parents[2]


def _stable_release() -> dict:
    return hermes_yaml.safe_load((REPO / ".github/workflows/stable-release.yml").read_text(encoding="utf-8"))


class Store:
    """An R2 channel store: conditional writes, ETags, readback."""

    def __init__(self):
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.revision = 0

    def get(self, key):
        return self.objects.get(key)

    def put(self, key, body, etag=None):
        from scripts.releases.channels import ChannelConflict

        current = self.objects.get(key)
        if (etag is None and current is not None) or (etag is not None and (current is None or current[1] != etag)):
            raise ChannelConflict(key)
        self.revision += 1
        self.objects[key] = (body, f'"{self.revision}"')


class Publisher:
    def __init__(self, store):
        self.store, self.public_base, self.promoted = store, "https://archive.example", []
        self.reader = type("Reader", (), {"read_bytes": lambda _, key: store.objects[key][0]})()
        self.verify_build = None

    def _write(self, key, value, etag=None):
        self.store.put(key, canonical_json(value), etag)

    def promote(self, build_id):
        self.promoted.append(build_id)
        return {"head": build_id}


@pytest.fixture
def claim_publication(monkeypatch, tmp_path):
    """publish_claim with every collaborator replaced by a recorder of what it was handed."""
    from scripts.bundles import channel_artifacts
    from scripts.releases import channel_releases, handoff, r2, stable

    request = {"buildId": "b" * 32, "releaseTag": CLAIM, "windowsVersion": "26.1028.12.0", "channel": "preview"}
    store = Store()
    publisher = Publisher(store)
    feed = tmp_path / "darwin" / "stable-mac.yml"
    seen = {"put": []}
    monkeypatch.setattr(stable, "check_claim", lambda env: {"policy": env.get("POLICY", "channel"),
                                                           "claim_tag": CLAIM, "commit": SHA, "channel": "preview"})
    monkeypatch.setattr(channel_claims, "require_admitted_request", lambda tag, claim, env: request)

    def make_publisher(env, actions):
        seen["actions"] = actions
        return publisher

    monkeypatch.setattr(channel_claims, "_publisher", make_publisher)
    monkeypatch.setattr(stable, "read_admitted_candidate",
                        lambda tag, commit, base, digest, *, archive: {"tag": tag, "archive": archive, "digest": digest,
                                                                      "packages": []})
    monkeypatch.setattr(channel_releases, "native_legs", lambda accepted: ("darwin-arm64",))
    monkeypatch.setattr(handoff, "fetch", lambda tag, commit, legs, root, includes, *, public_base:
                        seen.update(fetch=(tag, commit, legs, includes)))
    windows = {"platform": "windows", "version": "26.1028.12.0", "executableVersion": "26.1028.12.0"}
    monkeypatch.setattr(channel_releases, "read_native_receipts", lambda *a, **k: {"packages": [windows], "files": {}})

    def assemble(req, native, root, *, artifact_prefix, targets, local_bytes):
        seen["assemble"] = (artifact_prefix, local_bytes)
        feed.parent.mkdir(parents=True, exist_ok=True)
        feed.write_bytes(b"feed")
        return {"request": req, "packages": []}, [feed]

    monkeypatch.setattr(channel_artifacts, "assemble", assemble)
    monkeypatch.setattr(channel_releases, "match_accepted_packages", lambda manifest, accepted: None)

    def put(*, tag, key, file, key_is_full, immutable):
        seen["put"].append(key)
        store.put(key, Path(file).read_bytes())

    monkeypatch.setattr(r2, "put", put)
    return {"request": request, "store": store, "publisher": publisher, "seen": seen, "windows": windows,
            "env": {"CANDIDATE_MANIFEST_SHA256": "c" * 64}, "root": tmp_path}


def test_publish_claim_promotes_the_accepted_candidate_from_its_receipts(claim_publication):
    run = claim_publication
    result = channel_claims.publish_claim(run["env"], run["root"])
    prefix = "releases/channel-builds/" + "b" * 32 + "/"
    assert result == {"head": "b" * 32} and run["publisher"].promoted == ["b" * 32]
    # Only the promotion is authorized: admission's create and allocate are not.
    assert run["seen"]["actions"] == frozenset({"promote"})
    # It fetches metadata and feed legs only, from the claim's own archive, and assembles from receipts.
    assert run["seen"]["fetch"] == (CLAIM, SHA, ["darwin-arm64"], ["metadata-*.json", "*-mac.yml"])
    assert run["seen"]["assemble"] == (f"releases/tag/{CLAIM}/", False)
    assert run["seen"]["put"] == [prefix + "darwin/stable-mac.yml"]
    assert json.loads(run["store"].objects[prefix + "build.json"][0])["request"] == run["request"]
    assert run["publisher"].verify_build(run["request"], {"request": run["request"], "packages": []}) is True
    # A rerun finds its immutable feeds and manifest identical and promotes again.
    channel_claims.publish_claim(run["env"], run["root"])
    assert run["seen"]["put"] == [prefix + "darwin/stable-mac.yml"] and len(run["publisher"].promoted) == 2


def test_publish_claim_refuses_other_bytes_other_versions_and_other_policies(claim_publication):
    run = claim_publication
    prefix = "releases/channel-builds/" + "b" * 32 + "/"
    run["store"].put(prefix + "build.json", b"someone else's manifest")
    with pytest.raises(ChannelError, match="different bytes"):
        channel_claims.publish_claim(run["env"], run["root"])
    run["windows"]["version"] = "26.1028.12.1"
    with pytest.raises(ChannelError, match="timestamp"):
        channel_claims.publish_claim(run["env"], run["root"])
    with pytest.raises(ChannelError, match="channel claims only"):
        channel_claims.publish_claim({**run["env"], "POLICY": "canary"}, run["root"])
    assert run["publisher"].promoted == []


def test_admission_probes_cas_and_public_readback_on_the_real_channel():
    from hermes_cli.release_channels import channel_key

    store = Store()
    publisher = Publisher(store)
    key = channel_key("preview")
    store.put(key, canonical_json({"nextSequence": 1}))
    before = (json.loads(store.objects[key][0]), store.objects[key][1])
    store.put(key, canonical_json({"nextSequence": 2}), before[1])
    publisher._read = lambda name: (json.loads(store.objects[channel_key(name)][0]), store.objects[channel_key(name)][1])
    visible = {"nextSequence": 2}
    publisher.reader.resolve = lambda name: type("Resolution", (), {"terminal": visible})()
    channel_claims.probe_cas(publisher, "preview", before, {"sequence": 1})
    # R2 refused the stale replay; the record is still the allocation's.
    assert json.loads(store.objects[key][0]) == {"nextSequence": 2}
    # A rerun that advanced nothing has nothing stale to replay.
    channel_claims.probe_cas(publisher, "preview", publisher._read("preview"), {"sequence": 1})
    visible["nextSequence"] = 1
    with pytest.raises(ChannelError, match="public channel record"):
        channel_claims.probe_cas(publisher, "preview", before, {"sequence": 1})
    store.put = lambda key, body, etag=None: None
    visible["nextSequence"] = 2
    with pytest.raises(ChannelError, match="stale conditional"):
        channel_claims.probe_cas(publisher, "preview", before, {"sequence": 1})


def test_stable_release_admits_allocates_and_publishes_channel_claims():
    jobs = _stable_release()["jobs"]
    admit = jobs["admit"]
    assert admit["environment"] == "release-signing"
    (step,) = [step for step in admit["steps"] if step.get("id") == "admit"]
    assert {"CLOUDFLARE_R2_SECRET_ACCESS_KEY", "CLOUDFLARE_R2_PUBLIC_URL"} <= step["env"].keys()
    assert {"channel-build", "channel-request-sha256"} <= admit["outputs"].keys()
    candidates = [job for name, job in jobs.items() if name.startswith("candidates-")]
    assert candidates and all(job["with"]["channel_build"] == "${{ needs.admit.outputs.channel-build }}"
                              for job in candidates)
    publish = jobs["publish-channel"]
    assert "fromJSON(needs.admit.outputs.jobs)['publish-channel']" in publish["if"]
    assert any(step.get("run", "").endswith("-m scripts.releases.channel_claims publish")
               for step in publish["steps"])
    assert "publish-channel" in jobs["publication"]["needs"]


def test_desktop_candidates_build_channel_claims_and_no_direct_channel_dispatch_remains():
    workflow = _workflow()
    triggers = workflow.get("on", workflow.get(True))  # YAML 1.1 reads a bare `on` key as true
    dispatch = triggers["workflow_dispatch"]["inputs"]
    assert not {"channel", "branding", "disposable_channel", "disposable_receivers", "disposable_run"} & dispatch.keys()
    assert {"channel_build", "channel_request_sha256"} <= triggers["workflow_call"]["inputs"].keys()
    text = (REPO / ".github/workflows/desktop-bundled-release.yml").read_text(encoding="utf-8")
    assert "handoff stage --channel-request" not in text and "R2_DISPOSABLE_RUN" not in text
    assert "publish-channel" not in workflow["jobs"]
    jobs = workflow["jobs"]
    channel = admitted(["validate"], channel=True)
    for name in native_builds(jobs).values():
        # A channel claim's candidate stages under its claim archive, like every candidate.
        step = stage_step(jobs[name])
        for dispatch, needs in (("channel", channel), ("tag", admitted(["validate"])),
                                ("candidate", admitted(["validate"])), ("commit", admitted(["validate"]))):
            assert gate(step.get("if", "true"), DOWNLOADABLE_DISPATCHES[dispatch], needs, job_if=False), (name, dispatch)
        assert "--channel-request" in "".join(s.get("run", "") for s in jobs[name]["steps"])


def test_commit_only_status_page_is_not_published_for_a_channel_claim():
    jobs = _workflow()["jobs"]
    summary = jobs[commit_summary(jobs)]

    def runs(inputs, *, channel=False):
        needs = admitted(needs_of(summary), channel=channel)
        needs["validate"]["outputs"]["all-jobs"] = "true"
        return gate(summary["if"], inputs, needs)

    assert runs(DOWNLOADABLE_DISPATCHES["commit"])
    assert not runs(DOWNLOADABLE_DISPATCHES["channel"], channel=True)


def test_pinned_xml_retains_legacy_automatic_policy(tmp_path):
    from scripts.bundles.release_artifacts import write_appinstaller
    for policy in ("automatic", "pinned"):
        out = tmp_path / f"{policy}.appinstaller"
        write_appinstaller(out, identity="Fixture.App", publisher="CN=Fixture", version="0.0.7.0",
                           self_uri="https://archive.example/build/stable.appinstaller",
                           artifact_uri="https://archive.example/build/app.msixbundle", update_policy=policy)
        xml = ET.parse(out).getroot()
        assert (xml.find("{*}UpdateSettings/{*}OnLaunch") is not None) == (policy == "automatic")
        assert xml.find("{*}UpdateSettings/{*}ForceUpdateFromAnyVersion") is None
