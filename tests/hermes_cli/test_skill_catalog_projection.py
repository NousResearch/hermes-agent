"""Live projection contract using actual OS leases and spawn processes."""
import multiprocessing
import json
import os
import time
import uuid
from contextlib import contextmanager, suppress
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from hermes_cli.skill_catalog.identity import Descriptor, source_id
from hermes_cli.skill_catalog.locks import FileLease, writer_hold
from hermes_cli.skill_catalog.validation import CatalogError


def projection(home):
    from hermes_cli.skill_catalog.projection import Projection
    return Projection(home)


def entry(home, key="owner"):
    class Owner(SimpleNamespace):
        pass
    return Owner(source_id=source_id(key, "team"), plugin_key=key,
        descriptor=Descriptor(1, "team", "Team", "impl-1"), scope=str(home),
        registration_id=uuid.uuid4().hex, generation=uuid.uuid4().hex,
        stopped=False, ready=False, authorized=lambda: True)


def record(owner, target="a/b", **changes):
    result = dict(identifier=owner.source_id + ":item", name="Example",
        target_relative_path=target, installed_version="1", latest_version="1",
        managed_state="installed", installation_revision="receipt-1", actions=["update", "uninstall"])
    result.update(changes)
    return result


def review(owner, target="new", **changes):
    result = dict(review_id=uuid.uuid4().hex, action="install", identifier=owner.source_id + ":item",
        version="1", installation_revision=None, target_relative_path=target, summary="",
        files=[], preview_complete=True,
        scan=dict(verdict="safe", policy="allow", summary="", findings=[], complete=True),
        acknowledgements_required=[], policy_revision="policy", content_digest="digest",
        setup_guidance=[], confirmable=True, review_token=uuid.uuid4().hex,
        expires_at_ms=9007199254740991)
    result.update(changes)
    return result


@contextmanager
def started(home, records=()):
    p, owner = projection(home), entry(home)
    handle = p.start(owner)
    try:
        p.publish(owner, list(records))
        yield p, owner
    finally:
        owner.stopped = True
        p.stop(owner)
        handle.close()


def test_transaction_rollback_validation_and_noop(tmp_path):
    p = projection(tmp_path)
    with p.transaction() as doc:
        assert doc == dict(schema_version=1, revision=0, registrations={}, targets={}, reservations={}, path_revisions={})
    assert not (p.directory / "claims.json").exists()
    owner = entry(tmp_path)
    handle = p.start(owner)
    before = (p.directory / "claims.json").read_bytes()
    try:
        with pytest.raises(RuntimeError):
            with p.transaction() as doc:
                doc["revision"] += 1
                raise RuntimeError()
        assert (p.directory / "claims.json").read_bytes() == before
        with pytest.raises(CatalogError) as caught:
            with p.transaction() as doc:
                doc["revision"] = True
        assert caught.value.code == "ownership_state_unavailable"
        assert (p.directory / "claims.json").read_bytes() == before
        with p.transaction():
            with pytest.raises(CatalogError) as caught:
                FileLease(p.directory / "mutation.lock", deadline=time.monotonic()).acquire()
            assert caught.value.code == "busy"
    finally:
        owner.stopped = True
        p.stop(owner)
        handle.close()


def test_reconciling_publish_owner_stamp_and_disable_local(tmp_path):
    p, owner = projection(tmp_path), entry(tmp_path)
    target = tmp_path / "skills/a/b"
    target.mkdir(parents=True)
    handle = p.start(owner)
    try:
        row = p.read()["registrations"][owner.registration_id]
        assert row["state"] == "reconciling"
        assert row["process"]["pid"] == os.getpid()
        assert float(row["process"]["birth_id"]) == psutil.Process().create_time()
        assert row["process"]["machine_id"]
        with pytest.raises(CatalogError) as caught:
            with p.generic(["unrelated"]):
                pytest.fail("incomplete scan admitted")
        assert caught.value.code == "busy"
        p.publish(owner, [record(owner, identifier=source_id("forged", "team") + ":item")])
        assert p.read()["targets"]["a/b"]["identifier"] == owner.source_id + ":item"
        for path in ("a", "a/b", "a/b/c"):
            with pytest.raises(CatalogError) as caught:
                with p.generic([path]):
                    pytest.fail("live ownership bypass")
            assert caught.value.code == "ownership_conflict"
        owner.stopped = True
        p.stop(owner)
        assert target.is_dir()
        assert p.read()["path_revisions"] == {}
        with p.generic(["a/b"]):
            (target / "local").write_text("local")
    finally:
        handle.close()


def test_publish_validates_all_records_before_writing(tmp_path):
    with started(tmp_path) as (p, owner):
        before = (p.directory / "claims.json").read_bytes()
        with pytest.raises(CatalogError) as caught:
            p.publish(owner, [record(owner, "missing"), record(owner, "other", actions=["invalid"])])
        assert caught.value.code == "invalid_response"
        assert (p.directory / "claims.json").read_bytes() == before
        p.publish(owner, [record(owner, "absent"), record(owner, "removed", managed_state="removed"),
                          record(owner, "no-revision", installation_revision=None, actions=[])])
        assert not p.read()["targets"]


def test_reserve_exact_replay_bind_cas_and_no_secret(tmp_path):
    with started(tmp_path) as (p, owner):
        plan = review(owner)
        admission = p.reserve(owner, plan)
        revision = p.read()["revision"]
        assert p.reserve(owner, plan) == admission
        assert p.read()["revision"] == revision
        assert set(admission) == {"registration_id", "generation", "review_id", "reservation_id",
                                 "expected_path_revision", "expected_installation_revision"}
        assert plan["review_token"] not in (p.directory / "claims.json").read_text()
        payload = {k: v for k, v in admission.items() if k != "generation"}
        payload["operation_id"] = uuid.uuid4().hex
        bound = p.bind(owner, payload)
        assert bound == dict(admission, operation_id=payload["operation_id"])
        revision = p.read()["revision"]
        assert p.bind(owner, payload) == bound
        assert p.read()["revision"] == revision
        with p.transaction() as doc:
            row = p.binding(owner, doc, admission, payload["operation_id"])
            assert row is doc["reservations"][admission["reservation_id"]]
        with pytest.raises(CatalogError):
            p.bind(owner, dict(payload, operation_id=uuid.uuid4().hex))
        with pytest.raises(CatalogError):
            p.reserve(owner, dict(plan, target_relative_path="other"))
        with pytest.raises(CatalogError):
            p.reserve(owner, review(owner, "new/child"))


def test_reserve_install_absent_and_update_exact_revision(tmp_path):
    target = tmp_path / "skills/a/b"
    target.mkdir(parents=True)
    with started(tmp_path) as (p, owner):
        p.publish(owner, [record(owner)])
        for plan in (review(owner, "a/b"), review(owner, "a/b", action="update", installation_revision="wrong")):
            with pytest.raises(CatalogError):
                p.reserve(owner, plan)
        plan = review(owner, "a/b", action="update", installation_revision="receipt-1")
        result = p.reserve(owner, plan)
        assert result["expected_installation_revision"] == "receipt-1"
        assert result["expected_path_revision"] == p.read()["targets"]["a/b"]["path_revision"]
        owner.generation = uuid.uuid4().hex
        with pytest.raises(CatalogError):
            p.reserve(owner, review(owner, "other"))


def test_path_revisions_survive_removed_claim_within_epoch(tmp_path):
    (tmp_path / "skills/a/b").mkdir(parents=True)
    with started(tmp_path) as (p, owner):
        p.publish(owner, [record(owner)])
        first = p.read()["targets"]["a/b"]["path_revision"]
        p.publish(owner, [record(owner)])
        assert p.read()["targets"]["a/b"]["path_revision"] == first
        p.publish(owner, [])
        p.publish(owner, [record(owner)])
        assert p.read()["targets"]["a/b"]["path_revision"] > first


def _remote_owner(home, connection):
    p, owner = projection(Path(home)), entry(Path(home), "remote")
    handle = p.start(owner)
    p.publish(owner, [])
    connection.send(owner.registration_id)
    connection.recv()
    handle.close()


@contextmanager
def remote_owner(home):
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=_remote_owner, args=(str(home), child))
    process.start()
    child.close()
    try:
        assert parent.poll(10)
        registration_id = parent.recv()
        yield process, registration_id
    finally:
        if process.is_alive():
            with suppress(BrokenPipeError, EOFError):
                parent.send("close")
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join(10)
        parent.close()


@pytest.mark.platforms("posix", "windows")
@pytest.mark.parametrize("damage", [None, "missing", "corrupt", "schema"])
def test_cross_process_liveness_missing_corrupt_and_crash(tmp_path, damage):
    p = projection(tmp_path)
    with remote_owner(tmp_path) as (process, rid):
        assert rid in p.read()["registrations"]
        path = p.directory / "claims.json"
        if damage == "missing":
            path.unlink()
        elif damage == "corrupt":
            path.write_text("{not json")
        elif damage == "schema":
            path.write_text('{"schema_version":2}')
        if damage:
            with pytest.raises(CatalogError) as caught:
                with p.generic(["local"]):
                    pytest.fail("damaged live projection admitted")
            assert caught.value.code == "ownership_state_unavailable"
        process.kill()
        process.join(10)
        with p.generic(["local"]):
            pass
        assert p.read()["registrations"] == {}


def test_presence_permission_failure_never_proves_absence(tmp_path, monkeypatch):
    from hermes_cli.skill_catalog import locks
    original = locks._open_file

    def denied(path):
        if path.name == "presence.lock":
            raise PermissionError("private")
        return original(path)

    monkeypatch.setattr(locks, "_open_file", denied)
    with pytest.raises(CatalogError) as caught:
        with projection(tmp_path).generic(["local"]):
            pytest.fail("unknown absence admitted")
    assert caught.value.code == "ownership_state_unavailable"


def test_no_provider_generic_still_holds_gate_and_writer_chain(tmp_path):
    p = projection(tmp_path)
    with p.generic(["a/b", "z"]):
        with pytest.raises(CatalogError):
            FileLease(p.directory / "mutation.lock", deadline=time.monotonic()).acquire()
        with pytest.raises(CatalogError):
            with writer_hold(tmp_path, ["a"], deadline=time.monotonic()):
                pytest.fail("writer chain absent")


@pytest.mark.platforms("posix")
def test_redirect_and_final_validation_refuse(tmp_path, monkeypatch):
    from hermes_cli.skill_catalog import projection as module
    p = projection(tmp_path)
    (tmp_path / "skills").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    original = module.writer_hold

    @contextmanager
    def swap(*args, **kwargs):
        with original(*args, **kwargs):
            (tmp_path / "skills/a").symlink_to(outside, target_is_directory=True)
            yield

    monkeypatch.setattr(module, "writer_hold", swap)
    with pytest.raises(CatalogError):
        with p.generic(["a/b"]):
            pytest.fail("final redirect unchecked")
    assert list(outside.iterdir()) == []


def test_save_failure_preserves_prior_bytes(tmp_path, monkeypatch):
    from hermes_cli.skill_catalog import projection as module
    with started(tmp_path) as (p, owner):
        before = (p.directory / "claims.json").read_bytes()

        def fail(*args, **kwargs):
            raise OSError("private write failure")

        monkeypatch.setattr(module.os, "replace", fail)
        with pytest.raises(CatalogError) as caught:
            p.reserve(owner, review(owner))
        assert caught.value.code == "ownership_state_unavailable"
        assert (p.directory / "claims.json").read_bytes() == before
        monkeypatch.undo()


def test_stop_saves_and_closes_presence_inside_root_and_gate(tmp_path, monkeypatch):
    p, owner = projection(tmp_path), entry(tmp_path)
    handle = p.start(owner)
    original = p.save

    def observed(doc):
        with pytest.raises(CatalogError):
            FileLease(p.directory / "mutation.lock", deadline=time.monotonic()).acquire()
        with pytest.raises(CatalogError):
            with writer_hold(tmp_path, ["other"], deadline=time.monotonic()):
                pytest.fail("disable released root before publication")
        return original(doc)

    monkeypatch.setattr(p, "save", observed)
    try:
        owner.stopped = True
        p.stop(owner)
        with FileLease(p.directory / "presence.lock", deadline=time.monotonic()):
            pass
    finally:
        monkeypatch.undo()
        handle.close()


def test_dead_registration_is_pruned_on_disk_while_another_remains(tmp_path):
    with started(tmp_path) as (p, owner):
        with remote_owner(tmp_path) as (process, rid):
            process.kill()
            process.join(10)
            assert rid not in p.read()["registrations"]
            stored = json.loads((p.directory / "claims.json").read_bytes())
            assert rid not in stored["registrations"]
            assert owner.registration_id in stored["registrations"]


def test_live_lease_omitted_from_valid_json_refuses(tmp_path):
    with started(tmp_path) as (p, _owner):
        with remote_owner(tmp_path) as (_process, rid):
            with p.transaction() as doc:
                doc["registrations"].pop(rid)
            with pytest.raises(CatalogError) as caught:
                with p.generic(["local"]):
                    pytest.fail("unindexed live owner ignored")
            assert caught.value.code == "ownership_state_unavailable"
            # Restore the row by ending its real lease before fixture cleanup.
            _process.kill()
            _process.join(10)


def test_publish_other_owner_ancestor_refuses_and_idempotent_publish_preserves_reservation(tmp_path):
    (tmp_path / "skills/a/b").mkdir(parents=True)
    with started(tmp_path) as (p, owner):
        p.publish(owner, [record(owner)])
        admission = p.reserve(owner, review(owner, "a/b", action="update", installation_revision="receipt-1"))
        p.publish(owner, [record(owner)])
        assert admission["reservation_id"] in p.read()["reservations"]
        second = entry(tmp_path, "second")
        handle = p.start(second)
        try:
            with pytest.raises(CatalogError):
                p.publish(second, [record(second, "a")])
        finally:
            second.stopped = True
            p.stop(second)
            handle.close()


def _reserve_racer(home, plan, owner_data, ready, go, result):
    owner = SimpleNamespace(**owner_data, authorized=lambda: True)
    ready.send("ready")
    go.recv()
    try:
        result.send(("ok", projection(Path(home)).reserve(owner, plan)))
    except CatalogError as exc:
        result.send(("error", exc.code))


@pytest.mark.platforms("posix", "windows")
@pytest.mark.parametrize("exact", [True, False])
def test_spawn_reservation_races_one_exact_admission_or_one_overlap_winner(tmp_path, exact):
    with started(tmp_path) as (p, owner):
        ctx = multiprocessing.get_context("spawn")
        first = review(owner)
        plans = [first, first if exact else review(owner, "new/child")]
        data = {key: value for key, value in vars(owner).items() if key != "authorized"}
        processes, channels = [], []
        try:
            for plan in plans:
                ready, child_ready = ctx.Pipe()
                go, child_go = ctx.Pipe()
                result, child_result = ctx.Pipe()
                process = ctx.Process(target=_reserve_racer, args=(str(tmp_path), plan, data,
                    child_ready, child_go, child_result))
                process.start()
                for channel in (child_ready, child_go, child_result):
                    channel.close()
                processes.append(process)
                channels.append((ready, go, result))
            for ready, _, _ in channels:
                assert ready.poll(10)
                assert ready.recv() == "ready"
            for _, go, _ in channels:
                go.send("reserve")
            results = []
            for _, _, result in channels:
                assert result.poll(10)
                results.append(result.recv())
            if exact:
                assert results[0] == results[1]
                assert results[0][0] == "ok"
            else:
                assert sorted(row[0] for row in results) == ["error", "ok"]
            assert len(p.read()["reservations"]) == 1
        finally:
            for process in processes:
                process.join(10)
                if process.is_alive():
                    process.kill()
                    process.join(10)
            for group in channels:
                for channel in group:
                    channel.close()


def test_directory_sync_failure_restores_exact_previous_bytes(tmp_path, monkeypatch):
    from hermes_cli.skill_catalog import projection as module
    with started(tmp_path) as (p, owner):
        before = (p.directory / "claims.json").read_bytes()
        calls = []
        original = module._sync_directory

        def fail_after_replace(path):
            calls.append(path)
            if len(calls) == 2:
                raise OSError("directory sync failed")
            return original(path)

        monkeypatch.setattr(module, "_sync_directory", fail_after_replace)
        with pytest.raises(CatalogError):
            p.reserve(owner, review(owner))
        assert (p.directory / "claims.json").read_bytes() == before
        monkeypatch.undo()


def test_full_validation_rejects_overlapping_claims_and_unknown_bind_stays_unknown(tmp_path):
    with started(tmp_path) as (p, owner):
        plan = review(owner)
        admission = p.reserve(owner, plan)
        payload = {key: value for key, value in admission.items() if key != "generation"}
        payload["operation_id"] = uuid.uuid4().hex
        p.bind(owner, payload)
        with p.transaction() as doc:
            doc["reservations"][admission["reservation_id"]]["state"] = "unknown"
        p.bind(owner, payload)
        assert p.read()["reservations"][admission["reservation_id"]]["state"] == "unknown"
        before = (p.directory / "claims.json").read_bytes()
        with pytest.raises(CatalogError):
            with p.transaction() as doc:
                doc["path_revisions"].update({"a": 1, "a/b": 1})
                claim = dict(registration_id=owner.registration_id, identifier=owner.source_id + ":item",
                             installation_revision="r", path_revision=1, managed_state="installed")
                doc["targets"].update({"a": claim, "a/b": dict(claim)})
        assert (p.directory / "claims.json").read_bytes() == before


def test_reconciling_owner_cannot_reserve_and_unauthorized_owner_cannot_bind(tmp_path):
    p, owner = projection(tmp_path), entry(tmp_path)
    handle = p.start(owner)
    try:
        with pytest.raises(CatalogError) as caught:
            p.reserve(owner, review(owner))
        assert caught.value.code == "busy"
        p.publish(owner, [])
        admission = p.reserve(owner, review(owner))
        owner.authorized = lambda: False
        payload = {key: value for key, value in admission.items() if key != "generation"}
        payload["operation_id"] = uuid.uuid4().hex
        with pytest.raises(CatalogError) as caught:
            p.bind(owner, payload)
        assert caught.value.code == "permission_denied"
    finally:
        owner.stopped = True
        p.stop(owner)
        handle.close()


def test_reserve_waits_for_other_reconciling_registration(tmp_path):
    with started(tmp_path) as (p, owner):
        second = entry(tmp_path, "second")
        handle = p.start(second)
        try:
            with pytest.raises(CatalogError) as caught:
                p.reserve(owner, review(owner))
            assert caught.value.code == "busy"
        finally:
            second.stopped = True
            p.stop(second)
            handle.close()


@pytest.mark.parametrize("field,value", [("path_revisions", {".hub/owned": 1}),
                                           ("revision", 9007199254740992)])
def test_save_rejects_reserved_target_and_overflow_before_touching_bytes(tmp_path, field, value):
    with started(tmp_path) as (p, _owner):
        before = (p.directory / "claims.json").read_bytes()
        with pytest.raises(CatalogError):
            with p.transaction() as doc:
                doc[field] = value
        assert (p.directory / "claims.json").read_bytes() == before
