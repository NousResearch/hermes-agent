from __future__ import annotations

import json
import multiprocessing
import shutil
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from hermes_cli import profiles
from hermes_cli import bot_transfer
from hermes_cli.bot_transfer import (
    BOT_ID_FILENAME,
    export_bot_profile,
    get_profile_bot_id,
    import_bot_profile,
    pull_bot_profile,
    profile_is_cloneable,
    push_bot_profile,
    set_profile_cloneable,
)


@pytest.fixture
def profile_root(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    profile_root = root / "profiles"
    profile_root.mkdir(parents=True)
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: root)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profile_root)
    return root


def _source_profile(root: Path, name: str = "helper") -> Path:
    source = root / "profiles" / name
    (source / "skills" / "demo").mkdir(parents=True)
    (source / "memories").mkdir()
    (source / "sessions").mkdir()
    (source / "SOUL.md").write_text("Helpful bot\n", encoding="utf-8")
    (source / "config.yaml").write_text(
        "model: test\ngateway:\n  bot_sharing:\n    allow_push: true\n",
        encoding="utf-8",
    )
    (source / "skills" / "demo" / "SKILL.md").write_text("# Demo\n", encoding="utf-8")
    (source / "memories" / "MEMORY.md").write_text("private memory\n", encoding="utf-8")
    (source / "sessions" / "chat.json").write_text("private chat\n", encoding="utf-8")
    (source / ".env").write_text("OPENAI_API_KEY=private\n", encoding="utf-8")
    (source / "auth.json").write_text('{"token":"private"}\n', encoding="utf-8")
    return source


def _hold_clone_lock(profiles_root: str, ready) -> None:
    profiles._get_profiles_root = lambda: Path(profiles_root)
    with bot_transfer._clone_import_lock():
        ready.set()
        time.sleep(30)


def test_bot_export_is_definition_only_and_identity_is_stable(profile_root, tmp_path):
    source = _source_profile(profile_root)

    first, first_id = export_bot_profile("helper", str(tmp_path / "first.tar.gz"))
    second, second_id = export_bot_profile("helper", str(tmp_path / "second.tar.gz"))

    assert first_id == second_id == get_profile_bot_id(source)
    with tarfile.open(first, "r:gz") as archive:
        names = set(archive.getnames())
    assert "helper/SOUL.md" in names
    assert "helper/skills/demo/SKILL.md" in names
    assert f"helper/{BOT_ID_FILENAME}" in names
    assert not any("memories" in name for name in names)
    assert not any("sessions" in name for name in names)
    assert not any(name.endswith(("/.env", "/auth.json")) for name in names)
    with tarfile.open(first, "r:gz") as archive:
        config = archive.extractfile("helper/config.yaml").read().decode("utf-8")
        profile_meta = archive.extractfile("helper/profile.yaml").read().decode("utf-8")
    assert "bot_sharing" not in config
    assert "cloneable: false" in profile_meta


def test_bot_export_excludes_nested_credentials_and_cron_runtime(profile_root, tmp_path):
    source = _source_profile(profile_root)
    (source / "skills" / "demo" / ".env").write_text(
        "TOKEN=private\n", encoding="utf-8"
    )
    (source / "plugins" / "demo").mkdir(parents=True)
    (source / "plugins" / "demo" / "credentials.bin").write_bytes(b"\x00private")
    (source / "cron" / "output").mkdir(parents=True)
    (source / "cron" / "jobs.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "daily",
                        "name": "Daily summary",
                        "prompt": "Summarize the day",
                        "schedule": {"kind": "cron", "expr": "0 9 * * *"},
                        "repeat": {"times": None, "completed": 7},
                        "enabled": True,
                        "state": "scheduled",
                        "next_run_at": "2099-01-01T09:00:00+00:00",
                        "last_run_at": "2026-01-01T09:00:00+00:00",
                        "last_status": "success",
                        "last_delivery_unverified": {"platform": "telegram"},
                        "monitor_state": {"last_output_hash": "secret-state"},
                        "run_claim": {"by": "source"},
                        "fire_claim": {"by": "source"},
                        "failure_streak": 3,
                        "deliver": "telegram:123456:99",
                        "failure_deliver": "discord:private",
                        "origin": {"chat_id": "123456", "thread_id": "99"},
                        "workdir": "C:/source/private",
                        "attach_to_session": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (source / "cron" / "output" / "last.txt").write_text(
        "private\n", encoding="utf-8"
    )
    (source / "cron" / "ticker-heartbeat").write_text("runtime\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported binary file"):
        export_bot_profile("helper", str(tmp_path / "binary.tar.gz"))

    (source / "plugins" / "demo" / "credentials.bin").unlink()
    archive, _ = export_bot_profile("helper", str(tmp_path / "safe.tar.gz"))
    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
        cron_jobs = json.loads(tar.extractfile("helper/cron/jobs.json").read())[
            "jobs"
        ]
    assert "helper/cron/jobs.json" in names
    assert not any(name.endswith("/.env") for name in names)
    assert not any("cron/output" in name for name in names)
    assert "helper/cron/ticker-heartbeat" not in names
    assert cron_jobs == [
        {
            "id": "daily",
            "name": "Daily summary",
            "prompt": "Summarize the day",
            "schedule": {"kind": "cron", "expr": "0 9 * * *"},
            "repeat": {"times": None, "completed": 0},
            "enabled": False,
            "state": "paused",
            "paused_at": None,
            "paused_reason": "Imported bot clone requires review.",
            "next_run_at": None,
            "deliver": "local",
        }
    ]


def test_pull_policy_is_per_profile_and_fails_closed_for_malformed_yaml(profile_root):
    source = _source_profile(profile_root)

    assert profile_is_cloneable("helper") is False
    bot_id = set_profile_cloneable("helper", True)
    assert profile_is_cloneable("helper") is True
    assert bot_id == get_profile_bot_id(source)

    (source / "profile.yaml").write_text("cloneable: 'false'\n", encoding="utf-8")
    assert profile_is_cloneable("helper") is False


def test_clone_policy_rejects_profile_path_traversal(profile_root):
    outside = profile_root / "outside"
    outside.mkdir()

    assert profile_is_cloneable("../outside") is False
    with pytest.raises(ValueError, match="Invalid profile name"):
        set_profile_cloneable("../outside", True)

    assert not (outside / BOT_ID_FILENAME).exists()
    assert not (outside / "profile.yaml").exists()


def test_bot_identity_is_published_atomically_for_concurrent_exporters(
    profile_root,
):
    source = _source_profile(profile_root)

    with ThreadPoolExecutor(max_workers=8) as pool:
        identities = set(pool.map(lambda _: bot_transfer.ensure_profile_bot_id(source), range(8)))

    assert len(identities) == 1
    assert get_profile_bot_id(source) == identities.pop()
    assert not list(source.glob(f".{BOT_ID_FILENAME}.*.tmp"))


def test_bot_identity_publish_failure_leaves_no_partial_file(
    profile_root, monkeypatch
):
    source = _source_profile(profile_root)
    monkeypatch.setattr(bot_transfer.os, "link", lambda *_args: (_ for _ in ()).throw(OSError("fail")))

    with pytest.raises(OSError, match="fail"):
        bot_transfer.ensure_profile_bot_id(source)

    assert not (source / BOT_ID_FILENAME).exists()
    assert not list(source.glob(f".{BOT_ID_FILENAME}.*.tmp"))


def test_bot_import_allows_rename_but_rejects_name_and_identity_collisions(profile_root, tmp_path):
    source = _source_profile(profile_root)
    archive, bot_id = export_bot_profile("helper", str(tmp_path / "helper.tar.gz"))
    shutil.rmtree(source)

    imported, imported_id = import_bot_profile(str(archive), name="helper-copy")

    assert imported.name == "helper-copy"
    assert imported_id == bot_id == get_profile_bot_id(imported)
    assert (imported / "SOUL.md").read_text(encoding="utf-8") == "Helpful bot\n"
    assert profile_is_cloneable("helper-copy") is False

    with pytest.raises(FileExistsError, match="already exists as profile"):
        import_bot_profile(str(archive), name="another-name")

    other = _source_profile(profile_root, "occupied")
    (other / BOT_ID_FILENAME).unlink(missing_ok=True)
    other_archive, _ = export_bot_profile("occupied", str(tmp_path / "occupied.tar.gz"))
    shutil.rmtree(other)
    (profile_root / "profiles" / "occupied").mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        import_bot_profile(str(other_archive), name="occupied")


def test_bot_import_rejects_user_state_even_from_safe_tar(profile_root, tmp_path):
    staged = tmp_path / "staged" / "unsafe"
    staged.mkdir(parents=True)
    (staged / BOT_ID_FILENAME).write_text("a8c214f7-37ee-4f50-95a4-939a51631283\n", encoding="utf-8")
    (staged / "sessions").mkdir()
    (staged / "sessions" / "chat.json").write_text("private\n", encoding="utf-8")
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(staged, arcname="unsafe")

    with pytest.raises(ValueError, match="disallowed profile data: sessions"):
        import_bot_profile(str(archive))


def test_bot_import_resets_sender_sharing_policies(profile_root, tmp_path):
    staged = tmp_path / "staged" / "shared"
    staged.mkdir(parents=True)
    (staged / BOT_ID_FILENAME).write_text(
        "a8c214f7-37ee-4f50-95a4-939a51631283\n", encoding="utf-8"
    )
    (staged / "profile.yaml").write_text("cloneable: true\n", encoding="utf-8")
    (staged / "config.yaml").write_text(
        "model: test\ngateway:\n  bot_sharing:\n    allow_push: true\n",
        encoding="utf-8",
    )
    archive = tmp_path / "shared.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(staged, arcname="shared")

    imported, _ = import_bot_profile(str(archive))

    assert profile_is_cloneable("shared") is False
    assert "bot_sharing" not in (imported / "config.yaml").read_text(encoding="utf-8")


def test_bot_import_resanitizes_cron_jobs_and_keeps_them_paused(profile_root, tmp_path):
    staged = tmp_path / "staged" / "scheduled"
    (staged / "cron").mkdir(parents=True)
    (staged / BOT_ID_FILENAME).write_text(
        "a8c214f7-37ee-4f50-95a4-939a51631283\n", encoding="utf-8"
    )
    (staged / "cron" / "jobs.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "source-job",
                        "prompt": "Run safely",
                        "schedule": {"kind": "interval", "minutes": 5},
                        "repeat": {"times": 10, "completed": 4},
                        "enabled": True,
                        "next_run_at": "2000-01-01T00:00:00+00:00",
                        "deliver": "telegram:123456:99",
                        "failure_deliver": "telegram:123456:99",
                        "origin": {"chat_id": "123456", "thread_id": "99"},
                        "monitor_state": {"last_output_hash": "source"},
                        "last_run_at": "1999-01-01T00:00:00+00:00",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    archive = tmp_path / "scheduled.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(staged, arcname="scheduled")

    imported, _ = import_bot_profile(str(archive))
    job = json.loads((imported / "cron" / "jobs.json").read_text(encoding="utf-8"))[
        "jobs"
    ][0]

    assert job["enabled"] is False
    assert job["state"] == "paused"
    assert job["next_run_at"] is None
    assert job["deliver"] == "local"
    assert job["repeat"] == {"times": 10, "completed": 0}
    from cron.jobs import is_job_runnable

    assert is_job_runnable(job) is False
    assert not {
        "origin",
        "failure_deliver",
        "monitor_state",
        "last_run_at",
    } & job.keys()


def test_bot_clone_rehomes_cron_scripts_and_runs_them_from_receiver(
    profile_root, tmp_path, monkeypatch
):
    source = _source_profile(profile_root)
    scripts = source / "scripts"
    (scripts / "checks").mkdir(parents=True)
    script = scripts / "probe.py"
    monitor_script = scripts / "checks" / "monitor.py"
    script.write_text("print('copied job')\n", encoding="utf-8")
    monitor_script.write_text("print('copied monitor')\n", encoding="utf-8")
    (source / "cron").mkdir()
    (source / "cron" / "jobs.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "script-job",
                        "prompt": "",
                        "schedule": {"kind": "interval", "minutes": 5},
                        "script": str(script.resolve()),
                        "no_agent": True,
                    },
                    {
                        "id": "monitor-job",
                        "prompt": "Check changes",
                        "schedule": {"kind": "interval", "minutes": 5},
                        "monitor_script": str(monitor_script.resolve()),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    archive, _ = export_bot_profile("helper", str(tmp_path / "helper.tar.gz"))
    shutil.rmtree(source)
    imported, _ = import_bot_profile(str(archive), name="receiver")
    jobs = json.loads((imported / "cron" / "jobs.json").read_text(encoding="utf-8"))[
        "jobs"
    ]

    assert jobs[0]["script"] == "probe.py"
    assert jobs[1]["monitor_script"] == "checks/monitor.py"
    from cron import scheduler

    monkeypatch.setattr(scheduler, "_hermes_home", imported)
    assert scheduler._run_job_script(jobs[0]["script"]) == (True, "copied job")
    assert scheduler._run_job_script(jobs[1]["monitor_script"]) == (
        True,
        "copied monitor",
    )


@pytest.mark.parametrize("field", ["script", "monitor_script"])
@pytest.mark.parametrize("path_kind", ["outside", "missing"])
def test_bot_export_rejects_uncloneable_cron_script_paths(
    profile_root, tmp_path, field, path_kind
):
    source = _source_profile(profile_root)
    (source / "cron").mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("print('outside')\n", encoding="utf-8")
    value = str(outside.resolve()) if path_kind == "outside" else "missing.py"
    (source / "cron" / "jobs.json").write_text(
        json.dumps({"jobs": [{"id": "unsafe", field: value}]}),
        encoding="utf-8",
    )
    archive = tmp_path / f"unsafe-export-{field}-{path_kind}.tar.gz"

    with pytest.raises(ValueError, match="invalid cron job definitions"):
        export_bot_profile("helper", str(archive))

    assert not archive.exists()


@pytest.mark.parametrize("field", ["script", "monitor_script"])
@pytest.mark.parametrize("value", ["../outside.py", "missing.py"])
def test_bot_import_rejects_unsafe_or_missing_cron_script_paths(
    profile_root, tmp_path, field, value
):
    staged = tmp_path / "staged" / "unsafe-script"
    (staged / "cron").mkdir(parents=True)
    (staged / "scripts").mkdir()
    (staged / BOT_ID_FILENAME).write_text(
        "a8c214f7-37ee-4f50-95a4-939a51631283\n", encoding="utf-8"
    )
    (staged / "cron" / "jobs.json").write_text(
        json.dumps({"jobs": [{"id": "unsafe", field: value}]}),
        encoding="utf-8",
    )
    archive = tmp_path / f"unsafe-{field}-{Path(value).name}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(staged, arcname="unsafe-script")

    with pytest.raises(ValueError, match="invalid cron job definitions"):
        import_bot_profile(str(archive))

    assert not (profile_root / "profiles" / "unsafe-script").exists()


def test_bot_import_rejects_absolute_cron_script_path(profile_root, tmp_path):
    staged = tmp_path / "staged" / "absolute-script"
    (staged / "cron").mkdir(parents=True)
    (staged / "scripts").mkdir()
    script = staged / "scripts" / "probe.py"
    script.write_text("print('must stay relative')\n", encoding="utf-8")
    (staged / BOT_ID_FILENAME).write_text(
        "a8c214f7-37ee-4f50-95a4-939a51631283\n", encoding="utf-8"
    )
    (staged / "cron" / "jobs.json").write_text(
        json.dumps({"jobs": [{"id": "absolute", "script": str(script.resolve())}]}),
        encoding="utf-8",
    )
    archive = tmp_path / "absolute-script.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(staged, arcname="absolute-script")

    with pytest.raises(ValueError, match="invalid cron job definitions"):
        import_bot_profile(str(archive))

    assert not (profile_root / "profiles" / "absolute-script").exists()


def test_bot_import_reactivates_a_deleted_profile_name(profile_root, tmp_path):
    source = _source_profile(profile_root)
    archive, _ = export_bot_profile("helper", str(tmp_path / "helper.tar.gz"))
    shutil.rmtree(source)
    destination = profile_root / "profiles" / "helper"
    profiles.mark_named_profile_deleted(destination)

    imported, _ = import_bot_profile(str(archive))

    assert imported == destination
    assert profiles.profile_exists("helper") is True
    assert any(item.name == "helper" for item in profiles.list_profiles())
    assert ("helper", destination) in profiles.profiles_to_serve(multiplex=True)
    assert profiles.named_profile_is_deleted(destination) is False


def test_profile_import_restores_deleted_marker_when_publication_fails(
    profile_root, tmp_path, monkeypatch
):
    source = _source_profile(profile_root)
    archive, _ = export_bot_profile("helper", str(tmp_path / "helper.tar.gz"))
    shutil.rmtree(source)
    destination = profile_root / "profiles" / "helper"
    profiles.mark_named_profile_deleted(destination)
    monkeypatch.setattr(
        profiles.shutil,
        "move",
        lambda *_args: (_ for _ in ()).throw(OSError("publish failed")),
    )

    with pytest.raises(OSError, match="publish failed"):
        import_bot_profile(str(archive))

    assert profiles.named_profile_is_deleted(destination) is True
    assert profiles.profile_exists("helper") is False


def test_bot_import_bounds_expanded_archive_and_leaves_no_profile(
    profile_root, tmp_path, monkeypatch
):
    staged = tmp_path / "staged" / "large"
    staged.mkdir(parents=True)
    (staged / BOT_ID_FILENAME).write_text(
        "a8c214f7-37ee-4f50-95a4-939a51631283\n", encoding="utf-8"
    )
    (staged / "SOUL.md").write_bytes(b"0" * 2048)
    archive = tmp_path / "large.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(staged, arcname="large")
    assert archive.stat().st_size < 1024
    monkeypatch.setattr(bot_transfer, "MAX_BOT_CLONE_BYTES", 1024)

    with pytest.raises(ValueError, match="expanded-size limit"):
        import_bot_profile(str(archive))

    assert not (profile_root / "profiles" / "large").exists()


def test_bot_export_rejects_source_outside_import_budget(
    profile_root, tmp_path, monkeypatch
):
    source = _source_profile(profile_root)
    (source / "SOUL.md").write_bytes(b"0" * 2048)
    monkeypatch.setattr(bot_transfer, "MAX_BOT_CLONE_BYTES", 1024)

    with pytest.raises(ValueError, match="expanded-size limit"):
        export_bot_profile("helper", str(tmp_path / "large.tar.gz"))

    assert not (tmp_path / "large.tar.gz").exists()


def test_bot_export_rejects_too_many_archive_members(
    profile_root, tmp_path, monkeypatch
):
    _source_profile(profile_root)
    monkeypatch.setattr(bot_transfer, "MAX_BOT_CLONE_MEMBERS", 3)

    with pytest.raises(ValueError, match="member limit"):
        export_bot_profile("helper", str(tmp_path / "many.tar.gz"))

    assert not (tmp_path / "many.tar.gz").exists()


def test_clone_import_lock_cannot_be_stolen_and_recovers_after_owner_death(
    profile_root,
):
    profiles_root = profile_root / "profiles"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    holder = context.Process(target=_hold_clone_lock, args=(str(profiles_root), ready))
    holder.start()
    try:
        assert ready.wait(timeout=10)
        with pytest.raises(TimeoutError):
            with bot_transfer._clone_import_lock(timeout=0.1):
                pass
    finally:
        holder.terminate()
        holder.join(timeout=10)
    assert not holder.is_alive()

    with bot_transfer._clone_import_lock(timeout=1):
        pass


def test_pull_downloads_and_imports_remote_clone(profile_root, tmp_path, monkeypatch):
    source = _source_profile(profile_root)
    archive, bot_id = export_bot_profile("helper", str(tmp_path / "helper.tar.gz"))
    payload = archive.read_bytes()
    shutil.rmtree(source)
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == "https://remote.example/v1/bots/helper/clone"
        assert request.headers["Authorization"] == "Bearer remote-secret"
        return httpx.Response(200, content=payload, headers={"Content-Type": "application/gzip"})

    monkeypatch.setenv("GATEWAY_PROXY_KEY", "remote-secret")
    monkeypatch.setattr(
        bot_transfer.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )

    imported, imported_id = pull_bot_profile(
        "helper", remote="https://remote.example", name="local-helper"
    )

    assert imported.name == "local-helper"
    assert imported_id == bot_id


def test_push_uploads_clone_with_requested_remote_name(profile_root, monkeypatch):
    _source_profile(profile_root)
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "https://remote.example/v1/bots/clone?name=remote-helper"
        assert request.headers["Authorization"] == "Bearer remote-secret"
        assert request.headers["Content-Type"] == "application/gzip"
        assert 0 < len(request.content) <= bot_transfer.MAX_BOT_CLONE_BYTES
        return httpx.Response(
            201,
            json={
                "object": "hermes.bot_clone",
                "name": "remote-helper",
                "bot_id": "a8c214f7-37ee-4f50-95a4-939a51631283",
            },
        )

    monkeypatch.setenv("GATEWAY_PROXY_KEY", "remote-secret")
    monkeypatch.setattr(
        bot_transfer.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )

    name, bot_id = push_bot_profile(
        "helper", remote="https://remote.example/v1", name="remote-helper"
    )

    assert name == "remote-helper"
    assert bot_id == "a8c214f7-37ee-4f50-95a4-939a51631283"


def test_remote_clone_refuses_cleartext_bearer_auth_off_loopback(monkeypatch):
    monkeypatch.setenv("GATEWAY_PROXY_KEY", "remote-secret")

    with pytest.raises(ValueError, match="must use HTTPS"):
        pull_bot_profile("helper", remote="http://gateway.example")


def test_pull_wraps_network_failures(profile_root, monkeypatch):
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    monkeypatch.setenv("GATEWAY_PROXY_KEY", "remote-secret")
    monkeypatch.setattr(
        bot_transfer.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )

    with pytest.raises(bot_transfer.BotTransferError, match="request failed"):
        pull_bot_profile("helper", remote="https://remote.example")


def test_push_wraps_network_failures(profile_root, monkeypatch):
    _source_profile(profile_root)
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    monkeypatch.setenv("GATEWAY_PROXY_KEY", "remote-secret")
    monkeypatch.setattr(
        bot_transfer.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )

    with pytest.raises(bot_transfer.BotTransferError, match="request failed"):
        push_bot_profile("helper", remote="https://remote.example")


def test_push_rejects_malformed_success_response(profile_root, monkeypatch):
    _source_profile(profile_root)
    real_client = httpx.Client

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(201, json={"object": "hermes.bot_clone"})

    monkeypatch.setenv("GATEWAY_PROXY_KEY", "remote-secret")
    monkeypatch.setattr(
        bot_transfer.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )

    with pytest.raises(bot_transfer.BotTransferError, match="invalid bot clone response"):
        push_bot_profile("helper", remote="https://remote.example")
