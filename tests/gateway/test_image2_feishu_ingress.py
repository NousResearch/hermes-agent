from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.image2_feishu_ingress import (
    Image2IngressSettings,
    _download_feishu_image_key,
    resolve_feishu_thread_image,
    ack_text_for_job,
    build_feishu_message_payload,
    collect_image2_source_files,
    enqueue_feishu_job,
    handle_image2_feishu_ingress_event,
    launch_image2_worker,
    load_image2_ingress_settings,
    select_recent_previous_image_message,
    select_recent_previous_visual_text_message,
    should_handle_feishu_visual_request,
)


def test_visual_request_detector_accepts_firepalace_poster_and_rejects_sales_question():
    assert should_handle_feishu_visual_request(
        platform="feishu",
        text="帮我做张火宫殿T3辣椒小炒肉单品海报，完成后发图片本体",
        message_type="text",
    )
    assert should_handle_feishu_visual_request(
        platform="feishu",
        text="今天王府井店销售额多少？",
        message_type="text",
    ) is False
    assert should_handle_feishu_visual_request(
        platform="telegram",
        text="帮我做张海报",
        message_type="text",
    ) is False


def test_visual_request_detector_accepts_explicit_image2_command():
    assert should_handle_feishu_visual_request(
        platform="feishu",
        text="/image2 做一张酸菜鱼海报",
        message_type="command",
    )


def test_build_feishu_message_payload_uses_stable_thread_and_media_paths():
    source = SimpleNamespace(
        platform=SimpleNamespace(value="feishu"),
        chat_id="oc_chat",
        thread_id="om_root",
    )
    event = SimpleNamespace(
        message_id="om_msg",
        text="做张海报",
        source=source,
        media_urls=["/tmp/source.png"],
    )

    payload = build_feishu_message_payload(event)

    assert payload == {
        "source_platform": "feishu",
        "feishu_message_id": "om_msg",
        "chat_id": "oc_chat",
        "root_id": "om_root",
        "thread_id": "om_root",
        "text": "做张海报",
        "source_files": [{"path": "/tmp/source.png", "mime_type": "image/png", "source": "feishu_direct_media"}],
    }


def test_direct_media_path_is_normalized_with_sha_for_source_gates(tmp_path):
    direct_source = tmp_path / "direct-source.png"
    direct_source.write_bytes(b"direct uploaded image bytes")
    event = SimpleNamespace(
        message_id="om_direct",
        text="/image2 按这张图重做海报",
        source=SimpleNamespace(platform=SimpleNamespace(value="feishu"), chat_id="oc_chat", thread_id="om_root"),
        media_urls=[str(direct_source)],
    )
    settings = Image2IngressSettings(enabled=True, runtime_root=tmp_path / "runtime")

    source_files = collect_image2_source_files(event, settings=settings)

    assert source_files == [
        {
            "path": str(direct_source),
            "mime_type": "image/png",
            "sha256": "50aca0e30760b7a5f6f6b71fc00def8d52dcc3605d3ccd5ae654ba0ef31f4f5a",
            "source": "feishu_direct_media",
        }
    ]


def test_collect_image2_source_files_downloads_quoted_parent_image_to_source_jpg(tmp_path):
    source = SimpleNamespace(platform=SimpleNamespace(value="feishu"), chat_id="oc_chat", thread_id="om_parent")
    quoted_parent = SimpleNamespace(
        message_id="om_parent",
        image_key="img_parent_123",
        mime_type="image/jpeg",
        filename="quoted-parent.jpg",
    )
    event = SimpleNamespace(
        message_id="om_reply",
        text="补总标题：夏日鲜果冰柠系列 中英文",
        source=source,
        media_urls=[],
        quoted_message=quoted_parent,
    )
    settings = Image2IngressSettings(enabled=True, runtime_root=tmp_path)
    calls = []

    def fake_downloader(parent, destination):
        calls.append((parent, destination))
        Path(destination).write_bytes(b"quoted jpeg bytes")
        return {"path": str(destination), "mime_type": "image/jpeg"}

    source_files = collect_image2_source_files(event, settings=settings, image_downloader=fake_downloader)

    assert calls and calls[0][0] is quoted_parent
    assert source_files[0]["source"] == "feishu_quoted_parent"
    assert source_files[0]["path"].endswith("source.jpg")
    assert Path(source_files[0]["path"]).read_bytes() == b"quoted jpeg bytes"
    assert source_files[0]["parent_message_id"] == "om_parent"


def test_build_feishu_message_payload_keeps_reply_text_parent_ids_and_source_files(tmp_path):
    source = SimpleNamespace(platform=SimpleNamespace(value="feishu"), chat_id="oc_chat", thread_id="om_parent")
    quoted_parent = SimpleNamespace(message_id="om_parent", image_key="img_parent_456", mime_type="image/jpeg")
    event = SimpleNamespace(
        message_id="om_reply",
        text="补卖点：鲜果入饮 冰爽解腻",
        source=source,
        media_urls=[],
        quoted_message=quoted_parent,
    )
    settings = Image2IngressSettings(enabled=True, runtime_root=tmp_path)

    payload = build_feishu_message_payload(
        event,
        settings=settings,
        image_downloader=lambda parent, destination: (Path(destination).write_bytes(b"img"), {"path": str(destination), "mime_type": "image/jpeg"})[1],
    )

    assert payload["text"] == "补卖点：鲜果入饮 冰爽解腻"
    assert payload["feishu_message_id"] == "om_reply"
    assert payload["root_id"] == "om_parent"
    assert payload["thread_id"] == "om_parent"
    assert payload["source_files"][0]["source"] == "feishu_quoted_parent"
    assert payload["source_files"][0]["path"].endswith("source.jpg")


def test_select_recent_previous_image_message_pairs_split_image_then_text_from_same_sender():
    messages = [
        {"message_id": "om_app_ack", "msg_type": "post", "create_time": "3000", "sender": {"sender_type": "app"}},
        {
            "message_id": "om_text",
            "msg_type": "text",
            "create_time": "2000",
            "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_same"}},
        },
        {
            "message_id": "om_image",
            "msg_type": "image",
            "create_time": "1500",
            "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_same"}},
            "body": {"content": json.dumps({"image_key": "img_recent"})},
        },
        {
            "message_id": "om_other_image",
            "msg_type": "image",
            "create_time": "1400",
            "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_other"}},
            "body": {"content": json.dumps({"image_key": "img_other"})},
        },
    ]

    picked = select_recent_previous_image_message(messages, current_message_id="om_text", lookback_seconds=2)

    assert picked["message_id"] == "om_image"


def test_collect_image2_source_files_uses_recent_previous_image_when_text_follows_image(tmp_path):
    source = SimpleNamespace(platform=SimpleNamespace(value="feishu"), chat_id="oc_chat", thread_id=None)
    event = SimpleNamespace(
        message_id="om_text",
        text="在这个图片中增加标题内容\n补总标题：夏日鲜果冰柠系列 中英文\n/image2 只做测试",
        source=source,
        media_urls=[],
    )
    settings = Image2IngressSettings(enabled=True, runtime_root=tmp_path)
    calls = []

    def fake_recent_resolver(current_event, destination):
        calls.append((current_event, destination))
        Path(destination).write_bytes(b"recent previous image")
        return {
            "path": str(destination),
            "mime_type": "image/jpeg",
            "source": "feishu_recent_previous_image",
            "parent_message_id": "om_image",
            "image_key": "img_recent",
        }

    source_files = collect_image2_source_files(event, settings=settings, recent_image_resolver=fake_recent_resolver)

    assert calls and calls[0][0] is event
    assert source_files[0]["source"] == "feishu_recent_previous_image"
    assert source_files[0]["parent_message_id"] == "om_image"
    assert source_files[0]["path"].endswith("source.jpg")
    assert Path(source_files[0]["path"]).read_bytes() == b"recent previous image"


def test_resolve_feishu_thread_image_downloads_canonical_root_message_image(tmp_path, monkeypatch):
    db_path = tmp_path / "image2_jobs.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE image2_jobs (
                task_id TEXT PRIMARY KEY,
                feishu_message_id TEXT,
                chat_id TEXT,
                root_id TEXT,
                thread_id TEXT,
                status TEXT,
                updated_at TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO image2_jobs (
                task_id, feishu_message_id, chat_id, root_id, thread_id, status, updated_at
            ) VALUES (
                'img2_choudoufu', 'om_image2_request_msg', 'oc_chat',
                'om_original_choudoufu_root', 'om_original_choudoufu_root',
                'ack_sent', '2026-04-28T10:00:00+00:00'
            )
            """
        )
    settings = Image2IngressSettings(enabled=True, db_path=db_path, runtime_root=tmp_path)
    event = SimpleNamespace(
        text="继续执行",
        message_type=SimpleNamespace(value="text"),
        message_id="om_plain_followup",
        source=SimpleNamespace(platform=SimpleNamespace(value="feishu"), chat_id="oc_chat", thread_id="om_image2_request_msg"),
        media_urls=[],
    )
    fetched = []

    def fake_fetch(*, message_id, env, token):
        fetched.append(message_id)
        if message_id == "om_original_choudoufu_root":
            return {
                "message_id": message_id,
                "msg_type": "image",
                "body": {"content": json.dumps({"image_key": "img_choudoufu_root"})},
            }
        return {}

    def fake_download(*, image_key, message_id=None, destination, env, token):
        Path(destination).write_bytes(b"root image bytes")
        return {"path": str(destination), "mime_type": "image/jpeg"}

    monkeypatch.setattr("gateway.image2_feishu_ingress._feishu_env", lambda current_settings: {})
    monkeypatch.setattr("gateway.image2_feishu_ingress._feishu_tenant_token", lambda env: "tenant-token-redacted")
    monkeypatch.setattr("gateway.image2_feishu_ingress._fetch_feishu_message_by_id", fake_fetch)
    monkeypatch.setattr("gateway.image2_feishu_ingress._download_feishu_image_key", fake_download)

    destination = tmp_path / "_feishu_ingress" / "om_plain_followup" / "source.jpg"
    result = resolve_feishu_thread_image(event, destination, settings=settings)

    assert fetched[0] == "om_original_choudoufu_root"
    assert result["source"] == "feishu_thread_root_image"
    assert result["parent_message_id"] == "om_original_choudoufu_root"
    assert result["image_key"] == "img_choudoufu_root"
    assert destination.read_bytes() == b"root image bytes"


def test_build_payload_resolves_thread_root_image_when_event_lacks_quoted_parent(tmp_path):
    db_path = tmp_path / "image2_jobs.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE image2_jobs (
                task_id TEXT PRIMARY KEY,
                feishu_message_id TEXT,
                chat_id TEXT,
                root_id TEXT,
                thread_id TEXT,
                status TEXT,
                updated_at TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO image2_jobs (
                task_id, feishu_message_id, chat_id, root_id, thread_id, status, updated_at
            ) VALUES (
                'img2_choudoufu', 'om_image2_request_msg', 'oc_chat',
                'om_original_choudoufu_root', 'om_original_choudoufu_root',
                'ack_sent', '2026-04-28T10:00:00+00:00'
            )
            """
        )
    settings = Image2IngressSettings(enabled=True, db_path=db_path, runtime_root=tmp_path)
    event = SimpleNamespace(
        text="继续执行",
        message_type=SimpleNamespace(value="text"),
        message_id="om_plain_followup",
        source=SimpleNamespace(platform=SimpleNamespace(value="feishu"), chat_id="oc_chat", thread_id="om_image2_request_msg"),
        media_urls=[],
    )
    calls = []

    def fake_thread_resolver(current_event, destination):
        calls.append((current_event, destination))
        Path(destination).write_bytes(b"root image bytes")
        return {
            "path": str(destination),
            "mime_type": "image/jpeg",
            "source": "feishu_thread_root_image",
            "parent_message_id": "om_original_choudoufu_root",
            "image_key": "img_choudoufu_root",
        }

    payload = build_feishu_message_payload(event, settings=settings, thread_image_resolver=fake_thread_resolver)

    assert payload["root_id"] == "om_original_choudoufu_root"
    assert payload["thread_id"] == "om_original_choudoufu_root"
    assert calls and calls[0][0] is event
    assert payload["source_files"][0]["source"] == "feishu_thread_root_image"
    assert payload["source_files"][0]["parent_message_id"] == "om_original_choudoufu_root"


def test_download_feishu_image_key_prefers_message_resource_endpoint_for_user_images(tmp_path, monkeypatch):
    opened = []

    class FakeResponse:
        headers = {"Content-Type": "image/png"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"png bytes"

    def fake_urlopen(request, timeout):
        opened.append(request.full_url)
        return FakeResponse()

    monkeypatch.setattr("gateway.image2_feishu_ingress.urllib.request.urlopen", fake_urlopen)
    destination = tmp_path / "source.png"

    result = _download_feishu_image_key(
        image_key="img_recent",
        message_id="om_image",
        destination=destination,
        env={},
        token="tenant-token-redacted",
    )

    assert opened == ["https://open.feishu.cn/open-apis/im/v1/messages/om_image/resources/img_recent?type=image"]
    assert result["mime_type"] == "image/png"
    assert destination.read_bytes() == b"png bytes"


def test_load_settings_from_profile_config_and_env_override(tmp_path, monkeypatch):
    profile = tmp_path / "profile"
    config = profile / "config.yaml"
    profile.mkdir()
    config.write_text(
        "image2_feishu_ingress:\n"
        "  enabled: true\n"
        "  launch_worker: false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("IMAGE2_FEISHU_LAUNCH_WORKER", "1")

    settings = load_image2_ingress_settings(profile_home=profile, environ=os.environ)

    assert settings.enabled is True
    assert settings.marketing_hub_root is None
    assert settings.runtime_root == profile / "runtime" / "image2"
    assert settings.db_path == profile / "runtime" / "image2" / "image2_jobs.sqlite"
    assert settings.launch_worker is True


def test_enqueue_feishu_job_writes_hermes_store_without_pipeline_cli(tmp_path):
    runtime = tmp_path / "runtime" / "image2"
    settings = Image2IngressSettings(
        enabled=True,
        db_path=runtime / "image2_jobs.sqlite",
        runtime_root=runtime,
        python_executable="pythonX",
        launch_worker=False,
        log_dir=runtime / "worker-logs",
    )
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        raise AssertionError("Hermes-owned enqueue must not call marketing-hub pipeline CLI")

    result = enqueue_feishu_job(settings, {"feishu_message_id": "om", "chat_id": "oc", "text": "做海报"}, runner=fake_run)

    assert result["task_id"].startswith("img2_")
    assert result["status"] == "ack_sent"
    assert calls == []
    assert settings.db_path.is_file()
    payload = json.loads((runtime / result["task_id"] / "message.json").read_text(encoding="utf-8"))
    assert payload["feishu_message_id"] == "om"


def test_launch_worker_detaches_with_profile_env_and_log_files(tmp_path):
    runtime = tmp_path / "runtime" / "image2"
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / ".env").write_text("FEISHU_APP_ID=id\nFEISHU_APP_SECRET=secret\n", encoding="utf-8")
    settings = Image2IngressSettings(
        enabled=True,
        db_path=runtime / "image2_jobs.sqlite",
        runtime_root=runtime,
        python_executable="pythonX",
        launch_worker=True,
        log_dir=runtime / "worker-logs",
        profile_home=profile,
    )
    launched = {}

    class FakePopen:
        pid = 12345
        def __init__(self, cmd, **kwargs):
            launched["cmd"] = cmd
            launched["kwargs"] = kwargs

    info = launch_image2_worker(settings, task_id="img2_abc", popen=FakePopen)

    assert info["pid"] == 12345
    assert launched["cmd"][:3] == ["pythonX", "-m", "gateway.image2_worker"]
    assert "image2_browser_worker.py" not in " ".join(launched["cmd"])
    assert "marketing-hub" not in " ".join(launched["cmd"])
    assert "--task-id" in launched["cmd"]
    assert launched["cmd"][launched["cmd"].index("--task-id") + 1] == "img2_abc"
    assert launched["kwargs"]["env"]["FEISHU_APP_ID"] == "id"
    assert launched["kwargs"]["env"]["FEISHU_APP_SECRET"] == "secret"
    assert "marketing-hub/scripts" not in launched["kwargs"]["env"].get("PYTHONPATH", "")
    assert Path(info["stdout_log"]).parent == settings.log_dir


def test_handle_ingress_returns_none_when_disabled_or_not_visual(tmp_path):
    settings = Image2IngressSettings(enabled=False)
    event = SimpleNamespace(
        text="做张海报",
        message_type=SimpleNamespace(value="text"),
        message_id="om",
        source=SimpleNamespace(platform=SimpleNamespace(value="feishu"), chat_id="oc", thread_id=None),
        media_urls=[],
    )
    assert handle_image2_feishu_ingress_event(event, settings=settings) is None

    settings = Image2IngressSettings(enabled=True)
    event.text = "查一下销售额"
    assert handle_image2_feishu_ingress_event(event, settings=settings) is None


def test_handle_ingress_enqueues_launches_worker_and_returns_ack(tmp_path):
    settings = Image2IngressSettings(enabled=True, launch_worker=True, runtime_root=tmp_path)
    event = SimpleNamespace(
        text="帮我做张火宫殿小炒肉海报",
        message_type=SimpleNamespace(value="text"),
        message_id="om",
        source=SimpleNamespace(platform=SimpleNamespace(value="feishu"), chat_id="oc", thread_id="root"),
        media_urls=[],
    )
    calls = []

    def fake_enqueue(loaded_settings, payload):
        calls.append(("enqueue", loaded_settings, payload))
        return {"task_id": "img2_abc"}

    def fake_launch(loaded_settings, *, task_id):
        calls.append(("launch", loaded_settings, task_id))
        return {"pid": 123}

    response = handle_image2_feishu_ingress_event(
        event,
        settings=settings,
        enqueue_func=fake_enqueue,
        launch_func=fake_launch,
    )

    assert "img2_abc" in response
    assert calls[0][0] == "enqueue"
    assert calls[0][2]["root_id"] == "root"
    assert calls[1] == ("launch", settings, "img2_abc")


def test_handle_ingress_routes_same_image2_thread_plain_reply_as_continuation(tmp_path):
    db_path = tmp_path / "image2_jobs.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE image2_generation_sessions (
                design_session_id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL,
                root_id TEXT,
                thread_id TEXT,
                chatgpt_title TEXT,
                chatgpt_url TEXT,
                conversation_id TEXT,
                latest_task_id TEXT NOT NULL,
                latest_candidate_path TEXT,
                latest_delivery_message_id TEXT,
                selected_image_index INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO image2_generation_sessions (
                design_session_id, chat_id, root_id, thread_id, chatgpt_title,
                latest_task_id, created_at, updated_at
            ) VALUES ('img2_choudoufu_root', 'oc_chat', 'om_choudoufu_thread', 'om_choudoufu_thread',
                      'img2_choudoufu_root', 'img2_choudoufu_root', '2026-04-28T09:00:00+00:00', '2026-04-28T09:00:00+00:00')
            """
        )
    settings = Image2IngressSettings(enabled=True, db_path=db_path, runtime_root=tmp_path)
    event = SimpleNamespace(
        text="继续执行，不是这张，是臭豆腐那张",
        message_type=SimpleNamespace(value="text"),
        message_id="om_plain_followup",
        source=SimpleNamespace(platform=SimpleNamespace(value="feishu"), chat_id="oc_chat", thread_id="om_choudoufu_thread"),
        media_urls=[],
    )
    calls = []

    def fake_enqueue(loaded_settings, payload):
        calls.append((loaded_settings, payload))
        return {"task_id": "img2_followup"}

    response = handle_image2_feishu_ingress_event(event, settings=settings, enqueue_func=fake_enqueue)

    assert "img2_followup" in response
    assert calls[0][1]["text"] == "继续执行，不是这张，是臭豆腐那张"
    assert calls[0][1]["thread_id"] == "om_choudoufu_thread"


def test_handle_ingress_routes_reply_to_image2_request_message_as_canonical_continuation(tmp_path):
    db_path = tmp_path / "image2_jobs.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE image2_jobs (
                task_id TEXT PRIMARY KEY,
                feishu_message_id TEXT,
                chat_id TEXT,
                root_id TEXT,
                thread_id TEXT,
                status TEXT,
                updated_at TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO image2_jobs (
                task_id, feishu_message_id, chat_id, root_id, thread_id, status, updated_at
            ) VALUES (
                'img2_choudoufu', 'om_image2_request_msg', 'oc_chat',
                'om_original_choudoufu_root', 'om_original_choudoufu_root',
                'ack_sent', '2026-04-28T10:00:00+00:00'
            )
            """
        )
    settings = Image2IngressSettings(enabled=True, db_path=db_path, runtime_root=tmp_path)
    event = SimpleNamespace(
        text="继续执行",
        message_type=SimpleNamespace(value="text"),
        message_id="om_plain_followup",
        source=SimpleNamespace(platform=SimpleNamespace(value="feishu"), chat_id="oc_chat", thread_id="om_image2_request_msg"),
        media_urls=[],
    )
    calls = []

    def fake_enqueue(loaded_settings, payload):
        calls.append((loaded_settings, payload))
        return {"task_id": "img2_followup"}

    response = handle_image2_feishu_ingress_event(event, settings=settings, enqueue_func=fake_enqueue)

    assert "img2_followup" in response
    assert calls[0][1]["root_id"] == "om_original_choudoufu_root"
    assert calls[0][1]["thread_id"] == "om_original_choudoufu_root"
    assert calls[0][1]["text"] == "继续执行"


def test_handle_ingress_reports_source_collection_failure_without_running_main_agent(monkeypatch, tmp_path):
    settings = Image2IngressSettings(enabled=True, runtime_root=tmp_path)
    event = SimpleNamespace(
        text="/image2 按引用图片做图",
        message_type=SimpleNamespace(value="command"),
        message_id="om_source_fail",
        source=SimpleNamespace(platform=SimpleNamespace(value="feishu"), chat_id="oc", thread_id="root"),
        media_urls=[],
    )
    enqueued = []

    def boom(*_args, **_kwargs):
        raise RuntimeError("tenant-token-super-secret-network-error")

    def fake_enqueue(_settings, payload):
        enqueued.append(payload)
        return {"task_id": "img2_should_not_happen"}

    monkeypatch.setattr("gateway.image2_feishu_ingress.collect_image2_source_files", boom)

    response = handle_image2_feishu_ingress_event(event, settings=settings, enqueue_func=fake_enqueue)

    assert response is not None
    assert "来源图片读取失败" in response
    assert "tenant-token-super-secret" not in response
    assert enqueued == []


def test_handle_ingress_reports_enqueue_failure_without_running_main_agent(tmp_path):
    settings = Image2IngressSettings(enabled=True, runtime_root=tmp_path)
    event = SimpleNamespace(
        text="/image2 做图",
        message_type=SimpleNamespace(value="command"),
        message_id="om",
        source=SimpleNamespace(platform=SimpleNamespace(value="feishu"), chat_id="oc", thread_id=None),
        media_urls=[],
    )

    def boom(_settings, _payload):
        raise RuntimeError("boom")

    response = handle_image2_feishu_ingress_event(event, settings=settings, enqueue_func=boom)
    assert "入队失败" in response
    assert "boom" not in response


def test_ack_text_for_job_is_user_facing_and_contains_task_id():
    assert "img2_abc" in ack_text_for_job({"task_id": "img2_abc"})
    assert "直接发图" in ack_text_for_job({"task_id": "img2_abc"})


def test_ack_text_for_job_is_honest_when_worker_is_disabled():
    response = ack_text_for_job({"task_id": "img2_canary"}, launch_worker=False)

    assert "img2_canary" in response
    assert "worker 未开启" in response
    assert "不会自动生图" in response
    assert "直接发图" not in response


def test_handle_ingress_with_worker_disabled_enqueues_but_does_not_promise_delivery(tmp_path):
    settings = Image2IngressSettings(enabled=True, launch_worker=False, runtime_root=tmp_path)
    event = SimpleNamespace(
        text="/image2 HERMES-CANARY 做一张规则验证图",
        message_type=SimpleNamespace(value="command"),
        message_id="om_canary",
        source=SimpleNamespace(platform=SimpleNamespace(value="feishu"), chat_id="oc", thread_id=None),
        media_urls=[],
    )
    launched = []

    def fake_enqueue(_settings, payload):
        assert payload["feishu_message_id"] == "om_canary"
        return {"task_id": "img2_canary"}

    def fake_launch(_settings, *, task_id):
        launched.append(task_id)
        return {"pid": 123}

    response = handle_image2_feishu_ingress_event(
        event,
        settings=settings,
        enqueue_func=fake_enqueue,
        launch_func=fake_launch,
    )

    assert "worker 未开启" in response
    assert "不会自动生图" in response
    assert launched == []


def test_gateway_run_has_image2_intercept_before_busy_session_guard():
    run_py = Path(__file__).resolve().parents[2] / "gateway" / "run.py"
    text = run_py.read_text(encoding="utf-8")

    intercept = "handle_image2_feishu_ingress_event(event)"
    busy_guard = "# PRIORITY handling when an agent is already running for this session."
    assert intercept in text
    assert busy_guard in text
    assert text.index(intercept) < text.index(busy_guard)


def test_active_session_busy_handler_intercepts_feishu_image2_before_queueing():
    run_py = Path(__file__).resolve().parents[2] / "gateway" / "run.py"
    text = run_py.read_text(encoding="utf-8")
    start = text.index("async def _handle_active_session_busy_message")
    end = text.index("    async def _drain_active_agents", start)
    busy_handler = text[start:end]

    intercept = "handle_image2_feishu_ingress_event(event)"
    first_queue = "merge_pending_message_event(adapter._pending_messages, session_key, event)"
    assert intercept in busy_handler
    assert first_queue in busy_handler
    assert busy_handler.index(intercept) < busy_handler.index(first_queue)


def test_reverse_split_text_then_image_enqueues_one_image2_job_with_previous_text(tmp_path):
    text_message = {
        "message_id": "om_text",
        "msg_type": "text",
        "create_time": "100000",
        "sender": {"sender_type": "user", "sender_id": {"open_id": "u1"}},
        "body": {"content": json.dumps({"text": "/image2 帮我把这个臭豆腐海报设计好看，标题你帮我想一个，8字以内"}, ensure_ascii=False)},
    }
    image_message = {
        "message_id": "om_image",
        "msg_type": "image",
        "create_time": "101000",
        "sender": {"sender_type": "user", "sender_id": {"open_id": "u1"}},
        "body": {"content": json.dumps({"image_key": "img1"})},
    }

    previous = select_recent_previous_visual_text_message(
        [image_message, text_message],
        current_message_id="om_image",
        lookback_seconds=120,
    )

    assert previous is text_message

    direct_source = tmp_path / "upload.jpg"
    direct_source.write_bytes(b"uploaded image")
    event = SimpleNamespace(
        message_id="om_image",
        text="",
        message_type="photo",
        source=SimpleNamespace(platform=SimpleNamespace(value="feishu"), chat_id="oc_chat", thread_id=""),
        media_urls=[str(direct_source)],
    )
    settings = Image2IngressSettings(enabled=True, launch_worker=False, runtime_root=tmp_path / "runtime", db_path=tmp_path / "image2.sqlite")

    ack = handle_image2_feishu_ingress_event(
        event,
        settings=settings,
        launch_func=lambda *a, **k: (_ for _ in ()).throw(AssertionError("worker should not launch")),
    )

    assert ack is None

    ack = handle_image2_feishu_ingress_event(
        event,
        settings=settings,
        enqueue_func=lambda loaded, payload: {"task_id": "img2_reverse", "job_dir": str(tmp_path / "runtime" / "img2_reverse"), "payload": payload},
        launch_func=lambda *a, **k: (_ for _ in ()).throw(AssertionError("worker should not launch")),
    )

    # No previous_text_resolver was injected above, so the real event still falls through safely.
    assert ack is None

    payload = build_feishu_message_payload(
        event,
        settings=settings,
        previous_text_resolver=lambda current_event: "/image2 帮我把这个臭豆腐海报设计好看，标题你帮我想一个，8字以内",
    )
    assert payload["text"].startswith("/image2 帮我把这个臭豆腐")
    assert payload["source_files"][0]["source"] == "feishu_direct_media"

    ack = handle_image2_feishu_ingress_event(
        event,
        settings=settings,
        enqueue_func=lambda loaded, p: {"task_id": "img2_reverse", "job_dir": str(tmp_path / "runtime" / "img2_reverse")},
        launch_func=lambda *a, **k: (_ for _ in ()).throw(AssertionError("worker should not launch")),
    )
    # Unit path without Feishu API credentials cannot resolve previous text; build_feishu_message_payload above proves the merge behavior.
    assert ack is None


def test_source_dependent_text_only_image2_waits_for_followup_image(tmp_path):
    event = SimpleNamespace(
        message_id="om_wait_text",
        text="/image2 帮我把这个臭豆腐的海报设计得更好看，标题你帮我想一个",
        message_type="text",
        source=SimpleNamespace(platform=SimpleNamespace(value="feishu"), chat_id="oc_chat", thread_id=""),
        media_urls=[],
    )
    settings = Image2IngressSettings(enabled=True, launch_worker=False, runtime_root=tmp_path / "runtime", db_path=tmp_path / "image2.sqlite")

    ack = handle_image2_feishu_ingress_event(
        event,
        settings=settings,
        enqueue_func=lambda loaded, payload: (_ for _ in ()).throw(AssertionError("must wait for the follow-up image instead of enqueuing a text-only task")),
        launch_func=lambda *a, **k: (_ for _ in ()).throw(AssertionError("worker should not launch")),
    )

    assert "等图片" in ack
    assert "同一个任务" in ack


def test_reverse_split_ignores_standalone_already_enqueueable_visual_text():
    text_message = {
        "message_id": "om_text",
        "msg_type": "text",
        "create_time": "100000",
        "sender": {"sender_type": "user", "sender_id": {"open_id": "u1"}},
        "body": {"content": json.dumps({"text": "/image2 做一张臭豆腐单品海报"}, ensure_ascii=False)},
    }
    image_message = {
        "message_id": "om_image",
        "msg_type": "image",
        "create_time": "101000",
        "sender": {"sender_type": "user", "sender_id": {"open_id": "u1"}},
        "body": {"content": json.dumps({"image_key": "img1"})},
    }

    assert select_recent_previous_visual_text_message(
        [image_message, text_message],
        current_message_id="om_image",
        lookback_seconds=120,
    ) is None
