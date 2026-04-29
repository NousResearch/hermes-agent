from __future__ import annotations

import json
import sqlite3
import subprocess
import time
from pathlib import Path

from gateway.image2_candidate_gate import sha256_file
from gateway.image2_feishu_delivery import FeishuImageClient
from gateway.image2_generation import run_opencli_generation
from gateway.image2_store import Image2JobStore
from gateway.image2_worker import run_worker


def _enqueue(runtime: Path, payload: dict[str, object]) -> dict[str, object]:
    return Image2JobStore(db_path=runtime / "image2_jobs.sqlite", runtime_root=runtime).enqueue_feishu(payload)


def _row(db_path: Path, task_id: str) -> dict[str, object]:
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM image2_jobs WHERE task_id = ?", (task_id,)).fetchone()
    assert row is not None
    return dict(row)


def _write_png(path: Path, payload: bytes = b"not-a-real-png-but-test-image") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    future = time.time() + 30
    path.touch()
    return path


def test_worker_generates_reviews_delivers_and_marks_readback_verified(tmp_path):
    runtime = tmp_path / "runtime"
    db_path = runtime / "image2_jobs.sqlite"
    job = _enqueue(
        runtime,
        {
            "feishu_message_id": "msg-full-loop",
            "chat_id": "chat-full-loop",
            "root_id": "root-full-loop",
            "thread_id": "root-full-loop",
            "text": "生成一张臭豆腐单品海报，必须是臭豆腐，不要Logo文字",
        },
    )
    job_dir = Path(str(job["job_dir"]))
    browser_state = job_dir / "browser_state.json"
    browser_state.write_text(json.dumps({"cdp_reachable": True, "active_url": "https://chatgpt.com/images", "title": "ChatGPT Images"}), encoding="utf-8")
    calls: dict[str, object] = {}

    def fake_generator(*, job_dir: Path, prompt_text: str, environ: dict[str, str], source_files: list[object]) -> dict[str, object]:
        calls["generator_prompt"] = prompt_text
        candidate = _write_png(job_dir / "candidates" / "generated.png")
        return {"status": "saved", "files": [str(candidate)], "link": "https://chatgpt.com/c/test-full-loop"}

    def fake_reviewer(*, job_dir: Path, candidate: dict[str, object], prompt_text: str, environ: dict[str, str]) -> dict[str, object]:
        calls["review_candidate_sha"] = candidate["sha256"]
        return {"decision": "PASS", "issues": [], "reviewer": "unit-test"}

    def fake_delivery(*, image_path: Path, chat_id: str, reply_to: str, candidate_sha256: str, environ: dict[str, str]) -> dict[str, object]:
        calls["delivery"] = {"image_path": str(image_path), "chat_id": chat_id, "reply_to": reply_to, "candidate_sha256": candidate_sha256}
        return {"verified": True, "message_id": "om_full_loop", "image_key": "img_full_loop", "readback_msg_type": "image"}

    result = run_worker(
        db_path=db_path,
        runtime_root=runtime,
        task_id=str(job["task_id"]),
        worker_id="worker-full-loop-test",
        environ={
            "IMAGE2_WORKER_LIVE_ENABLED": "1",
            "IMAGE2_BROWSER_STATE_JSON": str(browser_state),
            "IMAGE2_REVIEWER_PROVIDER": "unit-test",
            "FEISHU_APP_ID": "present",
            "FEISHU_APP_SECRET": "present",
            "OPENCLI_CHROME_CDP_GUIDANCE": "0",
        },
        generator=fake_generator,
        reviewer=fake_reviewer,
        delivery_sender=fake_delivery,
    )

    assert result["status"] == "readback_verified"
    assert result["delivery_readback"]["readback_msg_type"] == "image"
    assert result["candidate_gate"]["status"] == "pass"
    assert result["review_gate"]["status"] == "pass"
    assert result["delivery_contract"]["status"] == "ready_to_send"
    assert calls["delivery"]["chat_id"] == "chat-full-loop"
    assert calls["delivery"]["reply_to"] == "root-full-loop"
    assert _row(db_path, str(job["task_id"]))["status"] == "readback_verified"
    assert _row(db_path, str(job["task_id"]))["last_error"] in (None, "")
    assert (job_dir / "generation_result.json").is_file()
    assert (job_dir / "delivery_readback.json").is_file()


def test_worker_fails_closed_when_generator_saves_no_candidate(tmp_path):
    runtime = tmp_path / "runtime"
    db_path = runtime / "image2_jobs.sqlite"
    job = _enqueue(runtime, {"feishu_message_id": "msg-no-candidate", "chat_id": "chat", "root_id": "root", "thread_id": "root", "text": "生成臭豆腐海报"})
    job_dir = Path(str(job["job_dir"]))
    browser_state = job_dir / "browser_state.json"
    browser_state.write_text(json.dumps({"cdp_reachable": True, "active_url": "https://chatgpt.com/images", "title": "ChatGPT Images"}), encoding="utf-8")

    result = run_worker(
        db_path=db_path,
        runtime_root=runtime,
        task_id=str(job["task_id"]),
        worker_id="worker-no-candidate-test",
        environ={"IMAGE2_WORKER_LIVE_ENABLED": "1", "IMAGE2_BROWSER_STATE_JSON": str(browser_state), "IMAGE2_REVIEWER_PROVIDER": "unit-test", "FEISHU_APP_ID": "present", "FEISHU_APP_SECRET": "present", "OPENCLI_CHROME_CDP_GUIDANCE": "0"},
        generator=lambda **kwargs: {"status": "blocked", "files": [], "reason": "unit-no-file"},
        reviewer=lambda **kwargs: {"decision": "PASS", "issues": []},
        delivery_sender=lambda **kwargs: {"verified": True, "message_id": "should-not-send", "readback_msg_type": "image"},
    )

    assert result["status"] == "failed_final"
    assert result["reason"] == "candidate_gate_no_candidates"
    assert not (job_dir / "delivery_readback.json").exists()
    assert _row(db_path, str(job["task_id"]))["status"] == "failed_final"


class _Response:
    def __init__(self, payload: dict[str, object]):
        self._payload = payload
    def raise_for_status(self) -> None:
        return None
    def json(self) -> dict[str, object]:
        return self._payload


def test_feishu_image_client_replies_with_native_image_and_exact_readback(tmp_path):
    image = _write_png(tmp_path / "candidate.png")
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_post(url: str, **kwargs):
        calls.append((url, kwargs))
        if url.endswith("/auth/v3/tenant_access_token/internal"):
            return _Response({"code": 0, "tenant_access_token": "token-redacted"})
        if url.endswith("/im/v1/images"):
            return _Response({"code": 0, "data": {"image_key": "img_unit_key"}})
        if url.endswith("/im/v1/messages/root-message/reply"):
            body = kwargs["json"]
            assert body["msg_type"] == "image"
            assert body["reply_in_thread"] is True
            assert json.loads(body["content"])["image_key"] == "img_unit_key"
            return _Response({"code": 0, "data": {"message_id": "om_unit_reply"}})
        raise AssertionError(url)

    def fake_get(url: str, **kwargs):
        calls.append((url, kwargs))
        assert url.endswith("/im/v1/messages/om_unit_reply")
        return _Response({"code": 0, "data": {"items": [{"message_id": "om_unit_reply", "msg_type": "image", "body": {"content": json.dumps({"image_key": "img_unit_key"})}}]}})

    client = FeishuImageClient(app_id="app", app_secret="secret", http_post=fake_post, http_get=fake_get)
    result = client.send_image_and_verify(image, chat_id="chat-id", reply_to="root-message", candidate_sha256=sha256_file(image))

    assert result["verified"] is True
    assert result["message_id"] == "om_unit_reply"
    assert result["readback_msg_type"] == "image"
    assert result["candidate_sha256"] == sha256_file(image)


def test_opencli_generation_salvages_fresh_candidate_when_cli_stdout_is_empty_and_exit_nonzero(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"

    def fake_run(cmd, cwd, env, text, capture_output, timeout):
        candidate = Path(cwd) / "candidates" / "chatgpt_salvaged.png"
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_bytes(b"fresh-image-by-opencli-before-nonzero-exit")
        return subprocess.CompletedProcess(cmd, 75, stdout="", stderr="saved image but command timed out after browser completion")

    monkeypatch.setattr("gateway.image2_generation.subprocess.run", fake_run)
    result = run_opencli_generation(job_dir=job_dir, prompt_text="生成臭豆腐海报", environ={"IMAGE2_OPENCLI_TIMEOUT": "1"}, source_files=[])

    assert result["status"] == "saved_with_nonzero_exit"
    assert len(result["files"]) == 1
    assert Path(result["files"][0]).name == "chatgpt_salvaged.png"
    saved_result = json.loads((job_dir / "generation_result.json").read_text(encoding="utf-8"))
    assert saved_result["files"] == result["files"]



def test_worker_requires_delivery_preflight_before_generation(tmp_path):
    runtime = tmp_path / "runtime"
    db_path = runtime / "image2_jobs.sqlite"
    job = _enqueue(runtime, {"feishu_message_id": "msg-no-creds", "chat_id": "chat", "root_id": "root", "thread_id": "root", "text": "生成臭豆腐海报"})
    job_dir = Path(str(job["job_dir"]))
    browser_state = job_dir / "browser_state.json"
    browser_state.write_text(json.dumps({"cdp_reachable": True, "active_url": "https://chatgpt.com/images", "title": "ChatGPT Images"}), encoding="utf-8")
    calls = {"generator": 0, "reviewer": 0, "delivery": 0}

    result = run_worker(
        db_path=db_path,
        runtime_root=runtime,
        task_id=str(job["task_id"]),
        worker_id="worker-no-delivery-preflight",
        environ={"IMAGE2_WORKER_LIVE_ENABLED": "1", "IMAGE2_BROWSER_STATE_JSON": str(browser_state), "IMAGE2_REVIEWER_PROVIDER": "unit-test", "OPENCLI_CHROME_CDP_GUIDANCE": "0"},
        generator=lambda **kwargs: calls.__setitem__("generator", calls["generator"] + 1) or {"status": "saved"},
        reviewer=lambda **kwargs: calls.__setitem__("reviewer", calls["reviewer"] + 1) or {"decision": "PASS", "issues": []},
        delivery_sender=lambda **kwargs: calls.__setitem__("delivery", calls["delivery"] + 1) or {"verified": True},
    )

    assert result["status"] == "failed_final"
    assert result["reason"] == "delivery_preflight_missing"
    assert calls == {"generator": 0, "reviewer": 0, "delivery": 0}
    assert not (job_dir / "generation_result.json").exists()


def test_opencli_generation_env_does_not_forward_delivery_or_api_secrets(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    seen_env = {}

    def fake_run(cmd, cwd, env, text, capture_output, timeout):
        seen_env.update(env)
        candidate = Path(cwd) / "candidates" / "safe.png"
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_bytes(b"safe-image")
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"saved_files": [str(candidate)]}), stderr="")

    monkeypatch.setenv("FEISHU_APP_SECRET", "must-not-leak")
    monkeypatch.setenv("GOOGLE_API_KEY", "must-not-leak")
    monkeypatch.setattr("gateway.image2_generation.subprocess.run", fake_run)

    result = run_opencli_generation(
        job_dir=job_dir,
        prompt_text="生成臭豆腐海报",
        environ={"FEISHU_APP_ID": "app", "FEISHU_APP_SECRET": "secret", "OPENCLI_CDP_URL": "http://127.0.0.1:9222"},
        source_files=[],
    )

    assert result["status"] == "saved"
    assert "FEISHU_APP_ID" not in seen_env
    assert "FEISHU_APP_SECRET" not in seen_env
    assert "GOOGLE_API_KEY" not in seen_env
    assert seen_env["OPENCLI_CDP_URL"] == "http://127.0.0.1:9222"


def test_feishu_image_client_fails_when_candidate_sha_changes_before_upload(tmp_path):
    image = _write_png(tmp_path / "candidate.png", b"reviewed")
    reviewed_sha = sha256_file(image)
    image.write_bytes(b"tampered")
    client = FeishuImageClient(app_id="app", app_secret="secret", http_post=lambda *a, **k: _Response({"code": 0}), http_get=lambda *a, **k: _Response({"code": 0}))

    try:
        client.send_image_and_verify(image, chat_id="chat-id", reply_to="root-message", candidate_sha256=reviewed_sha)
    except Exception as exc:
        assert "candidate_sha256 mismatch" in str(exc)
    else:
        raise AssertionError("expected candidate_sha256 mismatch")


def test_feishu_image_client_requires_exact_readback_image_key(tmp_path):
    image = _write_png(tmp_path / "candidate.png")

    def fake_post(url: str, **kwargs):
        if url.endswith("/auth/v3/tenant_access_token/internal"):
            return _Response({"code": 0, "tenant_access_token": "token-redacted"})
        if url.endswith("/im/v1/images"):
            return _Response({"code": 0, "data": {"image_key": "img_unit_key"}})
        return _Response({"code": 0, "data": {"message_id": "om_unit_reply"}})

    def fake_get(url: str, **kwargs):
        return _Response({"code": 0, "data": {"item": {"message_id": "om_unit_reply", "msg_type": "image", "body": {"content": json.dumps({})}}}})

    client = FeishuImageClient(app_id="app", app_secret="secret", http_post=fake_post, http_get=fake_get)
    try:
        client.send_image_and_verify(image, chat_id="chat-id", reply_to="root-message", candidate_sha256=sha256_file(image))
    except Exception as exc:
        assert "read-back image_key missing" in str(exc)
    else:
        raise AssertionError("expected missing readback image_key failure")


def test_feishu_image_client_accepts_item_readback_shape(tmp_path):
    image = _write_png(tmp_path / "candidate.png")

    def fake_post(url: str, **kwargs):
        if url.endswith("/auth/v3/tenant_access_token/internal"):
            return _Response({"code": 0, "tenant_access_token": "token-redacted"})
        if url.endswith("/im/v1/images"):
            return _Response({"code": 0, "data": {"image_key": "img_unit_key"}})
        return _Response({"code": 0, "data": {"message_id": "om_unit_reply"}})

    def fake_get(url: str, **kwargs):
        return _Response({"code": 0, "data": {"item": {"message_id": "om_unit_reply", "msg_type": "image", "body": {"content": json.dumps({"image_key": "img_unit_key"})}}}})

    client = FeishuImageClient(app_id="app", app_secret="secret", http_post=fake_post, http_get=fake_get)
    result = client.send_image_and_verify(image, chat_id="chat-id", reply_to="root-message", candidate_sha256=sha256_file(image))
    assert result["verified"] is True
    assert result["image_key"] == "img_unit_key"


def test_opencli_generation_title_includes_task_id_and_subject(tmp_path, monkeypatch):
    job_dir = tmp_path / "img2_unique123"
    captured = {}

    def fake_run(cmd, cwd, env, text, capture_output, timeout):
        captured["cmd"] = cmd
        out_dir = Path(cmd[cmd.index("--op") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        candidate = out_dir / "generated.png"
        candidate.write_bytes(b"fresh image")
        rows = [{"status": "✅ saved", "file": str(candidate), "link": "https://chatgpt.com/c/unique"}]
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(rows), stderr="")

    monkeypatch.setattr("gateway.image2_generation.subprocess.run", fake_run)

    result = run_opencli_generation(
        job_dir=job_dir,
        prompt_text="主视觉对象：臭豆腐。\n文案上图要求：主标题「外酥里嫩臭豆腐」",
        environ={"IMAGE2_OPENCLI_TIMEOUT": "1"},
        source_files=[],
    )

    title = captured["cmd"][captured["cmd"].index("--title") + 1]
    assert "img2_unique123" in title
    assert "臭豆腐" in title
    assert title != "海报设计"
    assert result["title"] == title


def test_opencli_generation_title_does_not_trust_generic_env_title(tmp_path, monkeypatch):
    job_dir = tmp_path / "img2_titleenv"
    captured = {}

    def fake_run(cmd, cwd, env, text, capture_output, timeout):
        captured["cmd"] = cmd
        out_dir = Path(cmd[cmd.index("--op") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        candidate = out_dir / "generated.png"
        candidate.write_bytes(b"fresh image")
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps([{"status": "✅ saved", "file": str(candidate)}]), stderr="")

    monkeypatch.setattr("gateway.image2_generation.subprocess.run", fake_run)

    run_opencli_generation(
        job_dir=job_dir,
        prompt_text="主视觉对象：臭豆腐。",
        environ={"IMAGE2_OPENCLI_TIMEOUT": "1", "IMAGE2_OPENCLI_TITLE": "海报设计"},
        source_files=[],
    )

    title = captured["cmd"][captured["cmd"].index("--title") + 1]
    assert title != "海报设计"
    assert "img2_titleenv" in title
    assert "臭豆腐" in title
