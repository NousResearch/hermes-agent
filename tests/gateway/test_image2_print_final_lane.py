
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from gateway.image2_feishu_ingress import Image2IngressSettings, handle_image2_feishu_ingress_event
from gateway.image2_feishu_delivery import FeishuImageClient
from gateway.image2_print import parse_print_spec, should_handle_print_request
from gateway.image2_store import Image2JobStore
from gateway.image2_worker import run_worker


def _row(db_path: Path, task_id: str) -> dict[str, object]:
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM image2_jobs WHERE task_id = ?", (task_id,)).fetchone()
    assert row is not None
    return dict(row)


def _event(text: str, *, message_id: str = "om_print", root_id: str = "om_root"):
    return SimpleNamespace(
        message_id=message_id,
        text=text,
        message_type="text",
        source=SimpleNamespace(platform=SimpleNamespace(value="feishu"), chat_id="oc_chat", thread_id=root_id, root_id=root_id),
        media_urls=[],
    )


def test_parse_print_spec_requires_size_and_uses_large_format_default_dpi():
    assert should_handle_print_request("定稿，出印刷版，尺寸 100×150cm")
    assert not should_handle_print_request("/image2 做一张 PSD 风格臭豆腐海报")
    parsed = parse_print_spec("定稿，出印刷版，尺寸 100×150cm")

    assert parsed["status"] == "ok"
    assert parsed["width_mm"] == 1000
    assert parsed["height_mm"] == 1500
    assert parsed["dpi"] == 150
    assert parsed["target_width_px"] == 5906
    assert parsed["target_height_px"] == 8858
    assert parsed["output_psd_type"] == "flat_single_layer"

    missing = parse_print_spec("定稿，出印刷版")
    assert missing["status"] == "need_clarification"
    assert missing["need_clarification"] == "size_required"


def test_parse_print_spec_a3_and_explicit_200dpi():
    a3 = parse_print_spec("定稿，出印刷版，尺寸 A3")
    assert a3["width_mm"] == 297
    assert a3["height_mm"] == 420
    assert a3["dpi"] == 300
    assert a3["target_width_px"] == 3508
    assert a3["target_height_px"] == 4961

    large = parse_print_spec("定稿，出印刷版，尺寸 100×133cm，200dpi")
    assert large["width_mm"] == 1000
    assert large["height_mm"] == 1330
    assert large["dpi"] == 200
    assert large["target_width_px"] == 7874
    assert large["target_height_px"] == 10472


def test_print_request_without_size_asks_for_size_and_does_not_enqueue(tmp_path):
    settings = Image2IngressSettings(enabled=True, runtime_root=tmp_path / "runtime", db_path=tmp_path / "runtime" / "image2_jobs.sqlite", launch_worker=True)
    calls = {"enqueue": 0, "launch": 0}

    ack = handle_image2_feishu_ingress_event(
        _event("定稿，出印刷版"),
        settings=settings,
        enqueue_func=lambda *_args, **_kwargs: calls.__setitem__("enqueue", calls["enqueue"] + 1),
        launch_func=lambda *_args, **_kwargs: calls.__setitem__("launch", calls["launch"] + 1),
    )

    assert "需要尺寸" in ack
    assert calls == {"enqueue": 0, "launch": 0}
    assert not settings.db_path.exists()


def test_print_request_enqueues_from_latest_verified_preview_in_same_thread(tmp_path):
    runtime = tmp_path / "runtime"
    db_path = runtime / "image2_jobs.sqlite"
    store = Image2JobStore(db_path=db_path, runtime_root=runtime)
    preview = store.enqueue_feishu({"feishu_message_id": "om_preview", "chat_id": "oc_chat", "root_id": "om_root", "thread_id": "om_root", "text": "/image2 做一张臭豆腐海报"})
    preview_dir = Path(str(preview["job_dir"]))
    approved = preview_dir / "candidates" / "approved.png"
    approved.parent.mkdir(parents=True, exist_ok=True)
    approved.write_bytes(b"approved-preview")
    approved_sha = hashlib.sha256(approved.read_bytes()).hexdigest()
    (preview_dir / "worker_result.json").write_text(json.dumps({"delivery_contract": {"status": "ready_to_send", "image_path": str(approved), "image_sha256": approved_sha}, "delivery_readback": {"verified": True, "message_id": "om_img", "readback_msg_type": "image"}}, ensure_ascii=False), encoding="utf-8")
    store.mark_readback_verified(task_id=str(preview["task_id"]), worker_id="unit")
    settings = Image2IngressSettings(enabled=True, runtime_root=runtime, db_path=db_path, launch_worker=True)
    launched: list[str] = []

    ack = handle_image2_feishu_ingress_event(
        _event("定稿，出印刷版，尺寸 100×150cm", message_id="om_print"),
        settings=settings,
        launch_func=lambda _settings, *, task_id: launched.append(task_id) or {"pid": 123},
    )

    assert "印刷定稿队列" in ack
    assert launched
    print_job = _row(db_path, launched[0])
    payload = json.loads(str(print_job["payload_json"]))
    assert payload["print_request"]["approved_task_id"] == preview["task_id"]
    assert payload["print_request"]["approved_image_path"] == str(approved)
    assert payload["print_request"]["approved_image_sha256"] == approved_sha
    assert payload["print_request"]["spec"]["dpi"] == 150
    assert payload["print_request"]["spec"]["output_psd_type"] == "flat_single_layer"


def test_print_request_without_thread_does_not_pick_random_chat_preview(tmp_path):
    runtime = tmp_path / "runtime"
    db_path = runtime / "image2_jobs.sqlite"
    store = Image2JobStore(db_path=db_path, runtime_root=runtime)
    preview = store.enqueue_feishu({"feishu_message_id": "om_preview", "chat_id": "oc_chat", "root_id": "om_other", "thread_id": "om_other", "text": "/image2 另一张海报"})
    preview_dir = Path(str(preview["job_dir"]))
    approved = preview_dir / "candidates" / "approved.png"
    approved.parent.mkdir(parents=True, exist_ok=True)
    approved.write_bytes(b"other-preview")
    approved_sha = hashlib.sha256(approved.read_bytes()).hexdigest()
    (preview_dir / "worker_result.json").write_text(json.dumps({"delivery_contract": {"status": "ready_to_send", "image_path": str(approved), "image_sha256": approved_sha}, "delivery_readback": {"verified": True, "message_id": "om_img", "readback_msg_type": "image"}}, ensure_ascii=False), encoding="utf-8")
    store.mark_readback_verified(task_id=str(preview["task_id"]), worker_id="unit")
    settings = Image2IngressSettings(enabled=True, runtime_root=runtime, db_path=db_path, launch_worker=True)

    ack = handle_image2_feishu_ingress_event(
        _event("定稿，出印刷版，尺寸 100×150cm", message_id="om_print", root_id=""),
        settings=settings,
        launch_func=lambda *_args, **_kwargs: {"pid": 123},
    )

    assert "没有找到同一飞书话题" in ack


def test_print_request_rejects_tampered_or_not_exactly_verified_preview(tmp_path):
    runtime = tmp_path / "runtime"
    db_path = runtime / "image2_jobs.sqlite"
    store = Image2JobStore(db_path=db_path, runtime_root=runtime)
    preview = store.enqueue_feishu({"feishu_message_id": "om_preview", "chat_id": "oc_chat", "root_id": "om_root", "thread_id": "om_root", "text": "/image2 做一张海报"})
    preview_dir = Path(str(preview["job_dir"]))
    approved = preview_dir / "candidates" / "approved.png"
    approved.parent.mkdir(parents=True, exist_ok=True)
    approved.write_bytes(b"original")
    original_sha = hashlib.sha256(approved.read_bytes()).hexdigest()
    approved.write_bytes(b"tampered")
    (preview_dir / "worker_result.json").write_text(json.dumps({"delivery_contract": {"status": "ready_to_send", "image_path": str(approved), "image_sha256": original_sha}, "delivery_readback": {"verified": True, "message_id": "om_img", "readback_msg_type": "image"}}, ensure_ascii=False), encoding="utf-8")
    store.mark_readback_verified(task_id=str(preview["task_id"]), worker_id="unit")
    settings = Image2IngressSettings(enabled=True, runtime_root=runtime, db_path=db_path, launch_worker=True)

    ack = handle_image2_feishu_ingress_event(_event("定稿，出印刷版，尺寸 100×150cm"), settings=settings)

    assert "没有找到同一飞书话题" in ack


def test_worker_print_lane_rejects_approved_preview_sha_mismatch(tmp_path):
    runtime = tmp_path / "runtime"
    db_path = runtime / "image2_jobs.sqlite"
    approved = tmp_path / "approved.png"
    approved.write_bytes(b"changed")
    spec = parse_print_spec("定稿，出印刷版，尺寸 100×150cm")
    job = Image2JobStore(db_path=db_path, runtime_root=runtime).enqueue_feishu({
        "feishu_message_id": "om_print",
        "chat_id": "oc_chat",
        "root_id": "om_root",
        "thread_id": "om_root",
        "text": "定稿，出印刷版，尺寸 100×150cm",
        "print_request": {"approved_task_id": "img2_preview", "approved_image_path": str(approved), "approved_image_sha256": "0" * 64, "spec": spec},
    })

    result = run_worker(
        db_path=db_path,
        runtime_root=runtime,
        task_id=str(job["task_id"]),
        worker_id="print-worker",
        environ={"IMAGE2_WORKER_LIVE_ENABLED": "1", "FEISHU_APP_ID": "present", "FEISHU_APP_SECRET": "present"},
    )

    assert result["status"] == "failed_final"
    assert result["reason"] == "print_approved_source_sha_mismatch"


def test_worker_print_lane_packages_psd_pdf_and_sends_documents(tmp_path):
    runtime = tmp_path / "runtime"
    db_path = runtime / "image2_jobs.sqlite"
    approved = tmp_path / "approved.png"
    approved.write_bytes(b"approved")
    approved_sha = hashlib.sha256(approved.read_bytes()).hexdigest()
    spec = parse_print_spec("定稿，出印刷版，尺寸 100×150cm")
    job = Image2JobStore(db_path=db_path, runtime_root=runtime).enqueue_feishu({
        "feishu_message_id": "om_print",
        "chat_id": "oc_chat",
        "root_id": "om_root",
        "thread_id": "om_thread",
        "text": "定稿，出印刷版，尺寸 100×150cm",
        "print_request": {"approved_task_id": "img2_preview", "approved_image_path": str(approved), "approved_image_sha256": approved_sha, "spec": spec},
    })
    calls: dict[str, object] = {}

    def fake_print_packager(*, job_dir: Path, approved_image_path: Path, spec: dict[str, object], environ: dict[str, str]):
        calls["packager"] = {"approved": str(approved_image_path), "dpi": spec["dpi"]}
        psd = job_dir / "print" / "psd" / "unit_flat.psd"
        pdf = job_dir / "print" / "proof" / "unit_proof.pdf"
        preview = job_dir / "print" / "proof" / "preview_1200.png"
        for path, payload in [(psd, b"psd"), (pdf, b"pdf"), (preview, b"png")]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        return {"status": "pass", "psd_path": str(psd), "pdf_path": str(pdf), "preview_path": str(preview), "approved_sha256": approved_sha, "target_width_px": 5906, "target_height_px": 8858, "dpi": 150}

    def fake_file_delivery(*, files: list[dict[str, object]], chat_id: str, reply_to: str, environ: dict[str, str]):
        calls["delivery"] = {"files": files, "chat_id": chat_id, "reply_to": reply_to}
        return {"verified": True, "readbacks": [{"message_id": "om_psd", "readback_msg_type": "file"}, {"message_id": "om_pdf", "readback_msg_type": "file"}]}

    result = run_worker(
        db_path=db_path,
        runtime_root=runtime,
        task_id=str(job["task_id"]),
        worker_id="print-worker",
        environ={"IMAGE2_WORKER_LIVE_ENABLED": "1", "FEISHU_APP_ID": "present", "FEISHU_APP_SECRET": "present"},
        print_packager=fake_print_packager,
        file_delivery_sender=fake_file_delivery,
    )

    assert result["status"] == "readback_verified"
    assert result["reason"] == "Feishu print files read-back verified"
    assert calls["delivery"]["chat_id"] == "oc_chat"
    assert calls["delivery"]["reply_to"] == "om_thread"
    assert _row(db_path, str(job["task_id"]))["status"] == "readback_verified"
    assert (Path(str(job["job_dir"])) / "print" / "reports" / "delivery_report.json").is_file()


class _Response:
    def __init__(self, payload: dict[str, object]):
        self._payload = payload
    def raise_for_status(self) -> None:
        return None
    def json(self) -> dict[str, object]:
        return self._payload


def test_feishu_file_client_sends_document_file_and_exact_readback(tmp_path):
    doc = tmp_path / "final_flat.psd"
    doc.write_bytes(b"fake psd bytes")
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_post(url: str, **kwargs):
        calls.append((url, kwargs))
        if url.endswith("/auth/v3/tenant_access_token/internal"):
            return _Response({"code": 0, "tenant_access_token": "token-redacted"})
        if url.endswith("/im/v1/files"):
            assert kwargs["data"]["file_type"] == "stream"
            assert kwargs["data"]["file_name"] == "final_flat.psd"
            return _Response({"code": 0, "data": {"file_key": "file_unit_key"}})
        if url.endswith("/im/v1/messages/om_root/reply"):
            body = kwargs["json"]
            assert body["msg_type"] == "file"
            assert json.loads(body["content"])["file_key"] == "file_unit_key"
            return _Response({"code": 0, "data": {"message_id": "om_file_reply"}})
        raise AssertionError(url)

    def fake_get(url: str, **kwargs):
        calls.append((url, kwargs))
        assert url.endswith("/im/v1/messages/om_file_reply")
        return _Response({"code": 0, "data": {"item": {"message_id": "om_file_reply", "msg_type": "file", "body": {"content": json.dumps({"file_key": "file_unit_key"})}}}})

    client = FeishuImageClient(app_id="app", app_secret="secret", http_post=fake_post, http_get=fake_get)
    result = client.send_file_and_verify(doc, chat_id="oc_chat", reply_to="om_root")

    assert result["verified"] is True
    assert result["message_id"] == "om_file_reply"
    assert result["readback_msg_type"] == "file"
    assert result["file_key"] == "file_unit_key"
