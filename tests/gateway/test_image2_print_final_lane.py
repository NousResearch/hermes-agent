
from __future__ import annotations

import hashlib
import json
import sqlite3
import pytest
from pathlib import Path

import gateway.image2_feishu_ingress as ingress
from types import SimpleNamespace

from gateway.image2_feishu_ingress import Image2IngressSettings, handle_image2_feishu_ingress_event
from gateway.image2_feishu_delivery import FeishuDeliveryError, FeishuImageClient
from gateway.image2_print import parse_print_spec, should_handle_print_request
from gateway.image2_print_reconstruction import build_print_reconstruction_prompt, reconstruct_print_source_with_chatgpt
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


def test_print_request_can_use_quoted_or_thread_user_image_as_direct_print_source(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    db_path = runtime / "image2_jobs.sqlite"
    source = tmp_path / "source.jpg"
    source.write_bytes(b"user-uploaded-approved-print-source")
    settings = Image2IngressSettings(enabled=True, runtime_root=runtime, db_path=db_path, launch_worker=True)
    launched: list[str] = []

    def fake_thread_image(_event, destination, **_kwargs):
        return {
            "path": str(source),
            "mime_type": "image/jpeg",
            "source": "feishu_thread_root_image",
            "parent_message_id": "om_user_image",
        }

    monkeypatch.setattr(ingress, "resolve_feishu_thread_image", fake_thread_image)

    ack = handle_image2_feishu_ingress_event(
        _event("你帮我把这个图生成印刷稿，80×120 厘米的", message_id="om_print", root_id="om_root"),
        settings=settings,
        launch_func=lambda _settings, *, task_id: launched.append(task_id) or {"pid": 123},
    )

    assert "印刷定稿队列" in ack
    assert launched
    print_job = _row(db_path, launched[0])
    payload = json.loads(str(print_job["payload_json"]))
    assert payload["print_request"]["approved_task_id"] == "feishu_source:om_user_image"
    assert payload["print_request"]["approved_image_path"] == str(source)
    assert payload["print_request"]["approved_image_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert payload["print_request"]["feishu_image_message_id"] == "om_user_image"
    assert payload["print_request"]["spec"]["target_width_px"] == 4724
    assert payload["print_request"]["spec"]["target_height_px"] == 7087
    assert payload["source_files"][0]["source"] == "feishu_thread_root_image"


def test_size_only_reply_in_image_thread_routes_to_direct_print_source(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    db_path = runtime / "image2_jobs.sqlite"
    source = tmp_path / "thread-root-source.png"
    source.write_bytes(b"ready-design-draft")
    settings = Image2IngressSettings(enabled=True, runtime_root=runtime, db_path=db_path, launch_worker=True)
    launched: list[str] = []

    def fake_thread_image(_event, destination, **_kwargs):
        return {
            "path": str(source),
            "mime_type": "image/png",
            "source": "feishu_thread_root_image",
            "parent_message_id": "om_root_image",
        }

    monkeypatch.setattr(ingress, "resolve_feishu_thread_image", fake_thread_image)

    ack = handle_image2_feishu_ingress_event(
        _event("80×120厘米", message_id="om_size_only", root_id="om_root_image"),
        settings=settings,
        launch_func=lambda _settings, *, task_id: launched.append(task_id) or {"pid": 123},
    )

    assert "印刷定稿队列" in ack
    assert launched
    print_job = _row(db_path, launched[0])
    payload = json.loads(str(print_job["payload_json"]))
    assert payload["print_request"]["approved_task_id"] == "feishu_source:om_root_image"
    assert payload["print_request"]["approved_image_path"] == str(source)
    assert payload["print_request"]["spec"]["width_mm"] == 800
    assert payload["print_request"]["spec"]["height_mm"] == 1200
    assert payload["print_request"]["spec"]["dpi"] == 150
    assert payload["print_request"]["spec"]["target_width_px"] == 4724
    assert payload["print_request"]["spec"]["target_height_px"] == 7087


def test_size_only_direct_print_source_takes_priority_over_accidental_preview(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    db_path = runtime / "image2_jobs.sqlite"
    store = Image2JobStore(db_path=db_path, runtime_root=runtime)
    preview = store.enqueue_feishu({"feishu_message_id": "om_preview", "chat_id": "oc_chat", "root_id": "om_root", "thread_id": "om_root", "text": "这个就直接是设计稿"})
    preview_dir = Path(str(preview["job_dir"]))
    accidental_preview = preview_dir / "candidates" / "preview.png"
    accidental_preview.parent.mkdir(parents=True, exist_ok=True)
    accidental_preview.write_bytes(b"accidental-generated-preview")
    accidental_sha = hashlib.sha256(accidental_preview.read_bytes()).hexdigest()
    (preview_dir / "worker_result.json").write_text(json.dumps({"delivery_contract": {"status": "ready_to_send", "image_path": str(accidental_preview), "image_sha256": accidental_sha}, "delivery_readback": {"verified": True, "message_id": "om_accidental_img", "readback_msg_type": "image"}}, ensure_ascii=False), encoding="utf-8")
    store.mark_readback_verified(task_id=str(preview["task_id"]), worker_id="unit")

    source = tmp_path / "thread-root-design.png"
    source.write_bytes(b"real-user-design-draft")
    settings = Image2IngressSettings(enabled=True, runtime_root=runtime, db_path=db_path, launch_worker=True)
    launched: list[str] = []

    def fake_thread_image(_event, destination, **_kwargs):
        return {
            "path": str(source),
            "mime_type": "image/png",
            "source": "feishu_thread_root_image",
            "parent_message_id": "om_root",
        }

    monkeypatch.setattr(ingress, "resolve_feishu_thread_image", fake_thread_image)

    ack = handle_image2_feishu_ingress_event(
        _event("80×120厘米", message_id="om_size_only", root_id="om_root"),
        settings=settings,
        launch_func=lambda _settings, *, task_id: launched.append(task_id) or {"pid": 123},
    )

    assert "印刷定稿队列" in ack
    print_job = _row(db_path, launched[0])
    payload = json.loads(str(print_job["payload_json"]))
    assert payload["print_request"]["approved_task_id"] == "feishu_source:om_root"
    assert payload["print_request"]["approved_image_path"] == str(source)
    assert payload["print_request"]["approved_image_sha256"] != accidental_sha



def test_print_request_inherits_size_from_latest_same_thread_preview_text(tmp_path):
    runtime = tmp_path / "runtime"
    db_path = runtime / "image2_jobs.sqlite"
    store = Image2JobStore(db_path=db_path, runtime_root=runtime)
    preview = store.enqueue_feishu({
        "feishu_message_id": "om_design_request",
        "chat_id": "oc_chat",
        "root_id": "om_topic",
        "thread_id": "om_topic",
        "text": "/image2 做一张火宫殿 T3 海报，尺寸 80×120cm",
    })
    preview_dir = Path(str(preview["job_dir"]))
    approved = preview_dir / "candidates" / "approved.png"
    approved.parent.mkdir(parents=True, exist_ok=True)
    approved.write_bytes(b"approved-preview-with-size")
    approved_sha = hashlib.sha256(approved.read_bytes()).hexdigest()
    (preview_dir / "worker_result.json").write_text(json.dumps({
        "delivery_contract": {"status": "ready_to_send", "image_path": str(approved), "image_sha256": approved_sha},
        "delivery_readback": {"verified": True, "message_id": "om_delivered_preview", "readback_msg_type": "image"},
        "generation_result": {"link": "https://chatgpt.com/c/size-thread", "title": "Image2 size thread"},
    }, ensure_ascii=False), encoding="utf-8")
    store.mark_readback_verified(task_id=str(preview["task_id"]), worker_id="unit")
    settings = Image2IngressSettings(enabled=True, runtime_root=runtime, db_path=db_path, launch_worker=True)

    ack = handle_image2_feishu_ingress_event(
        _event("生成印刷稿", message_id="om_print", root_id="om_topic"),
        settings=settings,
        launch_func=lambda _settings, *, task_id: {"pid": 123},
    )

    assert "需要尺寸" not in ack
    assert "800×1200mm / 150DPI" in ack
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM image2_jobs WHERE feishu_message_id = ?", ("om_print",)).fetchone()
    assert row is not None
    payload = json.loads(row["payload_json"])
    assert payload["print_request"]["approved_task_id"] == preview["task_id"]
    assert payload["print_request"]["spec"]["width_mm"] == 800
    assert payload["print_request"]["spec"]["height_mm"] == 1200
    assert payload["print_request"]["spec"]["target_width_px"] == 4724
    assert payload["print_request"]["spec"]["target_height_px"] == 7087


def test_size_only_followup_can_resolve_preview_by_delivered_image_message_id(tmp_path):
    runtime = tmp_path / "runtime"
    db_path = runtime / "image2_jobs.sqlite"
    store = Image2JobStore(db_path=db_path, runtime_root=runtime)
    preview = store.enqueue_feishu({
        "feishu_message_id": "om_design_request",
        "chat_id": "oc_chat",
        "root_id": "om_original_topic",
        "thread_id": "om_original_topic",
        "text": "/image2 做一张设计图",
    })
    preview_dir = Path(str(preview["job_dir"]))
    approved = preview_dir / "candidates" / "approved.png"
    approved.parent.mkdir(parents=True, exist_ok=True)
    approved.write_bytes(b"approved-preview-delivered-message")
    approved_sha = hashlib.sha256(approved.read_bytes()).hexdigest()
    (preview_dir / "worker_result.json").write_text(json.dumps({
        "delivery_contract": {"status": "ready_to_send", "image_path": str(approved), "image_sha256": approved_sha},
        "delivery_readback": {"verified": True, "message_id": "om_delivered_preview", "readback_msg_type": "image"},
    }, ensure_ascii=False), encoding="utf-8")
    store.mark_readback_verified(task_id=str(preview["task_id"]), worker_id="unit")
    settings = Image2IngressSettings(enabled=True, runtime_root=runtime, db_path=db_path, launch_worker=True)

    ack = handle_image2_feishu_ingress_event(
        _event("80×120cm", message_id="om_size", root_id="om_delivered_preview"),
        settings=settings,
        launch_func=lambda _settings, *, task_id: {"pid": 456},
    )

    assert "没有找到" not in ack
    assert "800×1200mm / 150DPI" in ack
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM image2_jobs WHERE feishu_message_id = ?", ("om_size",)).fetchone()
    assert row is not None
    payload = json.loads(row["payload_json"])
    assert payload["print_request"]["approved_task_id"] == preview["task_id"]
    assert payload["print_request"]["approved_image_path"] == str(approved)

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
    reconstructed = tmp_path / "reconstructed-print-source.png"
    reconstructed.write_bytes(b"reconstructed-print-source")
    reconstructed_sha = hashlib.sha256(reconstructed.read_bytes()).hexdigest()
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
        return {"status": "pass", "psd_path": str(psd), "pdf_path": str(pdf), "preview_path": str(preview), "approved_sha256": hashlib.sha256(Path(approved_image_path).read_bytes()).hexdigest(), "target_width_px": 5906, "target_height_px": 8858, "dpi": 150}

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
        print_reconstructor=lambda **kwargs: {"status": "pass", "image_path": str(reconstructed), "image_sha256": reconstructed_sha, "mode": "unit_reconstruction"},
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



def test_feishu_file_client_accepts_feishu_transformed_file_key_when_file_name_matches(tmp_path):
    doc = tmp_path / "final_flat.psd"
    doc.write_bytes(b"fake psd bytes")

    def fake_post(url: str, **kwargs):
        if url.endswith("/auth/v3/tenant_access_token/internal"):
            return _Response({"code": 0, "tenant_access_token": "token-redacted"})
        if url.endswith("/im/v1/files"):
            return _Response({"code": 0, "data": {"file_key": "file_upload_key"}})
        if url.endswith("/im/v1/messages/om_root/reply"):
            return _Response({"code": 0, "data": {"message_id": "om_file_reply"}})
        raise AssertionError(url)

    def fake_get(url: str, **kwargs):
        assert url.endswith("/im/v1/messages/om_file_reply")
        # Live Feishu may return a message-resource file_key that differs from the upload file_key.
        return _Response({"code": 0, "data": {"item": {"message_id": "om_file_reply", "msg_type": "file", "body": {"content": json.dumps({"file_key": "file_readback_key", "file_name": "final_flat.psd"})}}}})

    client = FeishuImageClient(app_id="app", app_secret="secret", http_post=fake_post, http_get=fake_get)
    result = client.send_file_and_verify(doc, chat_id="oc_chat", reply_to="om_root", file_name="final_flat.psd")

    assert result["verified"] is True
    assert result["message_id"] == "om_file_reply"
    assert result["file_key"] == "file_upload_key"
    assert result["readback_file_key"] == "file_readback_key"
    assert result["file_key_matches"] is False
    assert result["readback_file_name"] == "final_flat.psd"


def test_print_delivery_files_uses_transfer_psd_when_flat_psd_exceeds_feishu_limit(tmp_path):
    from gateway.image2_worker import _build_print_delivery_files

    approved = tmp_path / "approved.png"
    approved.write_bytes(b"approved")
    job_dir = tmp_path / "job"
    original_psd = job_dir / "print" / "psd" / "original_150dpi.psd"
    pdf = job_dir / "print" / "proof" / "proof.pdf"
    highres = job_dir / "print" / "highres" / "highres.png"
    for path, payload in [(original_psd, b"x" * 150), (pdf, b"pdf"), (highres, b"y" * 150)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    package_result = {
        "status": "pass",
        "psd_path": str(original_psd),
        "pdf_path": str(pdf),
        "highres_path": str(highres),
        "approved_sha256": hashlib.sha256(approved.read_bytes()).hexdigest(),
        "width_mm": 800,
        "height_mm": 1200,
        "dpi": 150,
    }
    calls: list[int] = []

    def fake_packager(*, job_dir: Path, approved_image_path: Path, spec: dict[str, object], environ: dict[str, str]):
        dpi = int(spec["dpi"])
        assert "target_width_px" not in spec
        assert "target_height_px" not in spec
        calls.append(dpi)
        fallback_psd = job_dir / "print" / "psd" / f"transfer_{dpi}dpi.psd"
        fallback_pdf = job_dir / "print" / "proof" / "proof.pdf"
        fallback_highres = job_dir / "print" / "highres" / "highres.png"
        for path, payload in [(fallback_psd, b"p" * (90 if dpi == 90 else 130)), (fallback_pdf, b"pdf"), (fallback_highres, b"png")]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        return {"status": "pass", "psd_path": str(fallback_psd), "pdf_path": str(fallback_pdf), "highres_path": str(fallback_highres), "approved_sha256": package_result["approved_sha256"], "dpi": dpi}

    files, meta = _build_print_delivery_files(
        package_result=package_result,
        job_dir=job_dir,
        approved_path=approved,
        print_request={"spec": {"width_mm": 800, "height_mm": 1200, "dpi": 150, "output_psd_type": "flat_single_layer"}},
        environ={"FEISHU_IM_FILE_UPLOAD_MAX_BYTES": "100", "IMAGE2_PRINT_FEISHU_TRANSFER_DPI_CANDIDATES": "100,90"},
        packager=fake_packager,
    )

    file_paths = [Path(str(item["path"])) for item in files]
    assert pdf in file_paths
    assert highres not in file_paths
    assert original_psd not in file_paths
    assert any(path.name == "transfer_90dpi.psd" for path in file_paths)
    assert any(path.name == "highres.png" and "print_transfer_90dpi" in str(path) for path in file_paths)
    assert calls == [100, 90]
    assert meta["psd_delivery_mode"] == "transfer_fallback"
    assert meta["original_psd_size"] == 150
    assert meta["transfer_psd_dpi"] == 90



def test_feishu_file_client_rejects_transformed_file_key_without_file_name(tmp_path):
    doc = tmp_path / "final_flat.psd"
    doc.write_bytes(b"fake psd bytes")

    def fake_post(url: str, **kwargs):
        if url.endswith("/auth/v3/tenant_access_token/internal"):
            return _Response({"code": 0, "tenant_access_token": "token-redacted"})
        if url.endswith("/im/v1/files"):
            return _Response({"code": 0, "data": {"file_key": "file_upload_key"}})
        if url.endswith("/im/v1/messages/om_root/reply"):
            return _Response({"code": 0, "data": {"message_id": "om_file_reply"}})
        raise AssertionError(url)

    def fake_get(url: str, **kwargs):
        return _Response({"code": 0, "data": {"item": {"message_id": "om_file_reply", "msg_type": "file", "body": {"content": json.dumps({"file_key": "file_readback_key"})}}}})

    client = FeishuImageClient(app_id="app", app_secret="secret", http_post=fake_post, http_get=fake_get)
    with pytest.raises(FeishuDeliveryError, match="read-back file_name missing"):
        client.send_file_and_verify(doc, chat_id="oc_chat", reply_to="om_root", file_name="final_flat.psd")


def test_worker_print_lane_fails_closed_when_no_deliverable_psd(tmp_path):
    runtime = tmp_path / "runtime"
    db_path = runtime / "image2_jobs.sqlite"
    approved = tmp_path / "approved.png"
    approved.write_bytes(b"approved")
    approved_sha = hashlib.sha256(approved.read_bytes()).hexdigest()
    reconstructed = tmp_path / "reconstructed-print-source.png"
    reconstructed.write_bytes(b"reconstructed-print-source")
    reconstructed_sha = hashlib.sha256(reconstructed.read_bytes()).hexdigest()
    spec = parse_print_spec("定稿，出印刷版，尺寸 100×150cm")
    job = Image2JobStore(db_path=db_path, runtime_root=runtime).enqueue_feishu({
        "feishu_message_id": "om_print",
        "chat_id": "oc_chat",
        "root_id": "om_root",
        "thread_id": "om_thread",
        "text": "定稿，出印刷版，尺寸 100×150cm",
        "print_request": {"approved_task_id": "img2_preview", "approved_image_path": str(approved), "approved_image_sha256": approved_sha, "spec": spec},
    })

    def fake_print_packager(*, job_dir: Path, approved_image_path: Path, spec: dict[str, object], environ: dict[str, str]):
        pdf = job_dir / "print" / "proof" / "unit_proof.pdf"
        highres = job_dir / "print" / "highres" / "unit.png"
        for path, payload in [(pdf, b"pdf"), (highres, b"png")]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        return {"status": "pass", "pdf_path": str(pdf), "highres_path": str(highres), "approved_sha256": hashlib.sha256(Path(approved_image_path).read_bytes()).hexdigest(), "dpi": 150}

    def should_not_send(**kwargs):
        raise AssertionError("sender must not be called without a deliverable PSD")

    result = run_worker(
        db_path=db_path,
        runtime_root=runtime,
        task_id=str(job["task_id"]),
        worker_id="print-worker",
        environ={"IMAGE2_WORKER_LIVE_ENABLED": "1", "FEISHU_APP_ID": "present", "FEISHU_APP_SECRET": "present"},
        print_packager=fake_print_packager,
        print_reconstructor=lambda **kwargs: {"status": "pass", "image_path": str(reconstructed), "image_sha256": reconstructed_sha, "mode": "unit_reconstruction"},
        file_delivery_sender=should_not_send,
    )

    assert result["status"] == "failed_final"
    assert result["reason"] == "print_delivery_file_missing"



def test_worker_print_lane_uses_chatgpt_reconstruction_before_packaging(tmp_path):
    runtime = tmp_path / "runtime"
    db_path = runtime / "image2_jobs.sqlite"
    approved = tmp_path / "approved.png"
    approved.write_bytes(b"approved-preview")
    approved_sha = hashlib.sha256(approved.read_bytes()).hexdigest()
    reconstructed = tmp_path / "reconstructed.png"
    reconstructed.write_bytes(b"chatgpt-reconstructed-preview")
    reconstructed_sha = hashlib.sha256(reconstructed.read_bytes()).hexdigest()
    spec = parse_print_spec("定稿，出印刷版，尺寸 80×120cm，200dpi")
    job = Image2JobStore(db_path=db_path, runtime_root=runtime).enqueue_feishu({
        "feishu_message_id": "om_print",
        "chat_id": "oc_chat",
        "root_id": "om_root",
        "thread_id": "om_root",
        "text": "定稿，出印刷版，尺寸 80×120cm，200dpi",
        "print_request": {"approved_task_id": "img2_preview", "approved_image_path": str(approved), "approved_image_sha256": approved_sha, "spec": spec, "chatgpt_url": "https://chatgpt.com/c/test"},
    })
    calls = {}

    def fake_reconstructor(**kwargs):
        calls["reconstructor"] = {
            "approved": str(kwargs["approved_image_path"]),
            "approved_sha256": kwargs["approved_sha256"],
            "history": kwargs["print_request"].get("chatgpt_url"),
        }
        return {"status": "pass", "image_path": str(reconstructed), "image_sha256": reconstructed_sha, "mode": "chatgpt_reconstruction", "attempts": [{"file_upload": True}, {"file_upload": False}]}

    def fake_packager(*, job_dir: Path, approved_image_path: Path, spec: dict[str, object], environ: dict[str, str]):
        calls["packager"] = {"source": str(approved_image_path), "dpi": spec["dpi"]}
        psd = job_dir / "print" / "psd" / "unit_flat.psd"
        pdf = job_dir / "print" / "proof" / "unit_proof.pdf"
        highres = job_dir / "print" / "highres" / "unit.png"
        for path, payload in [(psd, b"psd"), (pdf, b"pdf"), (highres, b"png")]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        return {"status": "pass", "psd_path": str(psd), "pdf_path": str(pdf), "highres_path": str(highres), "approved_sha256": reconstructed_sha, "dpi": spec["dpi"]}

    from gateway.image2_worker import run_worker

    result = run_worker(
        db_path=db_path,
        runtime_root=runtime,
        worker_id="print-worker",
        task_id=str(job["task_id"]),
        environ={"FEISHU_APP_ID": "app", "FEISHU_APP_SECRET": "secret"},
        print_packager=fake_packager,
        print_reconstructor=fake_reconstructor,
        file_delivery_sender=lambda **_kwargs: {"verified": True},
    )
    assert result["status"] == "readback_verified"
    assert calls["reconstructor"]["approved"] == str(approved)
    assert calls["reconstructor"]["approved_sha256"] == approved_sha
    assert calls["reconstructor"]["history"] == "https://chatgpt.com/c/test"
    assert calls["packager"]["source"] == str(reconstructed)
    assert result["print_reconstruction"]["mode"] == "chatgpt_reconstruction"


def test_worker_print_lane_fails_closed_when_chatgpt_reconstruction_rejects_source_echo(tmp_path):
    runtime = tmp_path / "runtime"
    db_path = runtime / "image2_jobs.sqlite"
    approved = tmp_path / "approved.png"
    approved.write_bytes(b"approved-preview")
    approved_sha = hashlib.sha256(approved.read_bytes()).hexdigest()
    spec = parse_print_spec("定稿，出印刷版，尺寸 80×120cm")
    job = Image2JobStore(db_path=db_path, runtime_root=runtime).enqueue_feishu({
        "feishu_message_id": "om_print",
        "chat_id": "oc_chat",
        "root_id": "om_root",
        "thread_id": "om_root",
        "text": "定稿，出印刷版，尺寸 80×120cm",
        "print_request": {"approved_task_id": "img2_preview", "approved_image_path": str(approved), "approved_image_sha256": approved_sha, "spec": spec},
    })
    from gateway.image2_worker import run_worker

    result = run_worker(
        db_path=db_path,
        runtime_root=runtime,
        worker_id="print-worker",
        task_id=str(job["task_id"]),
        environ={"FEISHU_APP_ID": "app", "FEISHU_APP_SECRET": "secret"},
        print_reconstructor=lambda **_kwargs: {"status": "rejected", "reason": "source_sha_match", "image_path": str(approved), "image_sha256": approved_sha},
        print_packager=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("packager must not run after rejected ChatGPT reconstruction")),
        file_delivery_sender=lambda **_kwargs: {"verified": True},
    )
    assert result["status"] == "failed_final"
    assert result["reason"] == "print_reconstruction_failed"
    assert "source_sha_match" in result["last_error"]


def test_print_reconstruction_prompt_names_exact_target_pixels_for_physical_dpi():
    prompt = build_print_reconstruction_prompt(
        print_request={"spec": {"width_mm": 800, "height_mm": 1200, "dpi": 200, "target_width_px": 6299, "target_height_px": 9449}},
        prompt_text="主视觉对象：夏日鲜果冰柠系列",
    )

    assert "800×1200mm" in prompt
    assert "200DPI" in prompt
    assert "目标像素尺寸：6299×9449px" in prompt
    assert "如果平台无法直接输出该像素尺寸" in prompt


def test_reconstruct_print_source_retries_without_file_when_upload_route_is_blocked(tmp_path):
    approved = tmp_path / "approved.png"
    approved.write_bytes(b"approved-preview")
    approved_sha = hashlib.sha256(approved.read_bytes()).hexdigest()
    out_file = tmp_path / "job" / "print" / "reconstruction" / "candidates" / "chatgpt_rebuilt.png"
    calls = []
    timeouts = []

    class Proc:
        def __init__(self, returncode=0, stdout="[]", stderr=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(cmd, cwd, env, text, capture_output, timeout):
        calls.append(list(cmd))
        timeouts.append(timeout)
        if "--file" in cmd:
            return Proc(0, '[{"status":"blocked","file":"-"}]', "blocked")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_bytes(b"rebuilt-not-source")
        return Proc(0, '[{"status":"saved","file":"%s","link":"https://chatgpt.com/c/rebuilt"}]' % out_file)

    result = reconstruct_print_source_with_chatgpt(
        job_dir=tmp_path / "job",
        approved_image_path=approved,
        approved_sha256=approved_sha,
        print_request={"spec": {"width_mm": 800, "height_mm": 1200, "dpi": 200}, "chatgpt_url": "https://chatgpt.com/c/old"},
        prompt_text="主视觉对象：夏日鲜果冰柠系列",
        environ={"PATH": "/usr/bin", "IMAGE2_PRINT_RECONSTRUCT_OPENCLI_TIMEOUT": "111", "IMAGE2_PRINT_RECONSTRUCT_SUBPROCESS_TIMEOUT": "222"},
        command_runner=fake_run,
    )
    assert result["status"] == "pass"
    assert result["image_path"] == str(out_file)
    assert result["image_sha256"] != approved_sha
    assert "--file" in calls[0]
    assert "--history" in calls[0]
    assert calls[0][calls[0].index("--timeout") + 1] == "111"
    assert timeouts == [222, 222]
    assert "--file" not in calls[1]
    assert "--history" in calls[1]
