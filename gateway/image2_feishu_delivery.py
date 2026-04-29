"""Native Feishu image delivery and exact read-back for Hermes Image2."""
from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

try:
    import requests
except Exception:  # pragma: no cover - tests inject HTTP callables
    requests = None  # type: ignore


class FeishuDeliveryError(RuntimeError):
    pass


class FeishuImageClient:
    def __init__(
        self,
        app_id: Optional[str] = None,
        app_secret: Optional[str] = None,
        *,
        base_url: str = "https://open.feishu.cn/open-apis",
        http_post: Optional[Callable[..., Any]] = None,
        http_get: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.app_id = app_id or os.environ.get("FEISHU_APP_ID")
        self.app_secret = app_secret or os.environ.get("FEISHU_APP_SECRET")
        if not self.app_id or not self.app_secret:
            raise FeishuDeliveryError("FEISHU_APP_ID / FEISHU_APP_SECRET are required")
        if (http_post is None or http_get is None) and requests is None:
            raise FeishuDeliveryError("requests is required unless http_post/http_get are injected")
        self.base_url = base_url.rstrip("/")
        self.http_post = http_post or requests.post  # type: ignore[union-attr]
        self.http_get = http_get or requests.get  # type: ignore[union-attr]

    @staticmethod
    def _check_payload(response: Any, *, action: str) -> dict[str, Any]:
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise FeishuDeliveryError(f"Feishu {action} failed: code={payload.get('code')} msg={payload.get('msg')}")
        return dict(payload)

    def tenant_access_token(self) -> str:
        payload = self._check_payload(
            self.http_post(
                f"{self.base_url}/auth/v3/tenant_access_token/internal",
                json={"app_id": self.app_id, "app_secret": self.app_secret},
                timeout=20,
            ),
            action="tenant_access_token",
        )
        token = payload.get("tenant_access_token") or (payload.get("data") or {}).get("tenant_access_token")
        if not token:
            raise FeishuDeliveryError("tenant_access_token missing in Feishu response")
        return str(token)

    def upload_image(self, image_path: Path, *, token: str) -> str:
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FeishuDeliveryError(f"image file does not exist: {image_path}")
        mime = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
        with image_path.open("rb") as handle:
            payload = self._check_payload(
                self.http_post(
                    f"{self.base_url}/im/v1/images",
                    headers={"Authorization": f"Bearer {token}"},
                    data={"image_type": "message"},
                    files={"image": (image_path.name, handle, mime)},
                    timeout=60,
                ),
                action="upload_image",
            )
        image_key = (payload.get("data") or {}).get("image_key")
        if not image_key:
            raise FeishuDeliveryError("image_key missing in Feishu upload response")
        return str(image_key)

    def send_image_message(self, *, chat_id: str, image_key: str, token: str, reply_to: str = "") -> str:
        body = {"msg_type": "image", "content": json.dumps({"image_key": image_key}, ensure_ascii=False), "uuid": str(uuid.uuid4())}
        if reply_to:
            body["reply_in_thread"] = True
            payload = self._check_payload(
                self.http_post(
                    f"{self.base_url}/im/v1/messages/{reply_to}/reply",
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"},
                    json=body,
                    timeout=30,
                ),
                action="reply_image_message",
            )
        else:
            payload = self._check_payload(
                self.http_post(
                    f"{self.base_url}/im/v1/messages?receive_id_type=chat_id",
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"},
                    json={"receive_id": chat_id, "msg_type": "image", "content": body["content"], "uuid": body["uuid"]},
                    timeout=30,
                ),
                action="send_image_message",
            )
        message_id = (payload.get("data") or {}).get("message_id")
        if not message_id:
            raise FeishuDeliveryError("message_id missing in Feishu send response")
        return str(message_id)

    def read_message(self, *, message_id: str, token: str) -> dict[str, Any]:
        payload = self._check_payload(
            self.http_get(
                f"{self.base_url}/im/v1/messages/{message_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=30,
            ),
            action="read_message",
        )
        data = payload.get("data") or {}
        if isinstance(data.get("item"), Mapping):
            return dict(data["item"])
        items = data.get("items") or []
        if items:
            return dict(items[0])
        if any(k in data for k in ("message_id", "msg_type", "content", "body")):
            return dict(data)
        raise FeishuDeliveryError("message read-back returned no item")

    @staticmethod
    def _readback_image_key(readback: Mapping[str, Any]) -> str:
        content = (readback.get("body") or {}).get("content") if isinstance(readback.get("body"), Mapping) else None
        content = content or readback.get("content")
        if not content:
            return ""
        try:
            parsed = json.loads(content) if isinstance(content, str) else content
        except json.JSONDecodeError:
            return ""
        return str(parsed.get("image_key") or "") if isinstance(parsed, Mapping) else ""

    def send_image_and_verify(
        self,
        image_path: Path,
        *,
        chat_id: str,
        reply_to: str = "",
        candidate_sha256: str = "",
    ) -> dict[str, Any]:
        image_path = Path(image_path)
        current_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
        expected_sha256 = str(candidate_sha256 or "")
        if expected_sha256 and current_sha256 != expected_sha256:
            raise FeishuDeliveryError("candidate_sha256 mismatch before upload")
        token = self.tenant_access_token()
        image_key = self.upload_image(image_path, token=token)
        message_id = self.send_image_message(chat_id=chat_id, reply_to=reply_to, image_key=image_key, token=token)
        readback = self.read_message(message_id=message_id, token=token)
        msg_type = readback.get("msg_type")
        if msg_type != "image":
            raise FeishuDeliveryError(f"read-back msg_type mismatch: expected image, got {msg_type!r}")
        readback_key = self._readback_image_key(readback)
        if not readback_key:
            raise FeishuDeliveryError("read-back image_key missing")
        if readback_key != image_key:
            raise FeishuDeliveryError("read-back image_key mismatch")
        return {
            "verified": True,
            "message_id": message_id,
            "image_key": image_key,
            "readback_msg_type": msg_type,
            "candidate_path": str(image_path),
            "candidate_sha256": current_sha256,
            "reply_to": reply_to,
            "chat_id": chat_id,
        }

    def upload_file(self, file_path: Path, *, token: str, file_name: str = "") -> str:
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FeishuDeliveryError(f"file does not exist: {file_path}")
        name = file_name or file_path.name
        mime = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        with file_path.open("rb") as handle:
            payload = self._check_payload(
                self.http_post(
                    f"{self.base_url}/im/v1/files",
                    headers={"Authorization": f"Bearer {token}"},
                    data={"file_type": "stream", "file_name": name},
                    files={"file": (name, handle, mime)},
                    timeout=120,
                ),
                action="upload_file",
            )
        file_key = (payload.get("data") or {}).get("file_key")
        if not file_key:
            raise FeishuDeliveryError("file_key missing in Feishu upload response")
        return str(file_key)

    def send_file_message(self, *, chat_id: str, file_key: str, token: str, reply_to: str = "") -> str:
        body = {"msg_type": "file", "content": json.dumps({"file_key": file_key}, ensure_ascii=False), "uuid": str(uuid.uuid4())}
        if reply_to:
            body["reply_in_thread"] = True
            payload = self._check_payload(
                self.http_post(
                    f"{self.base_url}/im/v1/messages/{reply_to}/reply",
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"},
                    json=body,
                    timeout=30,
                ),
                action="reply_file_message",
            )
        else:
            payload = self._check_payload(
                self.http_post(
                    f"{self.base_url}/im/v1/messages?receive_id_type=chat_id",
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"},
                    json={"receive_id": chat_id, "msg_type": "file", "content": body["content"], "uuid": body["uuid"]},
                    timeout=30,
                ),
                action="send_file_message",
            )
        message_id = (payload.get("data") or {}).get("message_id")
        if not message_id:
            raise FeishuDeliveryError("message_id missing in Feishu send response")
        return str(message_id)

    @staticmethod
    def _readback_file_key(readback: Mapping[str, Any]) -> str:
        content = (readback.get("body") or {}).get("content") if isinstance(readback.get("body"), Mapping) else None
        content = content or readback.get("content")
        if not content:
            return ""
        try:
            parsed = json.loads(content) if isinstance(content, str) else content
        except json.JSONDecodeError:
            return ""
        return str(parsed.get("file_key") or "") if isinstance(parsed, Mapping) else ""

    def send_file_and_verify(self, file_path: Path, *, chat_id: str, reply_to: str = "", file_name: str = "") -> dict[str, Any]:
        file_path = Path(file_path)
        file_sha256 = hashlib.sha256(file_path.read_bytes()).hexdigest()
        token = self.tenant_access_token()
        file_key = self.upload_file(file_path, token=token, file_name=file_name)
        message_id = self.send_file_message(chat_id=chat_id, reply_to=reply_to, file_key=file_key, token=token)
        readback = self.read_message(message_id=message_id, token=token)
        msg_type = readback.get("msg_type")
        if msg_type != "file":
            raise FeishuDeliveryError(f"read-back msg_type mismatch: expected file, got {msg_type!r}")
        readback_key = self._readback_file_key(readback)
        if not readback_key:
            raise FeishuDeliveryError("read-back file_key missing")
        if readback_key != file_key:
            raise FeishuDeliveryError("read-back file_key mismatch")
        return {
            "verified": True,
            "message_id": message_id,
            "file_key": file_key,
            "readback_msg_type": msg_type,
            "file_path": str(file_path),
            "file_sha256": file_sha256,
            "reply_to": reply_to,
            "chat_id": chat_id,
        }

    def send_files_and_verify(self, files: list[Mapping[str, Any]], *, chat_id: str, reply_to: str = "") -> dict[str, Any]:
        readbacks = []
        for item in files:
            path = Path(str(item.get("path") or "")).expanduser()
            if not path.is_file():
                raise FeishuDeliveryError(f"file does not exist: {path}")
            readbacks.append(self.send_file_and_verify(path, chat_id=chat_id, reply_to=reply_to, file_name=str(item.get("file_name") or path.name)))
        return {"verified": bool(readbacks) and all(item.get("verified") is True and item.get("readback_msg_type") == "file" for item in readbacks), "readbacks": readbacks, "chat_id": chat_id, "reply_to": reply_to}


def send_feishu_image_from_contract(
    *,
    image_path: Path,
    chat_id: str,
    reply_to: str,
    candidate_sha256: str,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    env = dict(os.environ)
    if environ:
        env.update({str(k): str(v) for k, v in environ.items()})
    client = FeishuImageClient(app_id=env.get("FEISHU_APP_ID"), app_secret=env.get("FEISHU_APP_SECRET"))
    return client.send_image_and_verify(Path(image_path), chat_id=str(chat_id), reply_to=str(reply_to or ""), candidate_sha256=str(candidate_sha256 or ""))


def send_feishu_files_from_print_package(
    *,
    files: list[Mapping[str, Any]],
    chat_id: str,
    reply_to: str,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    env = dict(os.environ)
    if environ:
        env.update({str(k): str(v) for k, v in environ.items()})
    client = FeishuImageClient(app_id=env.get("FEISHU_APP_ID"), app_secret=env.get("FEISHU_APP_SECRET"))
    return client.send_files_and_verify(files, chat_id=str(chat_id), reply_to=str(reply_to or ""))
