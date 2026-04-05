from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_dir


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


@dataclass
class WeChatAccount:
    account_id: str
    token: str
    base_url: str
    user_id: Optional[str] = None
    enabled: bool = True
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)


class WeChatStateStore:
    def __init__(self, root: Path | None = None):
        self.root = root or get_hermes_dir("platforms/wechat", "wechat")
        self.accounts_dir = self.root / "accounts"
        self.index_path = self.root / "accounts.json"

    def _account_path(self, account_id: str) -> Path:
        return self.accounts_dir / f"{account_id}.json"

    def _sync_path(self, account_id: str) -> Path:
        return self.accounts_dir / f"{account_id}.sync.json"

    def _context_path(self, account_id: str) -> Path:
        return self.accounts_dir / f"{account_id}.context_tokens.json"

    def list_account_ids(self) -> list[str]:
        if not self.index_path.exists():
            return []
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(raw, list):
            return []
        return [str(item) for item in raw if str(item).strip()]

    def _register_account_id(self, account_id: str) -> None:
        ids = self.list_account_ids()
        if account_id not in ids:
            ids.append(account_id)
            _atomic_write_json(self.index_path, ids)

    def save_account(self, account: WeChatAccount) -> None:
        payload = asdict(account)
        payload["updated_at"] = _utcnow()
        _atomic_write_json(self._account_path(account.account_id), payload)
        self._register_account_id(account.account_id)

    def load_account(self, account_id: str) -> Optional[WeChatAccount]:
        path = self._account_path(account_id)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return WeChatAccount(**raw)

    def save_sync_cursor(self, account_id: str, cursor: str) -> None:
        _atomic_write_json(self._sync_path(account_id), {"cursor": cursor, "updated_at": _utcnow()})

    def load_sync_cursor(self, account_id: str) -> Optional[str]:
        path = self._sync_path(account_id)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        value = raw.get("cursor")
        return str(value) if value else None

    def _load_context_tokens(self, account_id: str) -> Dict[str, Dict[str, str]]:
        path = self._context_path(account_id)
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return raw if isinstance(raw, dict) else {}

    def set_context_token(self, account_id: str, user_id: str, token: str) -> None:
        tokens = self._load_context_tokens(account_id)
        tokens[user_id] = {"token": token, "updated_at": _utcnow()}
        _atomic_write_json(self._context_path(account_id), tokens)

    def get_context_token(self, account_id: str, user_id: str) -> Optional[str]:
        tokens = self._load_context_tokens(account_id)
        entry = tokens.get(user_id)
        if not isinstance(entry, dict):
            return None
        token = entry.get("token")
        return str(token) if token else None

    def find_account_ids_by_context_token(self, user_id: str) -> list[str]:
        matches: list[str] = []
        for account_id in self.list_account_ids():
            if self.get_context_token(account_id, user_id):
                matches.append(account_id)
        return matches
