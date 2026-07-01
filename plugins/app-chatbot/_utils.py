"""Shared helpers for the app-chatbot plugin."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_OBJECT_ID_RE = re.compile(r"^[0-9a-fA-F]{24}$")

_REDACT_FIELDS = frozenset({
    "password",
    "emailOTP",
    "emailForgotPasswordVerifyToken",
})


def default_user_id() -> str:
    return str(os.getenv("APP_CHATBOT_DEFAULT_USER_ID", "") or "").strip()


def _ensure_mongodb_deps() -> None:
    try:
        from tools.lazy_deps import FeatureUnavailable, ensure

        ensure("tool.mongodb", prompt=False)
    except FeatureUnavailable as exc:
        raise RuntimeError(str(exc)) from exc


def parse_object_id(value: str) -> Any:
    _ensure_mongodb_deps()
    from bson import ObjectId
    from bson.errors import InvalidId

    raw = (value or "").strip()
    if not _OBJECT_ID_RE.fullmatch(raw):
        raise ValueError(f"Invalid user_id: expected 24-char hex ObjectId, got {value!r}")
    try:
        return ObjectId(raw)
    except InvalidId as exc:
        raise ValueError(f"Invalid ObjectId: {value!r}") from exc


def redact_document(doc: Any) -> Any:
    if isinstance(doc, dict):
        return {
            key: "[REDACTED]" if key in _REDACT_FIELDS else redact_document(val)
            for key, val in doc.items()
        }
    if isinstance(doc, list):
        return [redact_document(item) for item in doc]
    return doc


def serialize_doc(doc: Any) -> Any:
    _ensure_mongodb_deps()
    from bson import json_util

    if doc is None:
        return None
    return json.loads(json_util.dumps(redact_document(doc)))


def serialize_docs(docs: List[Any]) -> List[Any]:
    return [serialize_doc(doc) for doc in docs]


def get_database_name() -> str:
    try:
        from hermes_cli.config import cfg_get, load_config
        cfg = load_config()
        db_name = str(cfg_get(cfg, "mongodb", "default_database", default="") or "").strip()
        return db_name or "crwd_staging"
    except Exception:
        return "crwd_staging"


def get_mongo_db():
    uri = os.getenv("MONGODB_URI", "")
    if not uri:
        raise RuntimeError("MONGODB_URI is not set")
    _ensure_mongodb_deps()
    from pymongo import MongoClient
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    return client[get_database_name()]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def active_open_gig_filter(now: Optional[datetime] = None) -> Dict[str, Any]:
    """Filter for active, non-deleted, non-archived gigs with open end date."""
    current = now or utc_now()
    return {
        "status": "Active",
        "isDeleted": {"$ne": True},
        "isArchived": {"$ne": True},
        "$or": [
            {"end_date": {"$gte": current}},
            {"end_date": None},
            {"end_date": {"$exists": False}},
        ],
    }


def gig_payout(doc: Dict[str, Any]) -> Any:
    price = doc.get("price")
    if price not in (None, ""):
        return price
    payout = doc.get("payout")
    if payout not in (None, 0, 0.0):
        return payout
    stores = doc.get("gig_stores") or []
    if stores and isinstance(stores[0], dict):
        return stores[0].get("payout_amount")
    return None


def gig_proof_types(doc: Dict[str, Any]) -> List[str]:
    proof = doc.get("type_of_work_proof")
    if isinstance(proof, list) and proof:
        return [str(p) for p in proof]
    stores = doc.get("gig_stores") or []
    if not stores or not isinstance(stores[0], dict):
        return []
    store = stores[0]
    flags = []
    mapping = {
        "requires_order_id": "order_id",
        "requires_receipt": "receipt",
        "requires_review_rating": "review_rating",
        "requires_review_receipt": "review_receipt",
        "requires_review_link": "review_link",
        "requires_tracking_id": "tracking_id",
        "requires_store_address": "store_address",
        "requires_ugc_post": "ugc_post",
    }
    for field, label in mapping.items():
        if store.get(field):
            flags.append(label)
    return flags


def format_gig_item(doc: Dict[str, Any]) -> Dict[str, Any]:
    image = doc.get("image") or ""
    return {
        "_id": str(doc.get("_id", "")),
        "name": doc.get("name"),
        "subtitle": doc.get("subtitle"),
        "description": doc.get("description"),
        "payout": gig_payout(doc),
        "gig_type": doc.get("gig_type"),
        "status": doc.get("status"),
        "start_date": serialize_doc(doc.get("start_date")),
        "end_date": serialize_doc(doc.get("end_date")),
        "proof_type": gig_proof_types(doc),
        "image": image,
        "image_url": image,
        "number_of_people": doc.get("number_of_people"),
        "client_id": str(doc.get("client_id")) if doc.get("client_id") else None,
    }
