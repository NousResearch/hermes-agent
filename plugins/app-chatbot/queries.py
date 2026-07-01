"""CRWD support MongoDB queries — ported from lambdas/support/mongo_tool.py."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from ._utils import (
    active_open_gig_filter,
    format_gig_item,
    get_mongo_db,
    parse_object_id,
    serialize_doc,
    serialize_docs,
)

_OBJECT_ID_IN_TEXT_RE = re.compile(r"\b[0-9a-fA-F]{24}\b")


def get_enrolled_gig_ids(user_id: str) -> Set[str]:
    """Return gig IDs the user is actively enrolled in."""
    oid = parse_object_id(user_id)
    db = get_mongo_db()
    cursor = db.added_crwd_members.find(
        {
            "member": oid,
            "isDeleted": {"$ne": True},
            "$or": [
                {"isAccepted": True},
                {"isApproved": True},
                {"status": {"$in": ["Active", "Accepted", "Approved", "Joined"]}},
            ],
        },
        {"crwd_id": 1},
    )
    enrolled: Set[str] = set()
    for row in cursor:
        crwd_id = row.get("crwd_id")
        if crwd_id is not None:
            enrolled.add(str(crwd_id))
    return enrolled


def get_active_gigs(
    user_id: str,
    page: int = 1,
    limit: int = 10,
) -> Dict[str, Any]:
    """Fetch paginated active/open gigs excluding ones the user already joined."""
    if not user_id:
        return {"success": False, "error": "user_id is required", "items": []}

    page = max(1, int(page))
    limit = max(1, min(int(limit), 50))
    skip = (page - 1) * limit

    enrolled_ids = get_enrolled_gig_ids(user_id)
    enrolled_oids = []
    for gid in enrolled_ids:
        try:
            enrolled_oids.append(parse_object_id(gid))
        except ValueError:
            continue

    db = get_mongo_db()
    query: Dict[str, Any] = active_open_gig_filter()
    if enrolled_oids:
        query["_id"] = {"$nin": enrolled_oids}

    total = db.crwds.count_documents(query)
    docs = list(
        db.crwds.find(query)
        .sort([("createdAt", -1)])
        .skip(skip)
        .limit(limit)
    )
    items = [format_gig_item(doc) for doc in docs]
    return {
        "success": True,
        "items": items,
        "page": page,
        "limit": limit,
        "total": total,
        "has_more": skip + len(items) < total,
        "excluded_enrolled_count": len(enrolled_ids),
    }


def get_user_profile_by_id(user_id: str) -> Dict[str, Any]:
    """Look up a user by MongoDB users._id."""
    oid = parse_object_id(user_id)
    db = get_mongo_db()
    doc = db.users.find_one({"_id": oid, "isDeleted": {"$ne": True}})
    if not doc:
        return {"success": False, "error": f"User not found: {user_id}"}

    profile = serialize_doc(doc)
    return {
        "success": True,
        "user": {
            "_id": profile.get("_id", {}).get("$oid") if isinstance(profile.get("_id"), dict) else str(profile.get("_id", "")),
            "email": profile.get("email"),
            "first_name": profile.get("first_name"),
            "last_name": profile.get("last_name"),
            "full_name": profile.get("full_name"),
            "phone": profile.get("phone"),
            "status": profile.get("status"),
            "role": profile.get("role"),
            "isEmailVerified": profile.get("isEmailVerified"),
            "sign_up_request_status": profile.get("sign_up_request_status"),
            "createdAt": profile.get("createdAt"),
            "updatedAt": profile.get("updatedAt"),
        },
    }


def _find_gig_by_ref(gig_ref: str) -> Optional[Dict[str, Any]]:
    db = get_mongo_db()
    ref = (gig_ref or "").strip()
    if not ref:
        return None

    oid_match = _OBJECT_ID_IN_TEXT_RE.search(ref)
    if oid_match:
        try:
            oid = parse_object_id(oid_match.group(0))
            doc = db.crwds.find_one({"_id": oid, "isDeleted": {"$ne": True}})
            if doc:
                return doc
        except ValueError:
            pass

    active_filter = active_open_gig_filter()
    exact = db.crwds.find_one(
        {**active_filter, "name": {"$regex": f"^{re.escape(ref)}$", "$options": "i"}},
    )
    if exact:
        return exact

    fuzzy = db.crwds.find_one(
        {**active_filter, "name": {"$regex": re.escape(ref), "$options": "i"}},
    )
    if fuzzy:
        return fuzzy

    words = [w for w in re.split(r"\W+", ref) if len(w) >= 3]
    if len(words) >= 2:
        pattern = ".*".join(re.escape(word) for word in words)
        token_match = db.crwds.find_one(
            {**active_filter, "name": {"$regex": pattern, "$options": "i"}},
        )
        if token_match:
            return token_match

    return db.crwds.find_one(
        {"isDeleted": {"$ne": True}, "name": {"$regex": re.escape(ref), "$options": "i"}},
    )


def get_gig_details(gig_id: Optional[str] = None, name: Optional[str] = None) -> Dict[str, Any]:
    """Fetch one gig by Mongo _id, name substring, or fuzzy match."""
    gig_ref = (gig_id or name or "").strip()
    if not gig_ref:
        return {"success": False, "error": "Provide gig_id or name"}

    doc = _find_gig_by_ref(gig_ref if gig_id else name or gig_ref)
    if not doc:
        return {"success": False, "error": f"Gig not found: {gig_ref}"}

    item = format_gig_item(doc)
    item["terms_description"] = doc.get("terms_description")
    item["gig_stores"] = serialize_docs(doc.get("gig_stores") or [])
    item["targeting_rules"] = serialize_docs(doc.get("targeting_rules") or [])
    item["locations"] = serialize_docs(doc.get("locations") or [])
    return {"success": True, "gig": item}


def get_user_gig_history(user_id: str, limit: int = 50) -> Dict[str, Any]:
    """Past gig participation rows for a user."""
    oid = parse_object_id(user_id)
    db = get_mongo_db()
    limit = max(1, min(int(limit), 100))

    rows = list(
        db.added_crwd_members.find({"member": oid})
        .sort([("createdAt", -1)])
        .limit(limit)
    )
    items = []
    for row in rows:
        serialized = serialize_doc(row)
        items.append({
            "_id": serialized.get("_id"),
            "crwd_id": serialized.get("crwd_id"),
            "status": serialized.get("status"),
            "isApproved": serialized.get("isApproved"),
            "isAccepted": serialized.get("isAccepted"),
            "isDeleted": serialized.get("isDeleted"),
            "hasPaid": serialized.get("hasPaid"),
            "rejectionReason": serialized.get("rejectionReason"),
            "rejectionNotes": serialized.get("rejectionNotes"),
            "date": serialized.get("date"),
            "time": serialized.get("time"),
            "createdAt": serialized.get("createdAt"),
            "updatedAt": serialized.get("updatedAt"),
        })

    if not items:
        fallback = list(
            db.gig_participations.find({"user_id": oid})
            .sort([("createdAt", -1)])
            .limit(limit)
        ) if "gig_participations" in db.list_collection_names() else []
        if fallback:
            items = serialize_docs(fallback)

    return {"success": True, "items": items, "count": len(items)}


def get_user_joined_gigs(user_id: str, limit: int = 50) -> Dict[str, Any]:
    """Active memberships joined to full gig documents."""
    oid = parse_object_id(user_id)
    db = get_mongo_db()
    limit = max(1, min(int(limit), 100))

    memberships = list(
        db.added_crwd_members.find(
            {
                "member": oid,
                "isDeleted": {"$ne": True},
                "$or": [
                    {"isAccepted": True},
                    {"isApproved": True},
                    {"status": {"$in": ["Active", "Accepted", "Approved", "Joined"]}},
                ],
            },
        )
        .sort([("createdAt", -1)])
        .limit(limit)
    )

    items = []
    for membership in memberships:
        crwd_id = membership.get("crwd_id")
        gig_doc = db.crwds.find_one({"_id": crwd_id}) if crwd_id else None
        membership_data = serialize_doc(membership)
        entry: Dict[str, Any] = {
            "membership": {
                "_id": membership_data.get("_id"),
                "crwd_id": membership_data.get("crwd_id"),
                "status": membership_data.get("status"),
                "isApproved": membership_data.get("isApproved"),
                "isAccepted": membership_data.get("isAccepted"),
                "hasPaid": membership_data.get("hasPaid"),
                "createdAt": membership_data.get("createdAt"),
            },
        }
        if gig_doc:
            entry["gig"] = format_gig_item(gig_doc)
        items.append(entry)

    return {"success": True, "items": items, "count": len(items)}
