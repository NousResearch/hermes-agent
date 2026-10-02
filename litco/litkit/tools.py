"""The ``litkit`` Hermes toolset: the matter host's hands on the firm's LitKit instance.

Every tool calls LitKit through :class:`litco.litkit.client.LitKitClient`, which sends the
matter-pinned agent token and, during a turn, the acting lawyer's user assertion. What the
lawyer may not do, the tool may not do: a 401/403 comes back as a plain
``not permitted for this user on this matter`` result, never retried and never worked around.

Results are concise JSON. Anything large (search hit lists, census pages, gate findings) is
spilled to a file under the turn's working directory with a preview, following Hermes's
``<persisted-output>`` convention. Document text and PDFs are written to files by design
(``texts/``, ``pdfs/``), each text with a self-citing header so a quoted passage carries its
Bates cite without a second lookup. No tool writes outside the working directory.

Registration lives in ``plugins/litkit`` (a bundled backend plugin); :data:`TOOLS` is the list
it registers.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from urllib.parse import quote
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from litco.homes import register_deliverable, safe_segment
from litco.litkit.client import (TURN_GRANT_HEADER, LitKitClient, LitKitConfig, LitKitError,
                                 LitKitPermissionError, default_client)
from litco.litkit.context import (current_acting_user, current_cross_matter_grant, current_thread_id, current_turn,
                                  current_turn_grant)
from litco.litkit.links import document_link, file_link, folder_link
from litco.litkit.files import (TEXT_SEPARATOR, InputFileMissing, PathOutsideWorkDir, dumps, generate_preview,
                                input_path, output_dir, output_path, relative, spill, work_dir)
from litco.litkit import jev as _jev

TOOLSET = "litkit"
EXPORT_BATCH = 500
SEARCH_MAX = 500
DOCS_PAGE_MAX = 5000
CHANNEL_HISTORY_MAX = 100
CHANNEL_HISTORY_TEXT_MAX = 4000
_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def check_available() -> bool:
    """The toolset is offered only on a host configured for LitKit."""
    try:
        return LitKitConfig.from_env().configured
    except Exception:
        return False


def check_matter_host() -> bool:
    """Local tools (no LitKit call) are offered on any matter host: LitKit configured, or a turn
    server secret / matter id present."""
    if check_available():
        return True
    try:
        from litco.litkit.client import _secret
        return bool(_secret("LITCO_HOST_SECRET") or _secret("LITCO_MATTER_ID"))
    except Exception:
        return False


def _client() -> LitKitClient:
    return default_client()


def _mid(client: LitKitClient) -> str:
    return client.matter_id


def _ok(payload: Any, tool: str) -> str:
    return spill(dumps(payload), tool)


def _fail(message: str, **extra: Any) -> str:
    return dumps({"error": message, **extra})


def _tool(name: str) -> Callable[[Callable[..., Any]], Callable[..., str]]:
    """Wrap a handler: LitKit, validation and path errors become JSON error results."""
    def deco(fn: Callable[..., Any]) -> Callable[..., str]:
        def handler(args: Optional[Dict[str, Any]] = None, **_kw: Any) -> str:
            args = args if isinstance(args, dict) else {}
            try:
                result = fn(args)
            except LitKitPermissionError as exc:
                return dumps(exc.to_dict())
            except LitKitError as exc:
                return dumps(exc.to_dict())
            except PathOutsideWorkDir as exc:
                return _fail(str(exc))
            except InputFileMissing as exc:
                return _fail(str(exc), file_missing=True)
            except (ValueError, TypeError) as exc:
                return _fail(str(exc))
            return result if isinstance(result, str) else _ok(result, name)
        handler.__name__ = f"handle_{name}"
        return handler
    return deco


def _require(args: Dict[str, Any], key: str) -> Any:
    value = args.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"'{key}' is required")
    return value.strip() if isinstance(value, str) else value


def _uuid(args: Dict[str, Any], key: str) -> str:
    value = str(_require(args, key))
    if not _UUID.match(value):
        raise ValueError(f"'{key}' must be a LitKit id (uuid)")
    return value


def _clip(text: Any, limit: int = 300) -> Any:
    if isinstance(text, str) and len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _compact(obj: Dict[str, Any], *, drop: Iterable[str] = (), limit: int = 300) -> Dict[str, Any]:
    """Scalar fields only, long strings clipped, heavy keys dropped."""
    skip = set(drop) | {"extractedText", "contentMd", "storageKey", "metadata", "accessAcl"}
    out: Dict[str, Any] = {}
    for key, value in (obj or {}).items():
        if key in skip:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            if value is None or value == "":
                continue
            out[key] = _clip(value, limit)
    return out


def _linked(row: Dict[str, Any], key: str, build: Callable[..., str], *args: Any, **kw: Any) -> Dict[str, Any]:
    """Add ``row[key]`` when the link builds. A bad id leaves the row as it was, without the key."""
    try:
        link = build(*args, **kw)
    except ValueError:
        return row
    if link:
        row[key] = link
    return row


def _doc_label(doc: Dict[str, Any]) -> str:
    return str(doc.get("batesStart") or doc.get("fileName") or "document")


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _date_of(row: Dict[str, Any]) -> Optional[str]:
    for key in ("date", "documentDate", "dateSent", "dateCreated", "dateModified"):
        if row.get(key):
            return str(row[key])
    return None


def _cap_list(items: Any, n: int = 20) -> Any:
    if isinstance(items, list) and len(items) > n:
        return items[:n] + [f"... {len(items) - n} more (see the saved file)"]
    return items


def _save_json(subdir: str, stem: str, payload: Any) -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    target = output_path(subdir, f"{stem}-{stamp}.json")
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return target


# -- document text files -------------------------------------------------------

def text_filename(bates: Optional[str], doc_id: str) -> str:
    return safe_segment(bates or doc_id) + ".txt"


def text_header(*, doc_id: str, bates_start: Optional[str], bates_end: Optional[str], custodian: Optional[str],
                date: Optional[str], author: Optional[str] = None, subject: Optional[str] = None,
                file_name: Optional[str] = None, matter_id: str = "", route: str = "",
                truncated: Optional[Dict[str, Any]] = None) -> str:
    bates = bates_start or "(no Bates)"
    if bates_end and bates_end != bates_start:
        bates = f"{bates} - {bates_end}"
    lines = [f"Bates: {bates}", f"docId: {doc_id}", f"Custodian: {custodian or ''}", f"Date: {date or ''}"]
    if author:
        lines.append(f"Author: {author}")
    if subject:
        lines.append(f"Subject: {subject}")
    if file_name:
        lines.append(f"File: {file_name}")
    lines.append(f"Source: LitKit matter {matter_id}, {route} (retrieved {_now_iso()})")
    if truncated:
        lines.append(f"Truncated: served {truncated.get('served')} of {truncated.get('full')} characters")
    return "\n".join(lines) + "\n" + TEXT_SEPARATOR + "\n"


def _load_index(folder: Path) -> Dict[str, Any]:
    try:
        data = json.loads((folder / "index.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_index(folder: Path, index: Dict[str, Any]) -> Path:
    target = folder / "index.json"
    tmp = folder / "index.json.tmp"
    tmp.write_text(json.dumps(index, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(target)
    return target


# ---------------------------------------------------------------------------
# matter, search, census
# ---------------------------------------------------------------------------

@_tool("litkit_matter")
def litkit_matter(args: Dict[str, Any]) -> Any:
    client = _client()
    mid = _mid(client)
    record = client.get(f"/api/matters/{mid}")
    matter = _compact(record if isinstance(record, dict) else {},
                      drop=("openaiKeyAvailable", "openrouterKeyAvailable", "chatImageAttachments"))
    counts = client.get(f"/api/matters/{mid}/docs", params={"countOnly": "1"})
    facets = client.get(f"/api/matters/{mid}/search/facets")
    facets = facets if isinstance(facets, dict) else {}
    custodians = facets.get("custodians") or []
    productions = [{"id": p.get("id"), "name": p.get("name"), "status": p.get("status"),
                    "documents": p.get("fileCount")} for p in facets.get("productions") or [] if isinstance(p, dict)]
    return {"matter": matter, "documentCount": (counts or {}).get("total") if isinstance(counts, dict) else None,
            "custodianCount": len(custodians), "custodians": custodians, "productions": productions,
            "batesPrefixes": facets.get("batesPrefixes") or [],
            "actingUser": current_acting_user() or "(none: Matter Agent service role)"}


@_tool("litkit_search")
def litkit_search(args: Dict[str, Any]) -> Any:
    client = _client()
    mid = _mid(client)
    query = str(_require(args, "query"))
    if len(query) > 200:
        raise ValueError("query is limited to 200 characters")
    limit = max(1, min(int(args.get("limit") or 100), SEARCH_MAX))
    params = {"q": query, "limit": limit, "custodian": args.get("custodian"), "dateFrom": args.get("dateFrom"),
              "dateTo": args.get("dateTo"), "bates": args.get("bates"),
              "matchCase": "true" if args.get("matchCase") else None,
              "wholeWord": "true" if args.get("wholeWord") else None}
    try:
        body = client.get(f"/api/matters/{mid}/search", params=params)
    except LitKitError as exc:
        if exc.status == 504:
            return _fail("search timed out (LitKit gives search 5 seconds). Narrow it: a quoted phrase, a Bates "
                         "number, or a custodian/date filter. A timeout is not evidence the documents are absent.",
                         status=504)
        raise
    hits = body.get("hits") if isinstance(body, dict) else None
    hits = hits if isinstance(hits, list) else []
    compact = [{"docId": h.get("docId"), "bates": h.get("bates"), "batesEnd": h.get("batesEnd"),
                "page": h.get("page"), "snippet": _clip(h.get("snippet"), 280),
                **({"metadataOnly": True} if h.get("metadataOnly") else {})} for h in hits if isinstance(h, dict)]
    for hit in compact:  # a hit with Bates is cited by Bates; only the others need a link
        if not hit.get("bates"):
            _linked(hit, "link", document_link, mid, hit.get("docId"), "document", page=hit.get("page"))
    notes = []
    if len(hits) >= limit:
        notes.append(f"hit the {limit}-result limit; results are a mention-finder, not a census. "
                     "Narrow the query or use litkit_docs for a complete list.")
    if isinstance(body, dict) and body.get("searchRanked"):
        notes.append("common term: LitKit returned the best-ranked matches only, not every match.")
    return {"query": query, "hits": len(compact), "documents": len({h["docId"] for h in compact}),
            "notes": notes, "results": compact}


def _doc_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {"id": row.get("id"), "batesStart": row.get("batesStart"), "batesEnd": row.get("batesEnd"),
           "custodian": row.get("custodian"), "date": row.get("documentDate") or row.get("dateSent"),
           "author": _clip(row.get("author"), 120), "subject": _clip(row.get("subject"), 200),
           "fileName": _clip(row.get("fileName"), 160), "pageCount": row.get("pageCount"), "mime": row.get("mime")}
    return {k: v for k, v in out.items() if v not in (None, "")}


def _docs_params(args: Dict[str, Any], limit: int, cursor: str) -> Dict[str, Any]:
    return {"q": args.get("q"), "custodian": args.get("custodian"), "dateFrom": args.get("dateFrom"),
            "dateTo": args.get("dateTo"), "bates": args.get("bates"), "productionId": args.get("productionId"),
            "limit": limit, "cursor": cursor}


@_tool("litkit_docs")
def litkit_docs(args: Dict[str, Any]) -> Any:
    client = _client()
    mid = _mid(client)
    limit = max(1, min(int(args.get("limit") or 1000), DOCS_PAGE_MAX))
    save_as = args.get("saveAs")
    cursor = str(args.get("cursor") or "")

    def page(cur: str) -> Dict[str, Any]:
        try:
            body = client.get(f"/api/matters/{mid}/docs", params=_docs_params(args, limit, cur))
        except LitKitError as exc:
            kind = exc.body.get("errorKind") if isinstance(exc.body, dict) else None
            if exc.status == 422 and "cursor_overbroad" in (exc.code, kind):
                raise ValueError("this text query is too broad to page; add a filter (custodian, dates) or a "
                                 "more distinctive term") from None
            if exc.status == 504:
                raise ValueError("census page timed out (10 s budget); add a custodian or date filter") from None
            raise
        return body if isinstance(body, dict) else {}

    if not save_as:
        body = page(cursor)
        docs = [_doc_row(r) for r in body.get("docs") or [] if isinstance(r, dict)]
        for row in docs:  # a row with Bates is cited by Bates; only the others need a link
            if not row.get("batesStart"):
                _linked(row, "link", document_link, mid, row.get("id"), row.get("fileName") or "document")
        return {"total": body.get("total"), "returned": len(docs), "hasMore": bool(body.get("hasMore")),
                "nextCursor": body.get("nextCursor"), "docs": docs}

    max_pages = max(1, min(int(args.get("maxPages") or 400), 2000))
    total, rows, pages, per_custodian = None, 0, 0, {}
    seen: set = set()
    first = page(cursor)  # fail before creating the census file
    target = output_path("census", f"{safe_segment(str(save_as))}.jsonl")
    with open(target, "w", encoding="utf-8") as fh:
        while pages < max_pages:
            body = first if pages == 0 else page(cursor)
            pages += 1
            if pages == 1:
                total = body.get("total")
            for raw in body.get("docs") or []:
                if not isinstance(raw, dict) or raw.get("id") in seen:
                    continue
                seen.add(raw.get("id"))
                row = _doc_row(raw)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows += 1
                key = row.get("custodian") or "(none)"
                per_custodian[key] = per_custodian.get(key, 0) + 1
            cursor = body.get("nextCursor") or ""
            if not body.get("hasMore") or not cursor:
                cursor = ""
                break
    top = sorted(per_custodian.items(), key=lambda kv: -kv[1])[:25]
    return {"saved": relative(target), "rows": rows, "total": total, "pages": pages,
            "complete": cursor == "", "resumeCursor": cursor or None,
            "byCustodian": dict(top), "custodians": len(per_custodian)}


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------

def _resolve_doc_id(client: LitKitClient, args: Dict[str, Any]) -> str:
    if args.get("documentId"):
        return _uuid(args, "documentId")
    bates = args.get("bates")
    if not bates:
        raise ValueError("give documentId or bates")
    body = client.get(f"/api/matters/{_mid(client)}/bates-resolve", params={"bates": str(bates).strip()})
    doc_id = body.get("documentId") if isinstance(body, dict) else None
    if not doc_id:
        raise ValueError(f"no document in this matter carries Bates {bates}")
    return str(doc_id)


def _metadata(client: LitKitClient, doc_id: str) -> Dict[str, Any]:
    body = client.get(f"/api/documents/{doc_id}")
    doc = body.get("doc") if isinstance(body, dict) else None
    return doc if isinstance(doc, dict) else {}


@_tool("litkit_document")
def litkit_document(args: Dict[str, Any]) -> Any:
    client = _client()
    doc_id = _resolve_doc_id(client, args)
    body = client.get(f"/api/documents/{doc_id}")
    body = body if isinstance(body, dict) else {}
    raw = body.get("doc") if isinstance(body.get("doc"), dict) else {}
    doc = _compact(raw, limit=400)
    tags = [t.get("name") for t in body.get("tags") or [] if isinstance(t, dict) and t.get("name")]
    productions = [_compact(p) for p in body.get("productions") or [] if isinstance(p, dict)]
    result = {"document": doc, "tags": tags, "productions": productions,
              "redactions": len(body.get("redactions") or [])}
    return _linked(result, "link", document_link, _mid(client), doc_id, _doc_label(raw))


def _text_from_payload(body: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
    chunks = body.get("chunks") if isinstance(body.get("chunks"), list) else []
    if chunks:
        parts: List[str] = []
        last_page = None
        for ch in chunks:
            if not isinstance(ch, dict):
                continue
            page = ch.get("pageStart")
            if page is not None and page != last_page:
                parts.append(f"[page {page}]")
                last_page = page
            parts.append(str(ch.get("text") or ""))
        return "\n\n".join(parts), None
    text = str(body.get("extractedText") or "")
    capped = body.get("extractedTextCapped")
    trunc = ({"served": capped.get("servedLength"), "full": capped.get("fullLength")}
             if isinstance(capped, dict) else None)
    return text, trunc


@_tool("litkit_text")
def litkit_text(args: Dict[str, Any]) -> Any:
    client = _client()
    mid = _mid(client)
    doc_id = _resolve_doc_id(client, args)
    meta = _metadata(client, doc_id)
    body = client.get(f"/api/documents/{doc_id}/text")
    text, trunc = _text_from_payload(body if isinstance(body, dict) else {})
    header = text_header(doc_id=doc_id, bates_start=meta.get("batesStart"), bates_end=meta.get("batesEnd"),
                         custodian=meta.get("custodian"), date=_date_of(meta), author=meta.get("author"),
                         subject=meta.get("subject"), file_name=meta.get("fileName"), matter_id=mid,
                         route=f"GET /api/documents/{doc_id}/text", truncated=trunc)
    target = output_path(args.get("dir") or "texts", text_filename(meta.get("batesStart"), doc_id))
    target.write_text(header + text, encoding="utf-8")
    preview, more = generate_preview(text, int(args.get("previewChars") or 1500))
    result = {"saved": relative(target), "docId": doc_id, "bates": meta.get("batesStart"),
              "chars": len(text), "empty": not text.strip(), "preview": preview + ("\n..." if more else "")}
    if trunc:
        result["truncated"] = trunc
    if not text.strip():
        result["note"] = ("LitKit holds no extracted text for this document (image-only or native-only); "
                          "fetch the PDF with litkit_pdf and read it visually.")
    return _linked(result, "link", document_link, mid, doc_id, _doc_label(meta))


@_tool("litkit_pdf")
def litkit_pdf(args: Dict[str, Any]) -> Any:
    client = _client()
    doc_id = _resolve_doc_id(client, args)
    native = bool(args.get("native"))
    try:
        meta = _metadata(client, doc_id)
    except LitKitError:
        meta = {}
    stem = safe_segment(meta.get("batesStart") or doc_id)
    if native:
        ext = Path(str(meta.get("fileName") or "")).suffix or ".bin"
        target = output_path(args.get("dir") or "natives", stem + "." + safe_segment(ext.lstrip(".")))
        info = client.download(f"/api/documents/{doc_id}/native", target)
    else:
        target = output_path(args.get("dir") or "pdfs", stem + ".pdf")
        info = client.download(f"/api/documents/{doc_id}/pdf", target)
    head = info.pop("head", b"")
    result = {"saved": relative(target), "docId": doc_id, "bates": meta.get("batesStart"), "bytes": info["bytes"],
              "sha256": info["sha256"], "contentType": info["contentType"]}
    if not native and not head.startswith(b"%PDF"):
        result["warning"] = "the bytes do not start with %PDF; inspect before relying on this file"
    if not native and meta.get("pageCount") == 1 and str(meta.get("mime") or "").find("sheet") >= 0:
        result["note"] = "a one-page PDF of a spreadsheet is usually a slip sheet; fetch the native as well"
    return _linked(result, "link", document_link, _mid(client), doc_id, _doc_label(meta))


def _ids_from_census(raw_path: str) -> List[str]:
    path = input_path(raw_path)
    ids: List[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            value = (row.get("id") or row.get("docId")) if isinstance(row, dict) else None
            if value:
                ids.append(str(value))
    return ids


@_tool("litkit_export_text")
def litkit_export_text(args: Dict[str, Any]) -> Any:
    client = _client()
    mid = _mid(client)
    ids: List[str] = []
    if args.get("documentIds"):
        if not isinstance(args["documentIds"], list):
            raise ValueError("documentIds must be a list of ids")
        ids = [str(i).strip() for i in args["documentIds"] if str(i).strip()]
    if args.get("fromCensus"):
        ids.extend(_ids_from_census(str(args["fromCensus"])))
    ids = list(dict.fromkeys(i for i in ids if _UUID.match(i)))
    if not ids:
        raise ValueError("give documentIds or fromCensus (a census .jsonl written by litkit_docs)")
    folder = output_dir(args.get("dir") or "texts")
    index = _load_index(folder)
    overwrite = bool(args.get("overwrite"))
    todo = [i for i in ids if overwrite or not (i in index and (folder / index[i].get("file", "")).is_file())]
    stats = {"requested": len(ids), "skippedExisting": len(ids) - len(todo), "written": 0, "notFound": 0,
             "truncated": 0, "empty": 0, "errors": 0, "batches": 0}
    not_found: List[str] = []
    for start in range(0, len(todo), EXPORT_BATCH):
        batch = todo[start:start + EXPORT_BATCH]
        stats["batches"] += 1
        for line in client.stream_ndjson("POST", f"/api/matters/{mid}/export/text", json_body={"docIds": batch},
                                         timeout=600):
            doc_id = line.get("docId")
            if line.get("error"):
                if line["error"] == "not_found" and doc_id:
                    stats["notFound"] += 1
                    not_found.append(str(doc_id))
                else:
                    stats["errors"] += 1
                continue
            if not doc_id:
                continue
            text = str(line.get("text") or "")
            trunc = {"served": len(text), "full": line.get("fullLength")} if line.get("truncated") else None
            name = text_filename(line.get("batesStart"), str(doc_id))
            header = text_header(doc_id=str(doc_id), bates_start=line.get("batesStart"),
                                 bates_end=line.get("batesEnd"), custodian=line.get("custodian"),
                                 date=line.get("date"), matter_id=mid,
                                 route=f"POST /api/matters/{mid}/export/text", truncated=trunc)
            (folder / name).write_text(header + text, encoding="utf-8")
            index[str(doc_id)] = {"file": name, "bates": line.get("batesStart"), "batesEnd": line.get("batesEnd"),
                                  "custodian": line.get("custodian"), "date": line.get("date"),
                                  "textLen": len(text), **({"truncated": True} if trunc else {})}
            stats["written"] += 1
            stats["truncated"] += 1 if trunc else 0
            stats["empty"] += 0 if text.strip() else 1
        _write_index(folder, index)  # after every batch, so an interrupted run resumes
    _write_index(folder, index)
    result: Dict[str, Any] = {"dir": relative(folder), "index": relative(folder / "index.json"), **stats}
    if not_found:
        result["notFoundIds"] = _cap_list(not_found, 10)
    if stats["empty"]:
        result["note"] = f"{stats['empty']} documents have no extracted text; read their PDFs (litkit_pdf)."
    return result


# ---------------------------------------------------------------------------
# memos, LitSpace files, attachments
# ---------------------------------------------------------------------------

@_tool("litkit_memos")
def litkit_memos(args: Dict[str, Any]) -> Any:
    client = _client()
    mid = _mid(client)
    action = str(args.get("action") or "list")
    if action == "list":
        body = client.get("/api/matter-files", params={"matterId": mid, "kind": "memo"})
        files = body.get("files") if isinstance(body, dict) else []
        rows = [{"id": f.get("id"), "title": _clip(f.get("title") or f.get("name"), 160), "source": f.get("source"),
                 "createdAt": f.get("createdAt"), "refDocs": len(f.get("refDocIds") or [])}
                for f in files or [] if isinstance(f, dict) and f.get("entryType") != "folder"]
        return {"memos": len(rows), "results": rows}
    if action == "read":
        file_id = _uuid(args, "fileId")
        body = client.get(f"/api/matter-files/{file_id}")
        body = body if isinstance(body, dict) else {}
        text = str(body.get("contentMd") or body.get("extractedText") or "")
        title = str(body.get("title") or body.get("name") or file_id)
        target = output_path("memos", f"{safe_segment(title)[:80]}-{file_id[:8]}.md")
        target.write_text(f"# {title}\n\n<!-- LitKit matter file {file_id}, retrieved {_now_iso()} -->\n\n{text}",
                          encoding="utf-8")
        preview, more = generate_preview(text)
        return {"saved": relative(target), "title": title, "chars": len(text),
                "refDocIds": _cap_list(body.get("refDocIds") or [], 25), "preview": preview + ("\n..." if more else "")}
    raise ValueError("action must be list or read")


def _litspace_matter_id(client: LitKitClient) -> Optional[str]:
    body = client.get(f"/api/litspace/matters/{_mid(client)}/files", params={"limit": 1})
    files = body.get("files") if isinstance(body, dict) else None
    if isinstance(files, list) and files and isinstance(files[0], dict):
        return files[0].get("litspaceMatterId")
    return None


def _filename_from_disposition(value: str) -> Optional[str]:
    match = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)\"?", value or "", re.I)
    if not match:
        return None
    from urllib.parse import unquote
    return unquote(match.group(1)).strip() or None


@_tool("litkit_files")
def litkit_files(args: Dict[str, Any]) -> Any:
    client = _client()
    mid = _mid(client)
    action = str(args.get("action") or "list")
    if action == "list":
        params = {"limit": int(args.get("limit") or 200), "cursor": args.get("cursor") or None}
        body = client.get(f"/api/litspace/matters/{mid}/files", params=params)
        files = body.get("files") if isinstance(body, dict) else []
        rows = [_linked({"fileId": f.get("litspaceFileId"), "filename": f.get("filename"), "mime": f.get("mime"),
                         "bytes": f.get("bytes"), "updatedAt": f.get("updatedAt")},
                        "link", file_link, f.get("litspaceMatterId"), f.get("litspaceFileId"),
                        f.get("filename") or "file")
                for f in files or [] if isinstance(f, dict)]
        return {"files": len(rows), "nextCursor": (body or {}).get("nextCursor"), "results": rows}
    if action == "search":
        query = str(_require(args, "query"))
        if len(query) < 2:
            raise ValueError("query needs at least 2 characters")
        ls_id = _litspace_matter_id(client)
        if not ls_id:
            return {"query": query, "results": [], "note": "the matter's LitSpace holds no files"}
        body = client.get(f"/api/litspace/matters/{ls_id}/search", params={"q": query})
        rows = [{"fileId": r.get("documentId"), "filename": r.get("filename"), "folder": r.get("folderDisplay"),
                 "snippet": _clip(r.get("snippet"), 280), **({"seekSec": r["seekSec"]} if "seekSec" in r else {})}
                for r in (body or {}).get("rows") or [] if isinstance(r, dict)]
        for row in rows:
            _linked(row, "link", file_link, ls_id, row["fileId"], row.get("filename") or "file")
            if row.get("folder"):
                _linked(row, "folderLink", folder_link, ls_id, row["folder"])
        return {"query": query, "hits": len(rows), "results": rows}
    if action == "read":
        file_id = _uuid(args, "fileId")
        meta = client.get(f"/api/litspace/files/{file_id}")
        meta = meta if isinstance(meta, dict) else {}
        name = meta.get("filename")
        target = output_path(args.get("dir") or "files", name or file_id)
        info = client.download(f"/api/litspace/files/{file_id}/content", target, params={"op": "download"})
        info.pop("head", None)
        result = {"saved": relative(target), "fileId": file_id, "bytes": info["bytes"], "sha256": info["sha256"],
                  "contentType": info["contentType"]}
        if meta.get("litspaceMatterId"):
            _linked(result, "link", file_link, meta["litspaceMatterId"], file_id, name or "file")
        return result
    if action == "upload":
        path = input_path(str(_require(args, "path")), tool="litkit_files")
        status, body = client.upload(f"/api/litspace/matters/{mid}/files/upload", path,
                                     fields={"filename": args.get("filename"), "parentId": args.get("parentId")},
                                     filename=args.get("filename") or path.name)
        return {"uploaded": path.name, "status": status, "result": body}
    raise ValueError("action must be list, search, read or upload")


@_tool("litkit_attachment")
def litkit_attachment(args: Dict[str, Any]) -> Any:
    client = _client()
    mid = _mid(client)
    file_id = _uuid(args, "fileId")
    folder = output_dir(f"inbox/{safe_segment(file_id)}")
    tmp_name = safe_segment(args.get("filename") or file_id)
    target = folder / tmp_name
    info = client.download(f"/api/matters/{mid}/chat/attachments/{file_id}", target)
    info.pop("head", None)
    served = _filename_from_disposition(info.get("contentDisposition", ""))
    if served and not args.get("filename") and safe_segment(served) != tmp_name:
        final = folder / safe_segment(served)
        target.replace(final)
        target = final
    return {"saved": relative(target), "fileId": file_id, "bytes": info["bytes"], "sha256": info["sha256"],
            "contentType": info["contentType"]}


# ---------------------------------------------------------------------------
# deliverables and quote checks
# ---------------------------------------------------------------------------

def _gate_summary(gate: Any) -> Dict[str, Any]:
    if not isinstance(gate, dict):
        return {}
    quote = gate.get("quote") if isinstance(gate.get("quote"), dict) else {}
    out = {
        "blockedBy": gate.get("blockedBy"), "blockedReason": gate.get("blockedReason"),
        "quotesChecked": quote.get("checked"), "quotesVerified": quote.get("verifiedTokens"),
        "quotesUnverified": _cap_list(quote.get("unverified") or [], 15),
        "quotesHardFail": _cap_list(quote.get("hardFail") or [], 15),
        "quoteGateError": gate.get("quoteGateError"),
        "pleadingFlags": _cap_list(gate.get("pleadingFlags") or [], 15),
        "citationFlags": _cap_list(gate.get("citationFlags") or [], 15),
        "proseLintFlags": _cap_list(gate.get("proseLintFlags") or [], 15),
        "bytesRewritten": gate.get("bytesRewritten"),
    }
    if gate.get("quoteGateError"):
        out["quoteGateNote"] = "the quote pass could not run: quotations are UNCHECKED, not verified"
    return {k: v for k, v in out.items() if v not in (None, [], "")}


@_tool("litkit_deliver")
def litkit_deliver(args: Dict[str, Any]) -> Any:
    path = input_path(str(_require(args, "path")), tool="litkit_deliver")
    klass = str(_require(args, "deliverableClass"))
    client = _client()
    mid = _mid(client)
    provenance = args.get("provenance")
    if provenance is not None and not isinstance(provenance, (dict, str)):
        raise ValueError("provenance must be an object")
    fields = {"deliverableClass": klass, "filename": args.get("filename"), "folder": args.get("folder"),
              "note": args.get("note"),
              "documentId": _uuid(args, "documentId") if args.get("documentId") else None,
              "provenance": provenance}
    status, body = client.upload(f"/api/matters/{mid}/deliverables", path, fields=fields,
                                 filename=args.get("filename") or path.name, ok_statuses=(422,), timeout=600)
    body = body if isinstance(body, dict) else {}
    saved = _save_json("qa", f"deliver-{safe_segment(path.stem)}", {"status": status, "response": body})
    if status == 422 and body.get("error") == "validity_gate_failed":
        return {"blocked": True, "reason": "validity_gate_failed", "detail": body.get("detail"),
                "fullFindings": relative(saved),
                "instruction": "The file would not open. Rebuild it from its source and try again."}
    if status == 422 or body.get("blocked"):
        return {"blocked": True, "gate": _gate_summary(body.get("gate")), "fullFindings": relative(saved),
                "instruction": ("Nothing was committed. Report these gate findings to the user. Do not resubmit the "
                                "same file unchanged; fix the flagged passages (or ask the user) first.")}
    return {"blocked": False, "documentId": body.get("documentId"), "versionId": body.get("versionId"),
            "versionNumber": body.get("versionNumber"), "path": body.get("path"), "store": body.get("store"),
            "sha256": body.get("sha256"), "sizeBytes": body.get("sizeBytes"), "gate": _gate_summary(body.get("gate")),
            "fullFindings": relative(saved)}


@_tool("litkit_quote_check")
def litkit_quote_check(args: Dict[str, Any]) -> Any:
    client = _client()
    mid = _mid(client)
    if args.get("text"):
        text = str(args["text"])
        if len(text) > 50_000:
            raise ValueError("pasted text is limited to 50,000 characters; check the file instead")
        body = client.post(f"/api/matters/{mid}/quote-check", {"responseText": text}, idempotent=True)
        stem = "quote-check-text"
    else:
        path = input_path(str(_require(args, "path")), tool="litkit_quote_check")
        _, body = client.upload(f"/api/matters/{mid}/quote-check/file", path,
                                fields={"deliverableClass": args.get("deliverableClass")}, timeout=600)
        stem = f"quote-check-{safe_segment(path.stem)}"
    body = body if isinstance(body, dict) else {"result": body}
    saved = _save_json("qa", stem, body)
    out = {k: body.get(k) for k in ("deliverableClass", "checked", "verified", "wouldBlock", "blockReason")
           if k in body}
    for key in ("unverified", "hardFail"):
        if key in body:
            out[key] = _cap_list(body.get(key) or [], 20)
    if not ("checked" in body or "verified" in body):
        out["result"] = body
    out["fullFindings"] = relative(saved)
    return out


DELIVERABLE_CLASSES = ["draft", "filing", "production", "pleading", "brief", "memo", "work_product", "complaint",
                       "motion", "letter", "client_memo"]


@_tool("litco_deliver_local")
def litco_deliver_local(args: Dict[str, Any]) -> Any:
    """Attach a file to this turn's reply. No LitKit call: the turn server lists it in
    ``final.deliverables`` and the app fetches it from the host."""
    turn = current_turn()
    if turn is None or not turn.turn_id:
        raise ValueError("litco_deliver_local works only inside a turn; there is no thread to deliver to")
    path = input_path(str(_require(args, "path")), tool="litco_deliver_local")
    klass = args.get("deliverableClass")
    if klass is not None and klass not in DELIVERABLE_CLASSES:
        raise ValueError(f"deliverableClass must be one of {', '.join(DELIVERABLE_CLASSES)}")
    name = args.get("name")
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise ValueError("name must be a file name")
    entry = register_deliverable(turn.turn_id, work_dir(), path, name=name.strip() if name else None,
                                 deliverable_class=klass)
    return {"registered": True, "filename": entry.filename, "path": relative(entry.path),
            "deliverableClass": entry.deliverable_class,
            "message": f"{entry.filename} will be attached to this turn's reply"}


# ---------------------------------------------------------------------------
# review, ingest, proposals, tags, work sets
# ---------------------------------------------------------------------------

REVIEW_ACTIONS = ("list", "status", "records", "resume", "cancel", "pause", "create", "launch", "withdraw",
                  "proposal", "criteria", "work_sets", "accept_tags")
CRITERIA_ACTIONS = ("list", "get", "create", "update", "versions")
CRITERIA_SET_SCOPES = ("user", "firm")
REVIEW_TIERS = ("fast", "medium", "high")
WORK_SET_ROLES = ("inbox", "outbox")
CRITERION_KEYS = ("key", "title", "description", "seedQuery", "tagName", "tagId", "disposition")
CRITERION_DISPOSITIONS = ("apply", "propose", "propose_with_ambiguous")
SCOPE_KINDS = ("workSetId", "documentIds", "filter", "batesRange")
REVIEW_MAX_CRITERIA = 100
REVIEW_MAX_TAGS = 25
REVIEW_MAX_OPTIMIZATIONS = 5
_SET_ID = re.compile(r"^builtin:[a-z0-9_-]{1,40}$")


def _set_id(args: Dict[str, Any]) -> str:
    """A criteria set id: a LitKit uuid, or a built-in id such as ``builtin:privilege``."""
    value = str(_require(args, "criteriaSetId"))
    if not (_UUID.match(value) or _SET_ID.match(value)):
        raise ValueError("'criteriaSetId' must be a criteria set id (uuid) or a built-in id such as builtin:privilege")
    return value


def _criteria(value: Any) -> List[Dict[str, Any]]:
    """Criteria as LitKit stores them: one object per criterion, each with a title. A bare string is a title."""
    if not isinstance(value, list) or not value:
        raise ValueError("'criteria' must be a non-empty list of criteria ({title, description, tagName, ...})")
    if len(value) > REVIEW_MAX_CRITERIA:
        raise ValueError(f"a criteria set holds at most {REVIEW_MAX_CRITERIA} criteria")
    out: List[Dict[str, Any]] = []
    for n, item in enumerate(value, 1):
        item = {"title": item} if isinstance(item, str) else item
        if not isinstance(item, dict):
            raise ValueError(f"criterion {n} must be an object with a title")
        row = {k: item[k] for k in CRITERION_KEYS if item.get(k) not in (None, "")}
        title = str(row.get("title") or "").strip()
        if not title:
            raise ValueError(f"criterion {n} needs a title")
        row["title"] = title[:200]
        if "description" in row:
            row["description"] = str(row["description"])[:8000]
        if row.get("disposition") and row["disposition"] not in CRITERION_DISPOSITIONS:
            raise ValueError(f"criterion {n}: disposition must be one of {', '.join(CRITERION_DISPOSITIONS)}")
        out.append(row)
    return out


def _review_scope(value: Any) -> Dict[str, Any]:
    """Exactly one of workSetId, documentIds, filter, batesRange."""
    if not isinstance(value, dict):
        raise ValueError("'scope' must be an object with one of: " + ", ".join(SCOPE_KINDS))
    given = [k for k in SCOPE_KINDS if value.get(k) not in (None, "", [], {})]
    if len(given) != 1:
        raise ValueError("'scope' takes exactly one of: " + ", ".join(SCOPE_KINDS))
    kind = given[0]
    if kind == "workSetId":
        return {"workSetId": _uuid(value, "workSetId")}
    if kind == "documentIds":
        docs = value["documentIds"]
        if not isinstance(docs, list):
            raise ValueError("scope.documentIds must be a list of LitKit ids (uuid)")
        docs = [str(d).strip() for d in docs]
        bad = [d for d in docs if not _UUID.match(d)]
        if bad:
            raise ValueError(f"scope.documentIds must be LitKit ids (uuid); not ids: {bad[:3]}")
        return {"documentIds": list(dict.fromkeys(docs))}
    if kind == "filter":
        if not isinstance(value["filter"], dict):
            raise ValueError("scope.filter must be an object of review-grid filters (custodian, dateFrom, query, ...)")
        return {"filter": value["filter"]}
    rng = value["batesRange"]
    if not isinstance(rng, dict) or not str(rng.get("start") or "").strip() or not str(rng.get("end") or "").strip():
        raise ValueError("scope.batesRange must be {start, end}, e.g. {start: 'ABC0000001', end: 'ABC0004000'}")
    return {"batesRange": {"start": str(rng["start"]).strip(), "end": str(rng["end"]).strip()}}


def _review_tags(value: Any) -> List[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("'tags' must be a non-empty list of tag names: the only tags the run may write (or leave "
                         "it out: the criteria rows name the tags)")
    names = list(dict.fromkeys(str(t).strip() for t in value if str(t).strip()))
    if not names:
        raise ValueError("'tags' must name at least one tag")
    if len(names) > REVIEW_MAX_TAGS:
        raise ValueError(f"a run may write at most {REVIEW_MAX_TAGS} tags")
    too_long = [t for t in names if len(t) > 120]
    if too_long:
        raise ValueError(f"tag names are limited to 120 characters: {too_long[0][:40]}...")
    return names


def _first_pass(value: Any) -> Dict[str, Any]:
    """``{enabled, thresholds?: {low, priv}}`` for the propose body's ``firstPass`` (a Jev screen)."""
    if not isinstance(value, dict) or not isinstance(value.get("enabled"), bool):
        raise ValueError("'firstPass' must be {enabled: true|false, thresholds?: {low, priv}}")
    unknown = sorted(set(value) - {"enabled", "thresholds"})
    if unknown:
        raise ValueError(f"'firstPass' takes only enabled and thresholds; not {', '.join(unknown)}")
    out: Dict[str, Any] = {"enabled": value["enabled"]}
    if value.get("thresholds") is not None:
        given = value["thresholds"]
        limits = _jev.thresholds(given)
        out["thresholds"] = {k: limits[k] for k in ("low", "priv") if isinstance(given, dict) and k in given}
    return out


def _optimizations(value: Any) -> List[str]:
    """``optimizationsConsidered``: the short notes Ana weighed before proposing (shown on the Launch card)."""
    if not isinstance(value, list):
        raise ValueError("'optimizations_considered' must be a list of short strings")
    notes = list(dict.fromkeys(str(v).strip() for v in value if str(v).strip()))
    if len(notes) > REVIEW_MAX_OPTIMIZATIONS:
        raise ValueError(f"'optimizations_considered' takes at most {REVIEW_MAX_OPTIMIZATIONS} entries")
    too_long = [n for n in notes if len(n) > 200]
    if too_long:
        raise ValueError(f"each optimization is limited to 200 characters: {too_long[0][:40]}...")
    return notes


def _review_create(client: LitKitClient, mid: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Propose a review run. LitKit posts the proposal's card in the thread; Ana launches it (action=launch)
    after the person says yes there."""
    body: Dict[str, Any] = {"scope": _review_scope(args.get("scope"))}
    has_set, has_criteria = bool(args.get("criteriaSetId")), args.get("criteria") is not None
    if has_set == has_criteria:
        raise ValueError("give exactly one of criteriaSetId (a registered set, preferred) or criteria")
    if has_set:
        body["criteriaSetId"] = _set_id(args)
        if args.get("criteriaSetVersion") is not None:
            version = args["criteriaSetVersion"]
            if isinstance(version, bool) or not isinstance(version, int) or version < 1:
                raise ValueError("'criteriaSetVersion' must be a version number (1 or more)")
            body["criteriaSetVersion"] = version
    else:
        if args.get("criteriaSetVersion") is not None:
            raise ValueError("'criteriaSetVersion' goes with criteriaSetId")
        body["criteria"] = _criteria(args.get("criteria"))
    # The rows name the tags; when tags are sent, LitKit refuses a list that differs from the rows'.
    if args.get("tags") is not None:
        body["tags"] = _review_tags(args.get("tags"))
    for key in ("createMissingTags", "applyTags", "includeRationaleNotes"):
        if args.get(key) is not None:
            body[key] = bool(args[key])
    if args.get("tier") is not None:
        if args["tier"] not in REVIEW_TIERS:
            raise ValueError(f"'tier' must be one of {', '.join(REVIEW_TIERS)}")
        body["tier"] = args["tier"]
    if args.get("name"):
        body["name"] = str(args["name"]).strip()[:200]
    # Omitted unless Ana sets it, so the server's default applies (on when Jev is configured).
    first_pass = args.get("firstPass", args.get("first_pass"))
    if first_pass is not None:
        body["firstPass"] = _first_pass(first_pass)
    optimizations = args.get("optimizations_considered", args.get("optimizationsConsidered"))
    if optimizations is not None:
        notes = _optimizations(optimizations)
        if notes:
            body["optimizationsConsidered"] = notes
    quote_id = str(args.get("quoteId") or "").strip()
    if args.get("userConfirmed") and not quote_id:
        raise ValueError("userConfirmed goes with the quoteId from the price quote; send both after the person's "
                         "explicit yes")
    if quote_id:
        body["quoteId"] = safe_segment(quote_id)
    if args.get("userConfirmed"):
        body["userConfirmed"] = True
    # The thread this turn runs in, so LitKit posts the Launch card there (ana-review-contract).
    # LitKit takes only a thread uuid; any other session id (none outside a turn) is left out,
    # and the proposal then waits in the proposals queue.
    thread_id = current_thread_id()
    if thread_id and _UUID.match(thread_id):
        body["threadId"] = thread_id
    result = client.post(f"/api/matters/{mid}/review-jobs/propose", body)
    result = result if isinstance(result, dict) else {"result": result}
    quote = result.get("quote") if isinstance(result.get("quote"), dict) else {}
    if result.get("requiresApproval"):
        if result.get("needsSecondApprover"):
            result["next"] = (f"Nothing is proposed yet. The price is at or above the firm's approver threshold, so a "
                              f"different matter admin must approve quote {quote.get('id')} in LitKit billing. Tell "
                              "the person that; once it is approved, call create again with the same quoteId.")
        else:
            result["next"] = ("Nothing is proposed yet. Present this price to the person (amount and document count) "
                              "and ask whether to proceed. Only after an explicit yes, call create again with the "
                              f"same arguments plus quoteId={quote.get('id')} and userConfirmed=true.")
        return result
    proposal = result.get("proposal") if isinstance(result.get("proposal"), dict) else {}
    if proposal.get("id"):
        result["proposalId"] = proposal["id"]
    result["next"] = ("Proposed, not launched. Tell the person what you proposed: the scope, the criteria set and "
                      "version, the tags, the estimated document count and cost. Then ask whether to launch. When "
                      "they answer yes in this thread, call action=launch with this proposalId. Do not launch in this "
                      "turn; LitKit refuses until they have answered.")
    if body.get("userConfirmed"):
        # The person's yes to the price may already have told Ana to launch (Hermes ruling 4): one yes, not two.
        result["next"] = ("Proposed, not launched. If the person's yes to the price also told you to launch this run, "
                          "call action=launch with this proposalId now; LitKit decides from the thread whether that "
                          "message covers the launch. Otherwise tell them what you proposed (scope, criteria set and "
                          "version, tags, document count and cost) and ask whether to launch.")
    return result


LAUNCH_NOT_AVAILABLE = ("This LitKit cannot take a launch from the conversation yet. The Launch button on the card "
                        "in this thread starts the run.")
LAUNCH_ASK = ("Nothing launched. Ask the person, in one sentence, whether to launch this run, and call launch only "
              "after their answer in this thread says yes.")
LAUNCH_NO_GRANT = ("Nothing launched. LitKit could not tie this launch to the person's message in this turn. Ask "
                   "whether to launch, and launch in the turn that answers their yes.")
LAUNCH_REFUSALS = {
    "no_reply_after_card": LAUNCH_ASK,
    "reply_from_another_person": LAUNCH_ASK,
    "reply_edited": LAUNCH_ASK,
    "later_reply_in_thread": ("Nothing launched. Someone wrote in the thread after the message you are answering. Read "
                              "it and answer it, then ask whether to launch."),
    "needs_confirmation": ("Nothing launched. The person's reply was not a plain yes to this run. Answer what they "
                           "said; if they still want the run, ask them to confirm with a short yes."),
    "ambiguous_proposal": ("Nothing launched. More than one proposed run is waiting in this thread, so a yes does not "
                           "say which. Withdraw the proposals you replaced, then ask again."),
    "authorization_already_used": ("Nothing launched. That message already launched a run, and one message launches "
                                   "one run. Ask whether to launch this one as well."),
    "not_this_thread": ("Nothing launched. This run was proposed in another thread and can be launched only from "
                        "there. If the person wants it run from here, propose it again in this thread."),
    "needs_reproposal": ("Nothing launched. The scope, criteria or tags changed since the person saw the estimate. "
                         "Withdraw this proposal, propose again, report the new estimate, and ask."),
}


def _route_missing(status: int, body: Any) -> bool:
    """An app that predates the route: 405, or a 404 that is not one of the route's own JSON answers."""
    return status == 405 or (status == 404 and not (isinstance(body, dict) and body.get("error")))


def _refusal(proposal_id: str, body: Any) -> Dict[str, Any]:
    """A 409 from launch or withdraw: LitKit's sentence for the model, as a plain result, not an error."""
    body = body if isinstance(body, dict) else {}
    code = str(body.get("error") or "conflict")
    message = body.get("message") or _error_text(body) or "LitKit refused (HTTP 409)"
    out: Dict[str, Any] = {"proposalId": proposal_id, "refused": code, "message": message}
    if body.get("status"):
        out["proposalStatus"] = body["status"]
    if isinstance(body.get("launched"), dict):
        out["launchedRun"] = body["launched"]
    for key in ("reason", "estimatedDocCount", "currentDocCount"):
        if body.get(key) is not None:
            out[key] = body[key]
    out["next"] = LAUNCH_REFUSALS.get(code) or f"Nothing changed. {message}"
    return out


def _review_launch(client: LitKitClient, mid: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Launch a proposed run. LitKit decides from the thread whether the person said to; nothing here asserts it."""
    proposal_id = _uuid(args, "proposalId")
    thread_id = current_thread_id()
    if not thread_id or not _UUID.match(thread_id):
        raise ValueError("A run can be launched only from the thread it was proposed in.")
    if not current_acting_user():
        raise ValueError("A run can be launched only for the person who said yes in the thread, and this turn has "
                         "no lawyer on it.")
    # The app-minted turn grant ties the launch to the post this turn answers; LitKit requires it.
    grant = current_turn_grant()
    try:
        status, body = client.json_with_status("POST", f"/api/matters/{mid}/proposals/{proposal_id}/launch",
                                               json_body={"threadId": thread_id}, ok_statuses=(404, 405, 409),
                                               extra_headers={TURN_GRANT_HEADER: grant} if grant else None)
    except LitKitPermissionError as exc:
        if not str(exc.code or "").startswith("turn_grant"):
            raise
        return {"proposalId": proposal_id, "launched": False, "refused": exc.code,
                "message": (exc.body or {}).get("message") if isinstance(exc.body, dict) else None,
                "next": LAUNCH_NO_GRANT}
    if _route_missing(status, body):
        return {"proposalId": proposal_id, "launched": False, "message": LAUNCH_NOT_AVAILABLE,
                "next": LAUNCH_NOT_AVAILABLE}
    if status == 404:
        raise LitKitError(f"not found, or not visible to this user (HTTP 404: {_error_text(body)})", status=404,
                          body=body)
    if status == 409:
        return {"launched": False, **_refusal(proposal_id, body)}
    launched = body.get("launched") if isinstance(body, dict) and isinstance(body.get("launched"), dict) else {}
    job_id = launched.get("reviewJobId")
    if not job_id:
        return {"proposalId": proposal_id, "launched": False, "result": body,
                "next": "LitKit returned no job id, so nothing can be reported as launched. Check with "
                        "action=proposal before saying anything about the run."}
    already = bool(body.get("alreadyLaunched"))
    out: Dict[str, Any] = {"proposalId": proposal_id, "launched": True, "reviewJobId": job_id,
                           "scopeDocCount": launched.get("scopeDocCount")}
    if already:
        out["alreadyLaunched"] = True
        out["next"] = ("This run was already launched; no second run started. Tell the person it is running, and "
                       "follow it with status and records.")
    else:
        out["next"] = ("Launched. Tell the person the run started and how many documents it covers (this count is "
                       "the one taken at launch). Follow it with status and records, and report when it finishes.")
    return out


def _review_withdraw(client: LitKitClient, mid: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Withdraw a pending proposal, so a later yes cannot launch a card that was replaced."""
    proposal_id = _uuid(args, "proposalId")
    status, body = client.json_with_status("POST", f"/api/matters/{mid}/proposals/{proposal_id}/withdraw",
                                           json_body={}, ok_statuses=(404, 405, 409))
    if _route_missing(status, body):
        return {"proposalId": proposal_id, "withdrawn": False,
                "next": "This LitKit cannot withdraw a proposal from the conversation yet. Tell the person which "
                        "proposal replaces it."}
    if status == 404:
        raise LitKitError(f"not found, or not visible to this user (HTTP 404: {_error_text(body)})", status=404,
                          body=body)
    if status == 409:
        return {"withdrawn": False, **_refusal(proposal_id, body)}
    return {"proposalId": proposal_id, "withdrawn": True, "result": body}


def _review_proposal(client: LitKitClient, mid: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """A proposal's status and, once launched, its job."""
    proposal_id = _uuid(args, "proposalId")
    body = client.get(f"/api/matters/{mid}/proposals/{proposal_id}")
    row = body.get("proposal") if isinstance(body, dict) and isinstance(body.get("proposal"), dict) else {}
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    out = {"proposalId": proposal_id, "kind": row.get("kind"), "status": row.get("status"),
           "launched": row.get("launched"), "name": payload.get("name"), "threadId": row.get("threadId"),
           "createdAt": row.get("createdAt"), "reviewedAt": row.get("reviewedAt")}
    return {k: v for k, v in out.items() if v is not None or k in ("status", "launched")}


def _criteria_update(client: LitKitClient, base: str, args: Dict[str, Any]) -> Any:
    """A new version of a set: LitKit's publish takes the whole list and the version it edits."""
    set_id = _set_id(args)
    body: Dict[str, Any] = {"criteria": _criteria(args.get("criteria"))}
    if args.get("baseVersion") is not None:
        version = args["baseVersion"]
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise ValueError("'baseVersion' must be the set's version number you edited")
    else:
        found = client.get(f"{base}/{set_id}")
        version = (found.get("set") or {}).get("currentVersion") if isinstance(found, dict) else None
        if not isinstance(version, int):
            raise ValueError("LitKit did not report the set's current version; give baseVersion")
    body["baseVersion"] = version
    if args.get("changeNote"):
        body["changeNote"] = str(args["changeNote"])[:500]
    status, result = client.json_with_status("POST", f"{base}/{set_id}/publish", json_body=body,
                                             ok_statuses=(409,))
    result = result if isinstance(result, dict) else {"result": result}
    if status == 409 and result.get("error") == "stale_version":
        result["next"] = ("Someone published since that version. Merge into the returned criteria and update again "
                          "from currentVersion.")
    elif status == 409:
        raise LitKitError(f"HTTP 409: {_error_text(result) or 'conflict'}", status=409, body=result)
    elif result.get("written") is False:
        result["next"] = "Nothing changed: these criteria are the current version's."
    return result


def _review_criteria(client: LitKitClient, mid: str, args: Dict[str, Any]) -> Any:
    sub = str(args.get("criteriaAction") or "list")
    base = f"/api/matters/{mid}/criteria-sets"
    if sub == "list":
        return client.get(base)
    if sub == "publish":
        # Hosts before LitKit's 0288 sets published in a second step; sessions may still ask for it.
        if args.get("criteria") is not None:
            return _criteria_update(client, base, args)
        _set_id(args)
        return {"criteriaAction": "publish", "sent": False,
                "next": "There is no separate publish step: create and update each publish a version. Nothing "
                        "was sent."}
    if sub not in CRITERIA_ACTIONS:
        raise ValueError(f"criteriaAction must be one of {', '.join(CRITERIA_ACTIONS)}")
    if sub == "create":
        scope = args.get("setScope") or "user"
        if scope not in CRITERIA_SET_SCOPES:
            raise ValueError(f"'setScope' must be one of {', '.join(CRITERIA_SET_SCOPES)}")
        body: Dict[str, Any] = {"scope": scope, "name": str(_require(args, "name"))[:120],
                                "criteria": _criteria(args.get("criteria"))}
        if args.get("description"):
            body["description"] = str(args["description"])[:500]
        return client.post(base, body)
    if sub == "update":
        return _criteria_update(client, base, args)
    set_id = _set_id(args)
    if sub == "get":
        return client.get(f"{base}/{set_id}")
    return client.get(f"{base}/{set_id}/versions")


@_tool("litkit_review")
def litkit_review(args: Dict[str, Any]) -> Any:
    client = _client()
    mid = _mid(client)
    action = str(args.get("action") or "list")
    base = f"/api/matters/{mid}/review-jobs"
    if action == "list":
        return client.get(base, params={"status": args.get("status"), "limit": args.get("limit")})
    if action == "create":
        return _review_create(client, mid, args)
    if action == "launch":
        return _review_launch(client, mid, args)
    if action == "withdraw":
        return _review_withdraw(client, mid, args)
    if action == "proposal":
        return _review_proposal(client, mid, args)
    if action == "criteria":
        return _review_criteria(client, mid, args)
    if action == "work_sets":
        return client.get(f"/api/matters/{mid}/work-sets")
    if action not in REVIEW_ACTIONS:
        raise ValueError(f"action must be one of {', '.join(REVIEW_ACTIONS)}")
    job = safe_segment(str(_require(args, "jobId")))
    if action == "status":
        return client.get(f"{base}/{job}")
    if action == "records":
        return client.get(f"{base}/{job}/records")
    if action in ("resume", "cancel", "pause"):
        return {"action": action, "jobId": job, "result": client.post(f"{base}/{job}/{action}", {})}
    if action == "accept_tags":
        body = {"includeRationaleNotes": bool(args["includeRationaleNotes"])} \
            if args.get("includeRationaleNotes") is not None else {}
        return {"action": action, "jobId": job, "result": client.post(f"{base}/{job}/accept-all-tags", body)}
    raise ValueError(f"action must be one of {', '.join(REVIEW_ACTIONS)}")


@_tool("litkit_jev")
def litkit_jev(args: Dict[str, Any]) -> Any:
    return _jev.run(args)


@_tool("litkit_ingest")
def litkit_ingest(args: Dict[str, Any]) -> Any:
    client = _client()
    mid = _mid(client)
    action = str(args.get("action") or "productions")
    base = f"/api/matters/{mid}"
    pid = safe_segment(str(args["productionId"])) if args.get("productionId") else None
    job = safe_segment(str(args["jobId"])) if args.get("jobId") else None

    def need(value: Optional[str], name: str) -> str:
        if not value:
            raise ValueError(f"'{name}' is required for {action}")
        return value

    if action == "productions":
        return client.get(f"{base}/productions")
    if action == "production":
        return client.get(f"{base}/productions/{need(pid, 'productionId')}")
    if action == "progress":
        return client.get(f"{base}/productions/{pid}/progress" if pid else f"{base}/productions/progress")
    if action == "exceptions":
        return client.get(f"{base}/productions/{need(pid, 'productionId')}/exceptions")
    if action == "ingests":
        return client.get(f"{base}/ingests/{pid}" if pid else f"{base}/ingests")
    if action == "jobs":
        return client.get(f"{base}/ingest", params={"status": args.get("status")})
    if action == "job":
        return client.get(f"{base}/ingest/{need(job, 'jobId')}")
    if action == "resume":
        body = {"skipRowIndex": int(args["skipRowIndex"])} if args.get("skipRowIndex") is not None else {}
        return {"action": action, "result": client.post(f"{base}/ingest/{need(job, 'jobId')}/resume", body)}
    if action == "cancel":
        return {"action": action, "result": client.post(f"{base}/ingest/{need(job, 'jobId')}/cancel", {})}
    if action == "reingest":
        return {"action": action, "result": client.post(f"{base}/productions/{need(pid, 'productionId')}/reingest", {})}
    if action == "retry":
        body: Dict[str, Any] = {}
        if isinstance(args.get("exceptionIds"), list):
            body["exceptionIds"] = [str(x) for x in args["exceptionIds"]]
        if args.get("includeNonRetryable"):
            body["includeNonRetryable"] = True
        return {"action": action,
                "result": client.post(f"{base}/productions/{need(pid, 'productionId')}/exceptions/retry", body)}
    raise ValueError("unknown action")


@_tool("litkit_proposals")
def litkit_proposals(args: Dict[str, Any]) -> Any:
    client = _client()
    mid = _mid(client)
    action = str(args.get("action") or "create")
    if action == "list":
        return client.get(f"/api/matters/{mid}/proposals", params={"status": args.get("status")})
    if action != "create":
        raise ValueError("action must be create or list")
    kind = str(_require(args, "kind"))
    if kind not in ("tag", "privilege", "redaction"):
        raise ValueError("kind must be tag, privilege or redaction")
    payload = args.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object (tag: {tags:[...]}, privilege: {basis}, redaction: {rects:[...]})")
    body = {"kind": kind, "documentId": _uuid(args, "documentId"), "payload": payload,
            "rationale": str(_require(args, "rationale"))[:2000]}
    return client.post(f"/api/matters/{mid}/proposals", body)


@_tool("litkit_tags")
def litkit_tags(args: Dict[str, Any]) -> Any:
    client = _client()
    mid = _mid(client)
    action = str(args.get("action") or "list")
    if action == "list":
        body = client.get("/api/tags", params={"matterId": mid, "includeCounts": "1"})
        tags = body.get("tags") if isinstance(body, dict) else body
        rows = [_compact(t) for t in tags or [] if isinstance(t, dict)] if isinstance(tags, list) else tags
        return {"tags": rows}
    if action == "create":
        body = {"matterId": mid, "name": str(_require(args, "name"))[:80], "kind": args.get("kind") or "custom"}
        if args.get("color"):
            body["color"] = str(args["color"])
        return client.post("/api/tags", body)
    if action in ("apply", "remove"):
        tag_id = _uuid(args, "tagId")
        docs = args.get("documentIds")
        if not isinstance(docs, list) or not docs:
            raise ValueError("documentIds must be a non-empty list")
        docs = [str(d).strip() for d in docs]
        bad = [d for d in docs if not _UUID.match(d)]
        if bad:
            raise ValueError(f"documentIds must be LitKit ids (uuid); not ids: {bad[:3]}")
        if action == "apply" and len(docs) == 1:
            return client.post(f"/api/documents/{docs[0]}/tags", {"tagId": tag_id, "scope": args.get("scope") or "doc"})
        return client.post(f"/api/matters/{mid}/bulk-tag",
                           {"tagId": tag_id, "docIds": docs[:5000],
                            "action": "add" if action == "apply" else "remove"})
    raise ValueError("action must be list, create, apply or remove")


@_tool("litkit_work_sets")
def litkit_work_sets(args: Dict[str, Any]) -> Any:
    client = _client()
    mid = _mid(client)
    action = str(args.get("action") or "list")
    if action == "list":
        # The matter's sets, whoever made or holds them. /api/work-sets is one person's inbox and outbox.
        role = args.get("role") or None
        if role is not None and role not in WORK_SET_ROLES:
            raise ValueError(f"'role' must be one of {', '.join(WORK_SET_ROLES)}, or left out for every set")
        user = current_acting_user()
        if role and not user:
            raise ValueError("'role' needs a lawyer on this turn; leave it out to list every set on the matter")
        body = client.get(f"/api/matters/{mid}/work-sets")
        if not role or not isinstance(body, dict) or not isinstance(body.get("sets"), list):
            return body
        key = "assignee" if role == "inbox" else "creator"
        mine = [s for s in body["sets"] if isinstance(s, dict) and ((s.get(key) or {}).get("id") == user)]
        return {**body, "sets": mine}
    if action == "get":
        return client.get(f"/api/work-sets/{_uuid(args, 'workSetId')}")
    if action == "create":
        docs = args.get("documentIds")
        if not isinstance(docs, list) or not docs:
            raise ValueError("documentIds must be a non-empty list (up to 500)")
        body = {"matterId": mid, "assigneeId": _uuid(args, "assigneeId"), "name": str(_require(args, "name"))[:120],
                "docIds": [str(d) for d in docs][:500]}
        if args.get("message"):
            body["message"] = str(args["message"])[:2000]
        return client.post("/api/work-sets", body)
    if action in ("close", "reopen"):
        return client.patch(f"/api/work-sets/{_uuid(args, 'workSetId')}",
                            {"status": "done" if action == "close" else "open"})
    raise ValueError("action must be list, get, create, close or reopen")


# ---------------------------------------------------------------------------
# LitLex
# ---------------------------------------------------------------------------

@_tool("litkit_litlex")
def litkit_litlex(args: Dict[str, Any]) -> Any:
    client = _client()
    action = str(args.get("action") or "search")
    if action == "search":
        query = str(_require(args, "query"))
        params: Dict[str, Any] = {"q": query, "topK": args.get("topK"), "offset": args.get("offset"),
                                  "sort": args.get("sort"), "mode": args.get("mode"), "after": args.get("after"),
                                  "before": args.get("before")}
        courts = args.get("court")
        if courts:
            params["court"] = courts if isinstance(courts, list) else [courts]
        return client.get("/api/litlex/search", params=params, timeout=60)
    if action == "opinion":
        oid = safe_segment(str(_require(args, "opinionId")))
        body = client.get(f"/api/litlex/opinions/{oid}", params={"full": "1"}, timeout=60)
        target = output_path("litlex", f"{oid}.json")
        target.write_text(json.dumps(body, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        summary = _compact(body.get("opinion") if isinstance(body, dict) and isinstance(body.get("opinion"), dict)
                           else body if isinstance(body, dict) else {}, limit=200)
        return {"saved": relative(target), "opinionId": oid, "summary": summary,
                "note": "full opinion saved; read it from the file and cite the page anchors it carries"}
    if action == "citator":
        oid = safe_segment(str(_require(args, "opinionId")))
        return client.get(f"/api/litlex/opinions/{oid}/citator",
                          params={"treatment": args.get("treatment"), "limit": args.get("limit")})
    if action == "authorities":
        oid = safe_segment(str(_require(args, "opinionId")))
        return client.get(f"/api/litlex/opinions/{oid}/authorities")
    if action == "statute":
        return client.get("/api/litlex/statute", params={"cite": str(_require(args, "cite"))})
    if action == "resolve":
        cites = args.get("cites")
        if not isinstance(cites, list) or not cites:
            raise ValueError("cites must be a non-empty list of citation strings")
        return client.post("/api/litlex/cite-resolve", {"cites": [str(c) for c in cites]}, idempotent=True)
    if action == "cite_check":
        body: Dict[str, Any] = {"quote": str(_require(args, "quote"))}
        if args.get("opinionId"):
            body["opinionId"] = str(args["opinionId"])
        if args.get("paragraphNumber") is not None:
            body["paragraphNumber"] = int(args["paragraphNumber"])
        return client.post("/api/litlex/cite-check", body, idempotent=True)
    if action == "brief_check":
        path = input_path(str(_require(args, "path")), tool="litkit_litlex")
        _, body = client.upload("/api/litlex/brief-check", path, fields={"matterId": _mid(client)}, timeout=600)
        saved = _save_json("qa", f"brief-check-{safe_segment(path.stem)}", body)
        return {"fullFindings": relative(saved), "result": body}
    raise ValueError("action must be search, opinion, citator, authorities, statute, resolve, cite_check or "
                     "brief_check")


# ---------------------------------------------------------------------------
# notify, memory, keep-tool actions
# ---------------------------------------------------------------------------

NOTIFY_KINDS = ("agent_notify", "agent_task_complete", "agent_needs_input", "task_completed", "task_failed",
                "review_complete", "mention", "system")


@_tool("litkit_notify")
def litkit_notify(args: Dict[str, Any]) -> Any:
    client = _client()
    mid = _mid(client)
    kind = str(args.get("kind") or "agent_notify")
    if kind not in NOTIFY_KINDS:
        raise ValueError(f"kind must be one of {', '.join(NOTIFY_KINDS)}")
    body: Dict[str, Any] = {"kind": kind, "title": str(_require(args, "title"))[:200], "matterId": mid}
    if args.get("body"):
        body["body"] = str(args["body"])[:4000]
    if args.get("link"):
        link = str(args["link"])
        if not link.startswith("/"):
            raise ValueError("link must be an in-app path starting with /")
        body["link"] = link
    if args.get("priority"):
        body["priority"] = args["priority"]
    if args.get("matterWide"):
        target = "matter"
    else:
        user = args.get("userId") or current_acting_user()
        if not user:
            raise ValueError("no acting user on this turn: name a matter member (userId) or set matterWide")
        body["userId"] = str(user)
        target = f"user {user}"
    result = client.post("/api/notifications/emit", body)
    return {"notified": target, "result": result}


def _history_author(row: Dict[str, Any], names: Dict[str, str]) -> str:
    if row.get("role") == "assistant":
        return "Ana"
    user = row.get("authorUserId")
    if isinstance(user, str) and user:
        return names.get(user.lower()) or "A former member"
    ref = row.get("externalRef") if isinstance(row.get("externalRef"), dict) else {}
    slack = ref.get("name") if isinstance(ref.get("name"), str) else ""
    if slack.strip():
        return f"{slack.strip()} (Slack)"
    return "LitKit" if row.get("role") == "system" else "Someone outside LitKit"


@_tool("litkit_channel_history")
def litkit_channel_history(args: Dict[str, Any]) -> Any:
    """Recent messages across a matter channel's team threads, newest first. Read-only; LitKit
    leaves private threads out, even the acting lawyer's own."""
    client = _client()
    mid = _mid(client)
    turn = current_turn()
    slug = str(args.get("channel") or (turn.litkit_channel if turn is not None else None) or "").strip().lstrip("#")
    if not slug:
        raise ValueError("'channel' is required: this turn did not arrive in a LitKit channel")
    if len(slug) > 100:
        raise ValueError("'channel' is a channel slug such as depo-prep")
    limit = max(1, min(int(args.get("limit") or 50), CHANNEL_HISTORY_MAX))
    before = str(args["before"]).strip() if args.get("before") else None
    if before:
        try:
            _dt.datetime.fromisoformat(before.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError("'before' must be an ISO time, e.g. 2026-09-29T10:00:00Z (a page's nextBefore)")
    body = client.get(f"/api/matters/{mid}/channels/{quote(slug, safe='')}/history",
                      params={"before": before, "limit": limit})
    body = body if isinstance(body, dict) else {}
    names: Dict[str, str] = {}
    for person in body.get("people") or []:
        if isinstance(person, dict) and isinstance(person.get("id"), str):
            label = (person.get("name") or "").strip() or (person.get("email") or "").strip()
            if label:
                names[person["id"].lower()] = label
    rows = [{"author": _history_author(m, names), "at": m.get("createdAt"), "threadId": m.get("threadId"),
             "text": _clip(m.get("text") or "", CHANNEL_HISTORY_TEXT_MAX)}
            for m in body.get("messages") or [] if isinstance(m, dict)]
    channel = body.get("channel") if isinstance(body.get("channel"), dict) else {}
    return {"channel": {k: channel.get(k) for k in ("slug", "name", "topic") if channel.get(k)} or {"slug": slug},
            "messages": len(rows), "nextBefore": body.get("nextBefore"),
            "notes": ["newest first; team threads only (private threads are never listed). Page back with "
                      "before=nextBefore."],
            "rows": rows}


def _actions(client: LitKitClient, action: str, action_args: Dict[str, Any]) -> Any:
    return client.post("/api/agent/actions", {"action": action, "matterId": _mid(client), "args": action_args})


# LitKit's remember lane accepts only these kinds (REMEMBER_KINDS in litkit-app remember.ts); any
# other kind is a 400 "remember: invalid kind".
REMEMBER_KINDS = ("fact", "strategy", "custodian_note", "doc_cluster", "timeline_hint")

SHARED_MEMORY_ROUTE = "/api/agent/shared-memory"
SHARED_SCOPES = ("firm", "person")
SHARED_KINDS = ("convention", "preference", "tool_habit")
SHARED_CONTENT_MAX = 1000  # agent_shared_memory.content CHECK (FIRM_AGENT_HOST 4.1)
SHARED_READ_MAX = 50
SHARED_UNAVAILABLE = "firm and personal memory are not available on this LitKit yet"


def _remember_shared(client: LitKitClient, scope: str, args: Dict[str, Any]) -> Any:
    """A firm or person note goes to the shared-memory route, where LitKit's classifier decides.

    FIRM_AGENT_HOST 4.2: LitKit refuses matter facts (and records the verdict); a firm note lands
    ``proposed`` until a Firm admin confirms it. The host only shapes the request.
    """
    if not current_acting_user():
        raise ValueError(f"a {scope} note needs a lawyer on the turn; this turn has none")
    content = " ".join(str(_require(args, "content")).split())
    if len(content) > SHARED_CONTENT_MAX:
        raise ValueError(f"a {scope} note holds at most {SHARED_CONTENT_MAX} characters; state the convention "
                         "or preference in a sentence or two")
    kind = str(args.get("kind") or ("convention" if scope == "firm" else "preference"))
    if kind not in SHARED_KINDS:
        raise ValueError(f"a {scope} note's kind must be one of {', '.join(SHARED_KINDS)}")
    status, body = client.json_with_status("POST", SHARED_MEMORY_ROUTE, ok_statuses=(400, 404, 409, 422),
                                           json_body={"scope": scope, "kind": kind, "content": content})
    if status == 404:
        return {"saved": False, "scope": scope, "unavailable": True,
                "message": f"{SHARED_UNAVAILABLE}; nothing was saved. Save it as a matter note (scope=matter, "
                           "or scope=user for this lawyer only) if it belongs to this matter, and say so."}
    if status >= 400:
        return {"saved": False, "scope": scope, "status": status,
                "error": _error_text(body) or f"LitKit refused the {scope} note (HTTP {status})",
                "notes": ["LitKit keeps facts about a matter out of firm and personal memory. If this is a matter "
                          "fact, save it with scope=matter and tell the lawyer."]}
    body = body if isinstance(body, dict) else {}
    out = {"saved": True, "scope": scope, "id": body.get("id"), "status": body.get("status")}
    if body.get("status") == "proposed":
        out["notes"] = ["a Firm admin must confirm a firm convention before other matters see it"]
    return out


def _error_text(body: Any) -> Optional[str]:
    """LitKit's refusal as text: the error message and the verdict code, never the note itself."""
    if isinstance(body, str):
        return body[:300] or None
    if not isinstance(body, dict):
        return None
    err = body.get("error")
    if isinstance(err, dict):
        err = err.get("message") or err.get("code")
    return ": ".join(dict.fromkeys(str(x) for x in (err, body.get("code") or body.get("verdict")) if x)) or None


@_tool("litkit_remember")
def litkit_remember(args: Dict[str, Any]) -> Any:
    client = _client()
    scope = str(args.get("scope") or "matter")
    if scope not in ("matter", "user", "wall") + SHARED_SCOPES:
        raise ValueError("scope must be matter, user, wall, firm or person")
    if scope in SHARED_SCOPES:
        return _remember_shared(client, scope, args)
    if scope == "user" and not current_acting_user():
        raise ValueError("a private (user) note needs a lawyer on the turn; this turn has none")
    kind = str(args.get("kind") or "fact").strip()
    if kind not in REMEMBER_KINDS:
        raise ValueError(f"kind must be one of {', '.join(REMEMBER_KINDS)} (not {kind!r})")
    action_args: Dict[str, Any] = {"kind": kind, "content": str(_require(args, "content"))[:8000]}
    for key, target in (("key", "key"), ("expiresAt", "expires_at")):
        if args.get(key):
            action_args[target] = args[key]
    if isinstance(args.get("replacesIds"), list):
        action_args["replaces_ids"] = [str(x) for x in args["replacesIds"]][:12]
    if scope != "matter":
        acl: Dict[str, Any] = {"scope": scope}
        if scope == "wall":
            acl["wall_id"] = str(_require(args, "wallId"))
        action_args["acl"] = acl
    return _actions(client, "remember", action_args)


def _shared_items(client: LitKitClient, scope: str, notes: List[str]) -> List[Dict[str, Any]]:
    """Active firm or person notes. Fail-soft: a LitKit without the route (404) or a failed read
    leaves a note and an empty list, so matter recall still answers."""
    try:
        status, body = client.json_with_status("GET", SHARED_MEMORY_ROUTE, params={"scope": scope},
                                               ok_statuses=(404,))
    except LitKitError as exc:
        notes.append(f"{scope} memory could not be read ({exc})")
        return []
    if status == 404:
        if SHARED_UNAVAILABLE not in notes:
            notes.append(SHARED_UNAVAILABLE)
        return []
    items = body.get("items") if isinstance(body, dict) else None
    out = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and isinstance(item.get("content"), str):
            out.append({"id": item.get("id"), "kind": item.get("kind"),
                        "content": _clip(item["content"], SHARED_CONTENT_MAX)})
    return out[:SHARED_READ_MAX]


@_tool("litkit_recall")
def litkit_recall(args: Dict[str, Any]) -> Any:
    """Matter memories (wall-filtered, as before) plus the firm's conventions and, when a lawyer is on
    the turn, that lawyer's own notes (FIRM_AGENT_HOST 4.3), each labelled with its scope."""
    client = _client()
    scope = str(args.get("scope") or "all")
    if scope not in ("all", "matter") + SHARED_SCOPES:
        raise ValueError("scope must be all, matter, firm or person")
    out: Dict[str, Any] = {}
    notes: List[str] = []
    if scope in ("all", "matter"):
        action_args = {k: args[k] for k in ("query", "kind") if args.get(k)}
        if args.get("limit"):
            action_args["limit"] = max(1, min(int(args["limit"]), 50))
        out["matter"] = _actions(client, "recall", action_args)
    if scope in ("all", "firm"):
        out["firm"] = _shared_items(client, "firm", notes)
    if scope in ("all", "person"):
        if current_acting_user():
            out["person"] = _shared_items(client, "person", notes)
        else:
            notes.append("no lawyer on this turn, so no personal notes")
    notes.append("name your source when you use one: this matter, a firm convention, or the lawyer's preference")
    out["notes"] = notes
    return out


CROSS_MATTER_ROUTE = "/api/agent/cross-matter-search"
CROSS_MATTER_HITS_MAX = 10
CROSS_MATTER_SNIPPET_MAX = 300
CROSS_MATTER_UNAVAILABLE = ("Searching other matters is not available in this thread: it works only in a lawyer's own "
                            "private thread. Tell the lawyer: \"Ask me in a private thread and I'll search your "
                            "other matters.\"")


@_tool("litkit_cross_matter_search")
def litkit_cross_matter_search(args: Dict[str, Any]) -> Any:
    """FIRM_AGENT_HOST 6.1: search the acting lawyer's other matters through LitKit.

    LitKit checks the lawyer's own access on each matter, applies walls, audits, and returns at most
    ten labelled snippets. The result is returned inline and never spilled to a file, so no other
    matter's text is written into this matter's home.
    """
    query = str(_require(args, "query"))
    grant = current_cross_matter_grant()
    if not grant or not current_acting_user():
        return dumps({"available": False, "message": CROSS_MATTER_UNAVAILABLE})
    body: Dict[str, Any] = {"query": query[:1000]}
    if args.get("matterIds") is not None:
        ids = args["matterIds"]
        if not isinstance(ids, list) or not all(isinstance(x, str) and _UUID.match(x) for x in ids):
            raise ValueError("'matterIds' must be a list of LitKit matter ids (uuid)")
        body["matterIds"] = ids[:50]
    client = _client()
    try:
        status, result = client.json_with_status("POST", CROSS_MATTER_ROUTE, json_body=body, ok_statuses=(404,),
                                                 extra_headers={TURN_GRANT_HEADER: grant})
    except LitKitPermissionError as exc:
        out = exc.to_dict()
        if "grant" in str(exc.code or "").lower():
            out["message"] = CROSS_MATTER_UNAVAILABLE
        return dumps(out)
    if status == 404:
        return dumps({"available": False, "message": "Searching other matters is not available on this LitKit yet."})
    raw = result.get("hits") if isinstance(result, dict) else None
    hits = []
    for hit in raw if isinstance(raw, list) else []:
        if not isinstance(hit, dict):
            continue
        row = {k: hit.get(k) for k in ("matterId", "matterName", "documentId", "bates", "title") if hit.get(k)}
        row["snippet"] = _clip(hit.get("snippet") or "", CROSS_MATTER_SNIPPET_MAX)
        hits.append(row)
    hits = hits[:CROSS_MATTER_HITS_MAX]
    return dumps({"available": True, "hits": len(hits), "results": hits,
                  "notes": ["cite every hit with its matter name; the lawyer opens the document in that matter",
                            "never save a hit, or anything drawn from one, to memory",
                            "snippets only: this matter's tools cannot open another matter's documents"]})


PASSTHROUGH_ACTIONS = ("term_frequency", "find_redacted", "hot_documents", "refresh_dossier", "diagnose_issue",
                       "diagnose_ingest", "litlex_format_cite")


@_tool("litkit_actions")
def litkit_actions(args: Dict[str, Any]) -> Any:
    client = _client()
    action = str(_require(args, "action"))
    if action not in PASSTHROUGH_ACTIONS:
        raise ValueError(f"action must be one of {', '.join(PASSTHROUGH_ACTIONS)}")
    action_args = args.get("args") or {}
    if not isinstance(action_args, dict):
        raise ValueError("args must be an object")
    status, body = client.json_with_status("POST", "/api/agent/actions", ok_statuses=(400,),
                                           json_body={"action": action, "matterId": _mid(client), "args": action_args})
    if status == 400:
        return {"error": (body or {}).get("error") if isinstance(body, dict) else body, "status": 400,
                "action": action}
    return body


# ---------------------------------------------------------------------------
# schemas and registration table
# ---------------------------------------------------------------------------

def _schema(name: str, description: str, properties: Dict[str, Any], required: Iterable[str] = ()) -> Dict[str, Any]:
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": list(required)}}


_S = {"type": "string"}
_I = {"type": "integer"}
_B = {"type": "boolean"}
_IDS = {"type": "array", "items": {"type": "string"}}
_DATE = {"type": "string", "description": "YYYY-MM-DD"}

SCHEMAS: Dict[str, Dict[str, Any]] = {
    "litkit_matter": _schema(
        "litkit_matter",
        "The matter this host serves: name, document count, custodians, productions, Bates prefixes. Start here.",
        {}),
    "litkit_search": _schema(
        "litkit_search",
        "Search the matter's produced documents (Boolean and phrase syntax). LitKit allows 5 seconds and returns at "
        "most 500 hits, so this finds mentions; it is not a census. Use quoted phrases, distinctive names, or Bates "
        "numbers; single common words time out or rank. A timeout or zero hits is not proof of absence. Hits "
        "without Bates carry link: paste it as written when you name the document.",
        {"query": {"type": "string", "description": "e.g. '\"average selling price\"' or 'ABC0001234'"},
         "custodian": _S, "dateFrom": _DATE, "dateTo": _DATE, "bates": _S,
         "limit": {"type": "integer", "description": "1-500, default 100"}, "matchCase": _B, "wholeWord": _B},
        ["query"]),
    "litkit_docs": _schema(
        "litkit_docs",
        "Document census with custodian/date/Bates filters and cursor paging. Without saveAs: one page plus "
        "nextCursor. With saveAs: pages through everything into census/<saveAs>.jsonl and returns counts. Rows "
        "without Bates carry link: paste it as written when you name the document.",
        {"custodian": _S, "dateFrom": _DATE, "dateTo": _DATE, "bates": _S, "productionId": _S,
         "q": {"type": "string", "description": "optional text filter (must be narrow enough to page)"},
         "limit": {"type": "integer", "description": "rows per page, 1-5000, default 1000"},
         "cursor": {"type": "string", "description": "nextCursor from the previous page"},
         "saveAs": {"type": "string", "description": "census name; loops all pages to census/<name>.jsonl"},
         "maxPages": _I}),
    "litkit_document": _schema(
        "litkit_document", "Metadata for one document (by documentId or Bates): Bates range, custodian, dates, "
        "author, subject, tags, productions. Returns link: paste it as written when you name the document.",
        {"documentId": _S, "bates": _S}),
    "litkit_text": _schema(
        "litkit_text", "Extracted text of one document, saved to texts/<bates>.txt under a self-citing header "
        "(Bates, docId, custodian, date). Returns the path, a preview and link (paste it as written when you name "
        "the document); read the file for the full text.",
        {"documentId": _S, "bates": _S, "dir": {"type": "string", "description": "folder, default texts"},
         "previewChars": _I}),
    "litkit_pdf": _schema(
        "litkit_pdf", "Download a document's PDF (or its native file with native=true) to pdfs/ or natives/. "
        "Returns link: paste it as written when you name the document.",
        {"documentId": _S, "bates": _S, "native": _B, "dir": _S}),
    "litkit_export_text": _schema(
        "litkit_export_text", "Bulk text export: writes each document to texts/<bates>.txt with a self-citing "
        "header and keeps texts/index.json. Batches 500 ids per call, skips files already pulled, resumable.",
        {"documentIds": _IDS, "fromCensus": {"type": "string", "description": "census .jsonl from litkit_docs"},
         "dir": _S, "overwrite": _B}),
    "litkit_memos": _schema(
        "litkit_memos", "Prior review memos (source maps naming the best documents). action=list, or action=read "
        "with fileId to save one to memos/.",
        {"action": {"type": "string", "enum": ["list", "read"]}, "fileId": _S}),
    "litkit_files": _schema(
        "litkit_files", "The matter's LitSpace files: list, search, read (download to files/), or upload a file "
        "from the working directory. Rows carry link (and folderLink): paste it as written when you name the "
        "file or folder.",
        {"action": {"type": "string", "enum": ["list", "search", "read", "upload"]}, "query": _S, "fileId": _S,
         "path": _S, "filename": _S, "parentId": _S, "cursor": _S, "limit": _I, "dir": _S}, ["action"]),
    "litkit_deliver": _schema(
        "litkit_deliver", "Commit a finished work-product file to the matter through LitKit's deliverable gates "
        "(quote re-resolve, pleading, citation, prose). Returns gate findings; blocked=true means nothing was "
        "saved: report the findings to the user instead of resubmitting unchanged. documentId adds a new version.",
        {"path": {"type": "string", "description": "an existing file; write and check it before calling"},
         "deliverableClass": {"type": "string", "enum": DELIVERABLE_CLASSES},
         "documentId": _S, "folder": {"type": "string", "description": "under Work Product/"}, "filename": _S,
         "note": _S, "provenance": {"type": "object"}}, ["path", "deliverableClass"]),
    "litkit_quote_check": _schema(
        "litkit_quote_check", "Check every quotation in a draft file (or pasted text) against the matter's "
        "documents. Writes nothing to LitKit; findings saved to qa/.",
        {"path": _S, "text": _S, "deliverableClass": _S}),
    "litkit_review": _schema(
        "litkit_review", "Review & Tag. Register tags, criteria set "
        "(action=criteria) and scope; create proposes. Report the estimate, ask whether to launch, and call "
        "action=launch only after the person says yes in the thread. requiresApproval: quoteId and "
        "userConfirmed=true only after a yes to the price. Also withdraw, proposal, list, status, records, "
        "resume/cancel/pause, work_sets, accept_tags.",
        {"action": {"type": "string", "enum": list(REVIEW_ACTIONS)},
         "jobId": _S, "status": _S, "limit": _I,
         "proposalId": {"type": "string", "description": "launch, withdraw, proposal: the id create returned"},
         "criteriaAction": {"type": "string", "enum": list(CRITERIA_ACTIONS),
                            "description": "with action=criteria (default list)"},
         "criteriaSetId": {"type": "string", "description": "a criteria set id (uuid) or builtin:<name>"},
         "criteria": {"type": "array", "description": "criteria for criteriaAction create/update (or an unregistered "
                      "run): [{title, description, tagName, seedQuery, disposition}]",
                      "items": {"type": "object", "properties": {
                          "title": _S, "description": _S, "tagName": _S, "tagId": _S, "seedQuery": _S, "key": _S,
                          "disposition": {"type": "string", "enum": list(CRITERION_DISPOSITIONS)}},
                          "required": ["title"]}},
         "name": {"type": "string", "description": "criteria set name (create), or the run's name"},
         "description": _S, "changeNote": _S,
         "baseVersion": {"type": "integer", "description": "criteriaAction=update: the version you edited (default: "
                         "the current one)"},
         "criteriaSetVersion": {"type": "integer", "description": "create: run this version of the set (default: "
                                "current)"},
         "tier": {"type": "string", "enum": list(REVIEW_TIERS), "description": "create: model tier (default fast)"},
         "setScope": {"type": "string", "enum": list(CRITERIA_SET_SCOPES),
                      "description": "criteriaAction=create: user (default, the person's own set) or firm"},
         "scope": {"type": "object", "description": "documents for create; exactly one of {workSetId}, "
                   "{documentIds:[..]}, {filter:{custodian, dateFrom, dateTo, query, tagIds, ...}}, "
                   "{batesRange:{start, end}}"},
         "tags": {"type": "array", "items": {"type": "string"},
                  "description": "create: optional; the criteria rows name the tags. If sent, exactly the rows' "
                                 "tags"},
         "createMissingTags": _B, "applyTags": {"type": "boolean", "description": "create: tags apply unless false "
                                                "(false leaves them as proposals). With criteriaSetId leave it out: "
                                                "true is refused; set the rows' disposition instead"},
         "quoteId": {"type": "string", "description": "billing phase 2: the quote id from requiresApproval"},
         "userConfirmed": {"type": "boolean", "description": "billing phase 2: true only after the person said yes "
                           "to the quoted price"},
         "includeRationaleNotes": _B,
         "firstPass": {"type": "object", "description": "create: a Jev first pass before the full review, "
                       "{enabled, thresholds?: {low, priv}}. Omit for the server default (on when Jev is "
                       "configured). Decide it first: skill jev-first-pass-review",
                       "properties": {"enabled": _B, "thresholds": {"type": "object", "properties": {
                           "low": {"type": "number"}, "priv": {"type": "number"}}}},
                       "required": ["enabled"]},
         "optimizations_considered": {"type": "array", "items": {"type": "string"}, "maxItems": 5,
                                      "description": "create: up to 5 short notes (200 chars each) on the cost "
                                      "optimizations you weighed, e.g. 'Jev first pass on: 4,000 topical docs'"}}),
    "litkit_jev": _jev.SCHEMA,
    "litkit_ingest": _schema(
        "litkit_ingest", "Production and ingest status (productions, production, progress, exceptions, ingests, "
        "jobs, job) and recovery (resume, cancel, reingest, retry), which need matter admin rights.",
        {"action": {"type": "string", "enum": ["productions", "production", "progress", "exceptions", "ingests",
                                               "jobs", "job", "resume", "cancel", "reingest", "retry"]},
         "productionId": _S, "jobId": _S, "status": _S, "skipRowIndex": _I, "exceptionIds": _IDS,
         "includeNonRetryable": _B}),
    "litkit_proposals": _schema(
        "litkit_proposals", "Propose a tag, privilege call, or redaction for human approval, or list proposals.",
        {"action": {"type": "string", "enum": ["create", "list"]},
         "kind": {"type": "string", "enum": ["tag", "privilege", "redaction"]}, "documentId": _S,
         "payload": {"type": "object", "description": "tag {tags:[..]}; privilege {basis}; redaction "
                                                      "{rects:[{page,x,y,w,h}], reason}"},
         "rationale": _S, "status": _S}),
    "litkit_tags": _schema(
        "litkit_tags", "Matter tags: list, create, apply to or remove from documents.",
        {"action": {"type": "string", "enum": ["list", "create", "apply", "remove"]}, "name": _S,
         "kind": {"type": "string", "enum": ["issue", "privilege", "responsive", "custom"]}, "color": _S,
         "tagId": _S, "documentIds": _IDS, "scope": {"type": "string", "enum": ["doc", "attachments", "family"]}}),
    "litkit_work_sets": _schema(
        "litkit_work_sets", "Work sets (document batches assigned to a reviewer): list (every set on the matter), "
        "get, create, close, reopen.",
        {"action": {"type": "string", "enum": ["list", "get", "create", "close", "reopen"]}, "workSetId": _S,
         "assigneeId": _S, "name": _S, "message": _S, "documentIds": _IDS,
         "role": {"type": "string", "enum": list(WORK_SET_ROLES),
                  "description": "list: only sets handed to (inbox) or made by (outbox) this lawyer"}}),
    "litkit_litlex": _schema(
        "litkit_litlex", "LitLex legal research: search, opinion (saved to litlex/), citator, authorities, "
        "statute, resolve citations, cite_check a quotation, brief_check a draft file.",
        {"action": {"type": "string", "enum": ["search", "opinion", "citator", "authorities", "statute", "resolve",
                                               "cite_check", "brief_check"]},
         "query": _S, "opinionId": _S, "cite": _S, "cites": _IDS, "quote": _S, "paragraphNumber": _I, "path": _S,
         "topK": _I, "offset": _I, "sort": _S, "mode": {"type": "string", "enum": ["boolean", "hybrid"]},
         "court": _IDS, "after": _DATE, "before": _DATE, "treatment": _S, "limit": _I}, ["action"]),
    "litkit_notify": _schema(
        "litkit_notify", "Send a LitKit notification to the lawyer on this turn (default), a named matter member "
        "(userId), or the whole matter (matterWide).",
        {"title": _S, "body": _S, "link": {"type": "string", "description": "in-app path starting with /"},
         "userId": _S, "matterWide": _B, "kind": {"type": "string", "enum": list(NOTIFY_KINDS)}, "priority": _S},
        ["title"]),
    "litkit_remember": _schema(
        "litkit_remember", "Save a memory in LitKit. matter (default): for this matter's team; user: this lawyer "
        "only; wall: one wall. firm: a convention every matter follows (a Firm admin confirms it); person: this "
        "lawyer's preference in all their matters. Firm and person hold conventions and preferences only, never "
        "matter facts (parties, amounts, dates, Bates, case details); LitKit refuses those there.",
        {"content": {"type": "string", "description": "firm/person: at most 1000 characters"}, "kind": {
            "type": "string", "description": "matter/user/wall: one of " + ", ".join(REMEMBER_KINDS) + " (default fact); "
                                             "firm/person: convention, preference or tool_habit"},
         "key": _S, "scope": {"type": "string", "enum": ["matter", "user", "wall", "firm", "person"]},
         "wallId": _S, "expiresAt": _S, "replacesIds": _IDS}, ["content"]),
    "litkit_recall": _schema(
        "litkit_recall", "Recall memories from LitKit, labelled by scope: this matter's (wall-filtered for the lawyer "
        "on this turn), the firm's conventions, and the lawyer's own preferences. Say which one you relied on.",
        {"query": _S, "kind": _S, "limit": _I,
         "scope": {"type": "string", "enum": ["all", "matter", "firm", "person"], "description": "default all"}}),
    "litkit_cross_matter_search": _schema(
        "litkit_cross_matter_search", "Search the acting lawyer's OTHER matters for a term (private threads only). "
        "LitKit applies that lawyer's own access and walls, logs the search, and returns at most ten snippets, "
        "each labelled with its matter. Cite every hit with its matter name; never save a hit to memory. In a "
        "shared thread it reports that it is unavailable: offer to search from a private thread.",
        {"query": _S, "matterIds": {"type": "array", "items": {"type": "string"},
                                    "description": "optional: only these matters"}}, ["query"]),
    "litkit_actions": _schema(
        "litkit_actions", "LitKit analysis actions: term_frequency, find_redacted, hot_documents, refresh_dossier, "
        "diagnose_issue, diagnose_ingest, litlex_format_cite.",
        {"action": {"type": "string", "enum": list(PASSTHROUGH_ACTIONS)},
         "args": {"type": "object", "description": "the action's arguments, e.g. {query, topN} for term_frequency"}},
        ["action"]),
    "litkit_channel_history": _schema(
        "litkit_channel_history", "Recent messages in a matter channel's team threads, newest first: author, "
        "time, thread id, text. Private threads are never included. Read-only. Page back with before=nextBefore.",
        {"channel": {"type": "string", "description": "channel slug, e.g. depo-prep (default: this turn's channel)"},
         "limit": {"type": "integer", "description": "1-100, default 50"},
         "before": {"type": "string", "description": "ISO time; only messages older than this"}}),
    "litkit_attachment": _schema(
        "litkit_attachment", "Fetch a chat attachment by its LitKit fileId into inbox/<fileId>/.",
        {"fileId": _S, "filename": _S}, ["fileId"]),
    "litco_deliver_local": _schema(
        "litco_deliver_local", "Attach a file you wrote to this turn's reply so the lawyer receives it. Register "
        "every file you mean to hand over (scratch files are never sent). Only .docx .xlsx .pptx .pdf .md .txt "
        ".csv .png .jpg files in deliverables/ are sent without it. This does not commit to the matter; "
        "litkit_deliver does.",
        {"path": {"type": "string", "description": "an existing file in the working directory"},
         "name": {"type": "string", "description": "file name the lawyer sees (default: the file's own name)"},
         "deliverableClass": {"type": "string", "enum": DELIVERABLE_CLASSES}}, ["path"]),
}

HANDLERS: Dict[str, Callable[..., str]] = {
    "litkit_matter": litkit_matter, "litkit_search": litkit_search, "litkit_docs": litkit_docs,
    "litkit_document": litkit_document, "litkit_text": litkit_text, "litkit_pdf": litkit_pdf,
    "litkit_export_text": litkit_export_text, "litkit_memos": litkit_memos, "litkit_files": litkit_files,
    "litkit_deliver": litkit_deliver, "litkit_quote_check": litkit_quote_check, "litkit_review": litkit_review,
    "litkit_jev": litkit_jev, "litkit_ingest": litkit_ingest, "litkit_proposals": litkit_proposals,
    "litkit_tags": litkit_tags,
    "litkit_work_sets": litkit_work_sets, "litkit_litlex": litkit_litlex, "litkit_notify": litkit_notify,
    "litkit_remember": litkit_remember, "litkit_recall": litkit_recall, "litkit_actions": litkit_actions,
    "litkit_cross_matter_search": litkit_cross_matter_search,
    "litkit_attachment": litkit_attachment, "litkit_channel_history": litkit_channel_history,
    "litco_deliver_local": litco_deliver_local,
}

# Tools that make no LitKit call are offered on any matter host.
CHECKS: Dict[str, Callable[[], bool]] = {"litco_deliver_local": check_matter_host}

TOOLS: List[Tuple[str, Dict[str, Any], Callable[..., str]]] = [(n, SCHEMAS[n], HANDLERS[n]) for n in SCHEMAS]


def register(ctx: Any) -> None:
    """Register every LitKit tool in the ``litkit`` toolset (called by ``plugins/litkit``)."""
    for name, schema, handler in TOOLS:
        ctx.register_tool(name=name, toolset=TOOLSET, schema=schema, handler=handler,
                          check_fn=CHECKS.get(name, check_available), emoji="⚖")
