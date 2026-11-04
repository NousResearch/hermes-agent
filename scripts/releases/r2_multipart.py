"""Multipart transport for release files larger than a single upload part."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
from typing import Iterable
from xml.etree import ElementTree as ET


def _file_range(path: str, offset: int, size: int) -> Iterable[bytes]:
    with open(path, "rb") as handle:
        handle.seek(offset)
        while size:
            chunk = handle.read(min(size, 1024 * 1024))
            if not chunk:
                raise OSError(f"Upload source was truncated: {path}")
            size -= len(chunk)
            yield chunk


def upload_file(
    url: str, path: str, size: int, creds: dict[str, str],
    content_type: str | None, headers: dict[str, str], part_size: int,
) -> None:
    from . import r2

    if not 5 * 1024 * 1024 <= part_size <= 5 * 1024**3:
        raise ValueError("Multipart parts must be between 5 MiB and 5 GiB")
    if (size + part_size - 1) // part_size > 10000:
        raise ValueError("Multipart upload exceeds 10,000 parts")
    response = r2.signed_request(
        "POST", url + "?uploads=", creds=creds, now=r2.amz_timestamp(),
        content_length=0, content_type=content_type, extra_headers=headers,
    )
    upload_id = ET.fromstring(response.text()).findtext("{*}UploadId")
    if not upload_id:
        raise r2.R2RequestError("POST", url, 502, "Missing multipart upload ID")
    upload_url = url + "?" + r2.canonical_query({"uploadId": upload_id})

    def upload_part(offset: int) -> tuple[int, str]:
        number = offset // part_size + 1
        length = min(part_size, size - offset)
        digest = hashlib.sha256()
        for chunk in _file_range(path, offset, length):
            digest.update(chunk)
        part_url = url + "?" + r2.canonical_query({
            "partNumber": str(number), "uploadId": upload_id,
        })
        part = r2.signed_request(
            "PUT", part_url, creds=creds, now=r2.amz_timestamp(),
            body_iter_factory=lambda: _file_range(path, offset, length),
            body_hash=digest.hexdigest(), content_length=length,
        )
        etag = part.header("etag")
        if not etag:
            raise r2.R2RequestError("PUT", part_url, 502, "Missing part ETag")
        print(f"r2: uploaded part {number} ({length} bytes)", flush=True)
        return number, etag

    try:
        # Each worker reopens its range on retries; no whole-artifact buffer.
        with ThreadPoolExecutor() as pool:
            parts = list(pool.map(upload_part, range(0, size, part_size)))
        document = ET.Element("CompleteMultipartUpload")
        for number, etag in parts:
            part = ET.SubElement(document, "Part")
            ET.SubElement(part, "PartNumber").text = str(number)
            ET.SubElement(part, "ETag").text = etag
        body = ET.tostring(document, encoding="utf-8")
        conditions = {key: value for key, value in headers.items() if key.lower().startswith("if-")}
        try:
            response = r2.signed_request(
                "POST", upload_url, creds=creds, now=r2.amz_timestamp(),
                body=body, body_hash=hashlib.sha256(body).hexdigest(),
                content_length=len(body), content_type="application/xml",
                extra_headers=conditions, tries=1,
            )
        except (OSError, http.client.HTTPException) as error:
            # Completion may already have committed. A fresh conditional upload
            # either succeeds or reaches the caller's immutable digest check.
            raise r2.R2RequestError("POST", upload_url, 503, str(error)) from error
        # S3 can return an error XML document after sending HTTP 200 headers.
        result = ET.fromstring(response.text())
        if result.tag.rsplit("}", 1)[-1] != "CompleteMultipartUploadResult":
            code = result.findtext("{*}Code")
            status = 503 if code in {"InternalError", "SlowDown", "ServiceUnavailable", "RequestTimeout"} else 400
            raise r2.R2RequestError("POST", upload_url, status, response.text())
    except BaseException as error:
        try:
            r2.signed_request("DELETE", upload_url, creds=creds, now=r2.amz_timestamp())
        except Exception as cleanup_error:
            error.add_note(f"Multipart abort failed: {cleanup_error}")
        raise
