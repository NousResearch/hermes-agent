"""Async HTTP client for Nextcloud WebDAV + OCS file operations."""
import logging
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import unquote

import httpx

logger = logging.getLogger(__name__)

# Share permission bits
PERM_READ = 1
PERM_UPDATE = 2
PERM_CREATE = 4
PERM_DELETE = 8
PERM_SHARE = 16
PERM_ALL = 31

# XML namespaces used by NC WebDAV
_NS = {
    "d": "DAV:",
    "oc": "http://owncloud.org/ns",
    "nc": "http://nextcloud.org/ns",
}


@dataclass
class FileInfo:
    """Metadata for a remote Nextcloud file."""
    path: str
    etag: str
    size: int
    mtime: float
    file_id: int
    content_type: str = ""
    is_dir: bool = False


class NextcloudFilesClient:
    """Async HTTP client for Nextcloud file operations.

    Handles WebDAV (PROPFIND, GET, PUT, MKCOL, MOVE, DELETE) and
    OCS (file ID resolution, sharing) APIs.
    """

    def __init__(self, base_url: str, username: str, password: str):
        self._base_url = base_url.rstrip("/")
        self._username = username
        self._dav_base = f"/remote.php/dav/files/{username}"
        self._http = httpx.AsyncClient(
            auth=(username, password),
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=0),
        )
        self._ocs_http = httpx.AsyncClient(
            auth=(username, password),
            headers={
                "OCS-APIRequest": "true",
                "Accept": "application/json",
            },
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=0),
        )

    def _dav_url(self, remote_path: str) -> str:
        """Build full WebDAV URL from relative path."""
        clean = remote_path.lstrip("/")
        return f"{self._base_url}{self._dav_base}/{clean}"

    def _parse_propfind(self, xml_text: str) -> List[FileInfo]:
        """Parse a PROPFIND multistatus XML response into FileInfo list."""
        root = ET.fromstring(xml_text)
        results = []
        for response in root.findall("d:response", _NS):
            href = response.findtext("d:href", "", _NS)
            decoded = unquote(href)
            rel_path = decoded.removeprefix(self._dav_base)
            if not rel_path or rel_path == "/":
                continue

            propstat = response.find("d:propstat", _NS)
            if propstat is None:
                continue
            prop = propstat.find("d:prop", _NS)
            if prop is None:
                continue

            rt = prop.find("d:resourcetype", _NS)
            is_dir = rt is not None and rt.find("d:collection", _NS) is not None

            size_text = prop.findtext("d:getcontentlength", "0", _NS)
            size = int(size_text) if size_text else 0

            etag = prop.findtext("d:getetag", "", _NS)
            content_type = prop.findtext("d:getcontenttype", "", _NS)

            mtime_text = prop.findtext("d:getlastmodified", "", _NS)
            mtime = 0.0
            if mtime_text:
                try:
                    mtime = parsedate_to_datetime(mtime_text).timestamp()
                except Exception:
                    pass

            file_id_text = prop.findtext("oc:fileid", "0", _NS)
            file_id = int(file_id_text) if file_id_text else 0

            results.append(FileInfo(
                path=rel_path.rstrip("/"),
                etag=etag,
                size=size,
                mtime=mtime,
                file_id=file_id,
                content_type=content_type,
                is_dir=is_dir,
            ))
        return results

    async def propfind(self, path: str = "/", depth: str = "1") -> List[FileInfo]:
        """List files at path via WebDAV PROPFIND.

        depth: "0" (file only), "1" (direct children), "infinity" (recursive).
        """
        url = self._dav_url(path)
        body = """<?xml version="1.0"?>
        <d:propfind xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns" xmlns:nc="http://nextcloud.org/ns">
          <d:prop>
            <d:getcontentlength/>
            <d:getetag/>
            <d:getcontenttype/>
            <d:getlastmodified/>
            <d:resourcetype/>
            <oc:fileid/>
          </d:prop>
        </d:propfind>"""
        resp = await self._http.request(
            method="PROPFIND",
            url=url,
            content=body,
            headers={"Depth": depth, "Content-Type": "application/xml"},
        )
        resp.raise_for_status()
        return self._parse_propfind(resp.text)

    async def download(self, remote_path: str, local_path: Path) -> bool:
        """Download a file from NC via streaming GET. Uses atomic .tmp write."""
        import asyncio as _aio
        url = self._dav_url(remote_path)
        local_path = Path(local_path)
        await _aio.to_thread(local_path.parent.mkdir, parents=True, exist_ok=True)
        tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
        try:
            async with self._http.stream("GET", url) as resp:
                resp.raise_for_status()
                fh = await _aio.to_thread(open, tmp_path, "wb")
                try:
                    async for chunk in resp.aiter_bytes(chunk_size=65536):
                        await _aio.to_thread(fh.write, chunk)
                finally:
                    await _aio.to_thread(fh.close)
            await _aio.to_thread(tmp_path.rename, local_path)
            return True
        except Exception as e:
            logger.error("Download failed %s: %s", remote_path, e)
            if tmp_path.exists():
                tmp_path.unlink()
            return False

    async def upload(self, local_path: Path, remote_path: str) -> Optional[str]:
        """Upload a file via WebDAV PUT. Returns ETag on success, None on failure."""
        url = self._dav_url(remote_path)
        local_path = Path(local_path)
        import asyncio as _aio
        try:
            data = await _aio.to_thread(local_path.read_bytes)
            resp = await self._http.put(url, content=data)
            resp.raise_for_status()
            return resp.headers.get("ETag", "")
        except Exception as e:
            logger.error("Upload failed %s: %s", remote_path, e)
            return None

    async def upload_chunked(
        self, local_path: Path, remote_path: str,
        chunk_size: int = 5 * 1024 * 1024,
    ) -> Optional[str]:
        """Upload via NC Chunked Upload v2: MKCOL -> PUT chunks -> MOVE."""
        local_path = Path(local_path)
        file_size = local_path.stat().st_size
        upload_id = uuid.uuid4().hex
        upload_dir = f"/remote.php/dav/uploads/{self._username}/{upload_id}"
        upload_dir_url = f"{self._base_url}{upload_dir}"

        try:
            resp = await self._http.request(method="MKCOL", url=upload_dir_url)
            resp.raise_for_status()

            import asyncio as _aio
            offset = 0
            chunk_num = 0
            fh = await _aio.to_thread(open, local_path, "rb")
            try:
                while offset < file_size:
                    data = await _aio.to_thread(fh.read, chunk_size)
                    chunk_url = f"{upload_dir_url}/{chunk_num:05d}"
                    resp = await self._http.put(chunk_url, content=data)
                    resp.raise_for_status()
                    offset += len(data)
                    chunk_num += 1
            finally:
                await _aio.to_thread(fh.close)

            dest_url = self._dav_url(remote_path)
            resp = await self._http.request(
                method="MOVE",
                url=f"{upload_dir_url}/.file",
                headers={"Destination": dest_url},
            )
            resp.raise_for_status()
            return resp.headers.get("ETag", "")
        except Exception as e:
            logger.error("Chunked upload failed %s: %s", remote_path, e)
            return None

    async def mkdir(self, remote_path: str) -> bool:
        """Create a directory via WebDAV MKCOL."""
        url = self._dav_url(remote_path)
        try:
            resp = await self._http.request(method="MKCOL", url=url)
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.error("mkdir failed %s: %s", remote_path, e)
            return False

    async def delete(self, remote_path: str) -> bool:
        """Delete a file or directory via WebDAV DELETE."""
        url = self._dav_url(remote_path)
        try:
            resp = await self._http.delete(url)
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.error("delete failed %s: %s", remote_path, e)
            return False

    async def move(self, src: str, dst: str) -> bool:
        """Move/rename a file via WebDAV MOVE."""
        src_url = self._dav_url(src)
        dst_url = self._dav_url(dst)
        try:
            resp = await self._http.request(
                method="MOVE", url=src_url,
                headers={"Destination": dst_url},
            )
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.error("move failed %s -> %s: %s", src, dst, e)
            return False

    async def resolve_file_ids(self, file_ids: list) -> List[FileInfo]:
        """Resolve NC file IDs to FileInfo via OCS Files API."""
        results = []
        for fid in file_ids:
            url = f"{self._base_url}/ocs/v2.php/apps/files/api/v1/files/{fid}"
            try:
                resp = await self._ocs_http.get(url)
                resp.raise_for_status()
                data = resp.json().get("ocs", {}).get("data", {})
                path = data.get("path", "")
                name = data.get("name", "")
                full_path = f"{path}/{name}" if path else f"/{name}"
                results.append(FileInfo(
                    path=full_path,
                    etag=data.get("etag", ""),
                    size=data.get("size", 0),
                    mtime=float(data.get("mtime", 0)),
                    file_id=data.get("id", fid),
                    content_type=data.get("mimetype", ""),
                    is_dir=data.get("mimetype") == "httpd/unix-directory",
                ))
            except Exception as e:
                logger.warning("Failed to resolve file ID %d: %s", fid, e)
        return results

    async def share(
        self, remote_path: str, user: str, permissions: int = PERM_ALL,
    ) -> Optional[dict]:
        """Share a file/folder with a user via OCS Sharing API."""
        url = f"{self._base_url}/ocs/v2.php/apps/files_sharing/api/v1/shares"
        try:
            resp = await self._ocs_http.post(url, data={
                "path": remote_path,
                "shareWith": user,
                "shareType": 0,
                "permissions": permissions,
            })
            resp.raise_for_status()
            return resp.json().get("ocs", {}).get("data", {})
        except Exception as e:
            logger.error("Share failed %s -> %s: %s", remote_path, user, e)
            return None

    async def unshare(self, share_id: int) -> bool:
        """Remove a share by ID."""
        url = f"{self._base_url}/ocs/v2.php/apps/files_sharing/api/v1/shares/{share_id}"
        try:
            resp = await self._ocs_http.delete(url)
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.error("Unshare failed %d: %s", share_id, e)
            return False

    async def get_shares(self, path: Optional[str] = None) -> list:
        """List shares, optionally filtered by path."""
        url = f"{self._base_url}/ocs/v2.php/apps/files_sharing/api/v1/shares"
        params = {}
        if path:
            params["path"] = path
        try:
            resp = await self._ocs_http.get(url, params=params)
            resp.raise_for_status()
            return resp.json().get("ocs", {}).get("data", [])
        except Exception as e:
            logger.error("get_shares failed: %s", e)
            return []

    async def close(self):
        await self._http.aclose()
        await self._ocs_http.aclose()
