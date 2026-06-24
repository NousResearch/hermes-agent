"""
S3-backed storage backend for Hermes.

Provides a simple key/value store interface backed by Amazon S3 (or any
S3-compatible object store).  All public methods are intentionally kept
thin so that callers never have to import boto3 directly.
"""

from __future__ import annotations

import io
import logging
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)


class S3BackendError(Exception):
    """Raised when an S3 operation fails in an unrecoverable way."""


class S3Backend:
    """Key/value store backed by an S3 bucket.

    Parameters
    ----------
    bucket:
        Name of the S3 bucket to use.
    prefix:
        Optional key prefix (folder) that is prepended to every object key.
        A trailing ``/`` is added automatically if *prefix* is non-empty and
        does not already end with one.
    client:
        A pre-constructed ``boto3`` S3 client.  When *None* a default client
        is created via ``boto3.client("s3")``.  Injecting a client makes unit
        testing straightforward without hitting real AWS.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        client=None,
    ) -> None:
        if not bucket:
            raise ValueError("bucket must be a non-empty string")

        self._bucket = bucket

        # Normalise prefix so it always ends with "/" when non-empty.
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"
        self._prefix = prefix

        if client is None:
            import boto3  # lazy import so the module is usable without boto3 installed

            client = boto3.client("s3")
        self._client = client

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _full_key(self, key: str) -> str:
        """Return the full S3 object key for a logical *key*."""
        if not key:
            raise ValueError("key must be a non-empty string")
        return f"{self._prefix}{key}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def put(self, key: str, data: bytes) -> None:
        """Upload *data* to S3 under *key*.

        Parameters
        ----------
        key:
            Logical key (relative to the configured prefix).
        data:
            Raw bytes to store.

        Raises
        ------
        S3BackendError
            If the upload fails for any reason.
        """
        full_key = self._full_key(key)
        try:
            self._client.put_object(Bucket=self._bucket, Key=full_key, Body=data)
            logger.debug("put s3://%s/%s (%d bytes)", self._bucket, full_key, len(data))
        except Exception as exc:
            raise S3BackendError(
                f"Failed to put object s3://{self._bucket}/{full_key}: {exc}"
            ) from exc

    def get(self, key: str) -> bytes:
        """Download and return the bytes stored under *key*.

        Parameters
        ----------
        key:
            Logical key (relative to the configured prefix).

        Returns
        -------
        bytes
            The raw object body.

        Raises
        ------
        KeyError
            If the key does not exist in the bucket.
        S3BackendError
            If the download fails for any other reason.
        """
        full_key = self._full_key(key)
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=full_key)
            body: bytes = response["Body"].read()
            logger.debug("get s3://%s/%s (%d bytes)", self._bucket, full_key, len(body))
            return body
        except self._client.exceptions.NoSuchKey:
            raise KeyError(key)
        except Exception as exc:
            # Distinguish "not found" errors reported via ClientError.
            error_code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if error_code in ("NoSuchKey", "404"):
                raise KeyError(key) from exc
            raise S3BackendError(
                f"Failed to get object s3://{self._bucket}/{full_key}: {exc}"
            ) from exc

    def delete(self, key: str) -> None:
        """Delete the object stored under *key*.

        Deleting a key that does not exist is a no-op (S3 semantics).

        Parameters
        ----------
        key:
            Logical key (relative to the configured prefix).

        Raises
        ------
        S3BackendError
            If the deletion fails for any reason other than the key not
            existing.
        """
        full_key = self._full_key(key)
        try:
            self._client.delete_object(Bucket=self._bucket, Key=full_key)
            logger.debug("delete s3://%s/%s", self._bucket, full_key)
        except Exception as exc:
            raise S3BackendError(
                f"Failed to delete object s3://{self._bucket}/{full_key}: {exc}"
            ) from exc

    def exists(self, key: str) -> bool:
        """Return *True* if *key* exists in the bucket, *False* otherwise.

        Parameters
        ----------
        key:
            Logical key (relative to the configured prefix).

        Raises
        ------
        S3BackendError
            If the head-object call fails for a reason other than the object
            not existing.
        """
        full_key = self._full_key(key)
        try:
            self._client.head_object(Bucket=self._bucket, Key=full_key)
            return True
        except Exception as exc:
            error_code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if error_code in ("NoSuchKey", "404", "403"):
                return False
            # Re-raise unexpected errors.
            raise S3BackendError(
                f"Failed to check existence of s3://{self._bucket}/{full_key}: {exc}"
            ) from exc

    def list_keys(self, sub_prefix: str = "") -> List[str]:
        """Return a list of logical keys under an optional *sub_prefix*.

        Parameters
        ----------
        sub_prefix:
            Additional prefix to filter keys.  Combined with the backend's
            own prefix.

        Returns
        -------
        list[str]
            Logical keys (with the backend prefix stripped).

        Raises
        ------
        S3BackendError
            If the listing fails.
        """
        return list(self.iter_keys(sub_prefix=sub_prefix))

    def iter_keys(self, sub_prefix: str = "") -> Iterator[str]:
        """Yield logical keys under an optional *sub_prefix* lazily.

        Unlike :meth:`list_keys`, this method is a generator that yields keys
        one at a time as pages are retrieved from S3, avoiding the need to
        hold all keys in memory simultaneously.

        Parameters
        ----------
        sub_prefix:
            Additional prefix to filter keys.  Combined with the backend's
            own prefix.

        Yields
        ------
        str
            Logical keys (with the backend prefix stripped).

        Raises
        ------
        S3BackendError
            If the listing fails.
        """
        search_prefix = f"{self._prefix}{sub_prefix}"
        try:
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._bucket, Prefix=search_prefix):
                for obj in page.get("Contents", []):
                    full_key: str = obj["Key"]
                    # Strip the backend prefix so callers get logical keys.
                    logical_key = full_key[len(self._prefix):]
                    yield logical_key
        except S3BackendError:
            raise
        except Exception as exc:
            raise S3BackendError(
                f"Failed to list objects in s3://{self._bucket}/{search_prefix}: {exc}"
            ) from exc

    def copy(self, src_key: str, dst_key: str) -> None:
        """Copy an object within the same bucket from *src_key* to *dst_key*.

        Both keys are resolved relative to the backend's configured prefix.

        Parameters
        ----------
        src_key:
            Logical source key (relative to the configured prefix).
        dst_key:
            Logical destination key (relative to the configured prefix).

        Raises
        ------
        KeyError
            If *src_key* does not exist in the bucket.
        S3BackendError
            If the copy fails for any other reason.
        """
        full_src = self._full_key(src_key)
        full_dst = self._full_key(dst_key)
        copy_source = {"Bucket": self._bucket, "Key": full_src}
        try:
            self._client.copy_object(
                CopySource=copy_source,
                Bucket=self._bucket,
                Key=full_dst,
            )
            logger.debug(
                "copy s3://%s/%s -> s3://%s/%s",
                self._bucket,
                full_src,
                self._bucket,
                full_dst,
            )
        except Exception as exc:
            error_code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if error_code in ("NoSuchKey", "404"):
                raise KeyError(src_key) from exc
            raise S3BackendError(
                f"Failed to copy s3://{self._bucket}/{full_src} -> "
                f"s3://{self._bucket}/{full_dst}: {exc}"
            ) from exc

    def put_text(self, key: str, text: str, encoding: str = "utf-8") -> None:
        """Convenience wrapper: encode *text* and call :meth:`put`."""
        self.put(key, text.encode(encoding))

    def get_text(self, key: str, encoding: str = "utf-8") -> str:
        """Convenience wrapper: call :meth:`get` and decode the result."""
        return self.get(key).decode(encoding)

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "S3Backend":
        return self

    def __exit__(self, *_: object) -> None:
        pass  # nothing to close; boto3 manages its own connection pool

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"S3Backend(bucket={self._bucket!r}, prefix={self._prefix!r})"
        )
