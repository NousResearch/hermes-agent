"""
Unit tests for hermes_storage.s3_backend.

All tests use a lightweight fake S3 client so that no real AWS credentials or
network access are required.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, call
from io import BytesIO

from hermes_storage.s3_backend import S3Backend, S3BackendError


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_client(objects: dict | None = None):
    """Return a minimal fake boto3 S3 client backed by an in-memory dict.

    Parameters
    ----------
    objects:
        Pre-populated ``{key: bytes}`` mapping that represents the initial
        state of the bucket.
    """
    store: dict[str, bytes] = dict(objects or {})

    client = MagicMock()

    # ---- put_object -------------------------------------------------------
    def _put_object(Bucket, Key, Body):
        store[Key] = Body if isinstance(Body, bytes) else Body.read()
        return {}

    client.put_object.side_effect = _put_object

    # ---- get_object -------------------------------------------------------
    class _NoSuchKey(Exception):
        response = {"Error": {"Code": "NoSuchKey"}}

    client.exceptions.NoSuchKey = _NoSuchKey

    def _get_object(Bucket, Key):
        if Key not in store:
            raise _NoSuchKey(f"NoSuchKey: {Key}")
        return {"Body": BytesIO(store[Key])}

    client.get_object.side_effect = _get_object

    # ---- delete_object ----------------------------------------------------
    def _delete_object(Bucket, Key):
        store.pop(Key, None)
        return {}

    client.delete_object.side_effect = _delete_object

    # ---- head_object ------------------------------------------------------
    class _ClientError(Exception):
        def __init__(self, code):
            self.response = {"Error": {"Code": code}}

    def _head_object(Bucket, Key):
        if Key not in store:
            raise _ClientError("404")
        return {"ContentLength": len(store[Key])}

    client.head_object.side_effect = _head_object

    # ---- copy_object ------------------------------------------------------
    class _CopyNoSuchKey(Exception):
        def __init__(self):
            self.response = {"Error": {"Code": "NoSuchKey"}}

    def _copy_object(CopySource, Bucket, Key):
        src_key = CopySource["Key"]
        if src_key not in store:
            raise _CopyNoSuchKey()
        store[Key] = store[src_key]
        return {}

    client.copy_object.side_effect = _copy_object

    # ---- list_objects_v2 (via paginator) ----------------------------------
    def _paginate(Bucket, Prefix=""):
        matching = [{"Key": k} for k in store if k.startswith(Prefix)]
        yield {"Contents": matching} if matching else {}

    paginator = MagicMock()
    paginator.paginate.side_effect = lambda Bucket, Prefix="": _paginate(
        Bucket, Prefix
    )
    client.get_paginator.return_value = paginator

    # Expose the internal store for assertions in tests.
    client._store = store

    return client


@pytest.fixture()
def client():
    return _make_client()


@pytest.fixture()
def backend(client):
    return S3Backend(bucket="test-bucket", prefix="hermes/", client=client)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_requires_non_empty_bucket(self):
        with pytest.raises(ValueError, match="bucket"):
            S3Backend(bucket="", client=MagicMock())

    def test_prefix_normalised_with_trailing_slash(self):
        b = S3Backend(bucket="b", prefix="my-prefix", client=MagicMock())
        assert b._prefix == "my-prefix/"

    def test_prefix_already_has_trailing_slash(self):
        b = S3Backend(bucket="b", prefix="my-prefix/", client=MagicMock())
        assert b._prefix == "my-prefix/"

    def test_empty_prefix_stays_empty(self):
        b = S3Backend(bucket="b", prefix="", client=MagicMock())
        assert b._prefix == ""

    def test_repr_contains_bucket_and_prefix(self):
        b = S3Backend(bucket="mybucket", prefix="p/", client=MagicMock())
        r = repr(b)
        assert "mybucket" in r
        assert "p/" in r

    def test_context_manager_returns_self(self, backend):
        with backend as b:
            assert b is backend


# ---------------------------------------------------------------------------
# put / get
# ---------------------------------------------------------------------------


class TestPutGet:
    def test_put_stores_bytes(self, backend, client):
        backend.put("file.txt", b"hello")
        assert client._store["hermes/file.txt"] == b"hello"

    def test_get_retrieves_bytes(self, backend, client):
        client._store["hermes/file.txt"] = b"world"
        assert backend.get("file.txt") == b"world"

    def test_get_missing_key_raises_key_error(self, backend):
        with pytest.raises(KeyError):
            backend.get("does-not-exist")

    def test_put_then_get_roundtrip(self, backend):
        data = b"\x00\x01\x02\xff"
        backend.put("binary", data)
        assert backend.get("binary") == data

    def test_put_overwrites_existing(self, backend):
        backend.put("k", b"v1")
        backend.put("k", b"v2")
        assert backend.get("k") == b"v2"

    def test_put_empty_key_raises_value_error(self, backend):
        with pytest.raises(ValueError, match="key"):
            backend.put("", b"data")

    def test_get_empty_key_raises_value_error(self, backend):
        with pytest.raises(ValueError, match="key"):
            backend.get("")

    def test_put_propagates_client_error(self, backend, client):
        client.put_object.side_effect = RuntimeError("network error")
        with pytest.raises(S3BackendError, match="network error"):
            backend.put("k", b"v")

    def test_get_propagates_unexpected_client_error(self, backend, client):
        client.get_object.side_effect = RuntimeError("unexpected")
        with pytest.raises(S3BackendError, match="unexpected"):
            backend.get("k")


# ---------------------------------------------------------------------------
# put_text / get_text
# ---------------------------------------------------------------------------


class TestTextHelpers:
    def test_put_text_encodes_utf8_by_default(self, backend, client):
        backend.put_text("t.txt", "héllo")
        assert client._store["hermes/t.txt"] == "héllo".encode("utf-8")

    def test_get_text_decodes_utf8_by_default(self, backend, client):
        client._store["hermes/t.txt"] = "héllo".encode("utf-8")
        assert backend.get_text("t.txt") == "héllo"

    def test_put_text_custom_encoding(self, backend, client):
        backend.put_text("t.txt", "hello", encoding="ascii")
        assert client._store["hermes/t.txt"] == b"hello"

    def test_get_text_custom_encoding(self, backend, client):
        client._store["hermes/t.txt"] = "hello".encode("latin-1")
        assert backend.get_text("t.txt", encoding="latin-1") == "hello"

    def test_text_roundtrip(self, backend):
        backend.put_text("msg", "こんにちは")
        assert backend.get_text("msg") == "こんにちは"


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_removes_key(self, backend, client):
        client._store["hermes/k"] = b"v"
        backend.delete("k")
        assert "hermes/k" not in client._store

    def test_delete_nonexistent_key_is_noop(self, backend):
        # Should not raise.
        backend.delete("ghost")

    def test_delete_propagates_client_error(self, backend, client):
        client.delete_object.side_effect = RuntimeError("boom")
        with pytest.raises(S3BackendError, match="boom"):
            backend.delete("k")

    def test_delete_empty_key_raises_value_error(self, backend):
        with pytest.raises(ValueError, match="key"):
            backend.delete("")


# ---------------------------------------------------------------------------
# exists
# ---------------------------------------------------------------------------


class TestExists:
    def test_exists_true_when_key_present(self, backend, client):
        client._store["hermes/present"] = b"data"
        assert backend.exists("present") is True

    def test_exists_false_when_key_absent(self, backend):
        assert backend.exists("absent") is False

    def test_exists_propagates_unexpected_error(self, backend, client):
        client.head_object.side_effect = RuntimeError("unexpected")
        with pytest.raises(S3BackendError):
            backend.exists("k")


# ---------------------------------------------------------------------------
# list_keys
# ---------------------------------------------------------------------------


class TestListKeys:
    def test_list_keys_returns_logical_keys(self, backend, client):
        client._store["hermes/a"] = b""
        client._store["hermes/b"] = b""
        client._store["hermes/c"] = b""
        keys = backend.list_keys()
        assert sorted(keys) == ["a", "b", "c"]

    def test_list_keys_with_sub_prefix(self, backend, client):
        client._store["hermes/logs/2024-01"] = b""
        client._store["hermes/logs/2024-02"] = b""
        client._store["hermes/other"] = b""
        keys = backend.list_keys(sub_prefix="logs/")
        assert sorted(keys) == ["logs/2024-01", "logs/2024-02"]

    def test_list_keys_empty_bucket(self, backend):
        assert backend.list_keys() == []

    def test_list_keys_propagates_client_error(self, backend, client):
        client.get_paginator.side_effect = RuntimeError("paginator error")
        with pytest.raises(S3BackendError, match="paginator error"):
            backend.list_keys()

    def test_list_keys_no_prefix_backend(self, client):
        """Backend with no prefix should return keys as-is."""
        b = S3Backend(bucket="test-bucket", prefix="", client=client)
        client._store["raw-key"] = b""
        keys = b.list_keys()
        assert "raw-key" in keys

    def test_list_keys_uses_full_prefix_for_search(self, backend, client):
        """Ensure the paginator is called with the combined prefix."""
        client._store["hermes/sub/item"] = b""
        backend.list_keys(sub_prefix="sub/")
        paginator = client.get_paginator.return_value
        paginator.paginate.assert_called_once_with(
            Bucket="test-bucket", Prefix="hermes/sub/"
        )

    def test_list_keys_returns_list_not_iterator(self, backend, client):
        """list_keys must return a concrete list, not a lazy iterator."""
        client._store["hermes/x"] = b""
        result = backend.list_keys()
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# iter_keys
# ---------------------------------------------------------------------------


class TestIterKeys:
    def test_iter_keys_yields_logical_keys(self, backend, client):
        client._store["hermes/a"] = b""
        client._store["hermes/b"] = b""
        keys = list(backend.iter_keys())
        assert sorted(keys) == ["a", "b"]

    def test_iter_keys_with_sub_prefix(self, backend, client):
        client._store["hermes/logs/2024-01"] = b""
        client._store["hermes/logs/2024-02"] = b""
        client._store["hermes/other"] = b""
        keys = list(backend.iter_keys(sub_prefix="logs/"))
        assert sorted(keys) == ["logs/2024-01", "logs/2024-02"]

    def test_iter_keys_empty_bucket(self, backend):
        assert list(backend.iter_keys()) == []

    def test_iter_keys_returns_iterator(self, backend, client):
        """iter_keys must return a lazy iterator, not a list."""
        client._store["hermes/x"] = b""
        result = backend.iter_keys()
        # Should be an iterator (has __next__), not a list
        assert hasattr(result, "__next__")

    def test_iter_keys_propagates_client_error(self, backend, client):
        client.get_paginator.side_effect = RuntimeError("paginator error")
        with pytest.raises(S3BackendError, match="paginator error"):
            list(backend.iter_keys())

    def test_iter_keys_uses_full_prefix_for_search(self, backend, client):
        """Ensure the paginator is called with the combined prefix."""
        client._store["hermes/sub/item"] = b""
        list(backend.iter_keys(sub_prefix="sub/"))
        paginator = client.get_paginator.return_value
        paginator.paginate.assert_called_once_with(
            Bucket="test-bucket", Prefix="hermes/sub/"
        )

    def test_iter_keys_no_prefix_backend(self, client):
        """Backend with no prefix should yield keys as-is."""
        b = S3Backend(bucket="test-bucket", prefix="", client=client)
        client._store["raw-key"] = b""
        keys = list(b.iter_keys())
        assert "raw-key" in keys

    def test_list_keys_delegates_to_iter_keys(self, backend, client):
        """list_keys should produce the same results as list(iter_keys())."""
        client._store["hermes/p"] = b""
        client._store["hermes/q"] = b""
        assert sorted(backend.list_keys()) == sorted(list(backend.iter_keys()))


# ---------------------------------------------------------------------------
# copy
# ---------------------------------------------------------------------------


class TestCopy:
    def test_copy_duplicates_object(self, backend, client):
        client._store["hermes/src"] = b"payload"
        backend.copy("src", "dst")
        assert client._store["hermes/dst"] == b"payload"

    def test_copy_does_not_remove_source(self, backend, client):
        client._store["hermes/src"] = b"payload"
        backend.copy("src", "dst")
        assert client._store["hermes/src"] == b"payload"

    def test_copy_missing_src_raises_key_error(self, backend):
        with pytest.raises(KeyError):
            backend.copy("nonexistent", "dst")

    def test_copy_overwrites_existing_dst(self, backend, client):
        client._store["hermes/src"] = b"new"
        client._store["hermes/dst"] = b"old"
        backend.copy("src", "dst")
        assert client._store["hermes/dst"] == b"new"

    def test_copy_empty_src_key_raises_value_error(self, backend):
        with pytest.raises(ValueError, match="key"):
            backend.copy("", "dst")

    def test_copy_empty_dst_key_raises_value_error(self, backend):
        with pytest.raises(ValueError, match="key"):
            backend.copy("src", "")

    def test_copy_calls_copy_object_with_correct_args(self, backend, client):
        client._store["hermes/src"] = b"data"
        backend.copy("src", "dst")
        client.copy_object.assert_called_once_with(
            CopySource={"Bucket": "test-bucket", "Key": "hermes/src"},
            Bucket="test-bucket",
            Key="hermes/dst",
        )

    def test_copy_propagates_unexpected_client_error(self, backend, client):
        client._store["hermes/src"] = b"data"
        client.copy_object.side_effect = RuntimeError("network failure")
        with pytest.raises(S3BackendError, match="network failure"):
            backend.copy("src", "dst")

    def test_copy_respects_prefix(self, client):
        """copy uses the backend prefix for both src and dst keys."""
        b = S3Backend(bucket="test-bucket", prefix="ns/", client=client)
        client._store["ns/orig"] = b"val"
        b.copy("orig", "clone")
        assert client._store["ns/clone"] == b"val"


# ---------------------------------------------------------------------------
# Prefix isolation
# ---------------------------------------------------------------------------


class TestPrefixIsolation:
    def test_two_backends_same_bucket_different_prefix_are_isolated(self, client):
        b1 = S3Backend(bucket="test-bucket", prefix="ns1/", client=client)
        b2 = S3Backend(bucket="test-bucket", prefix="ns2/", client=client)

        b1.put("key", b"from-b1")
        b2.put("key", b"from-b2")

        assert b1.get("key") == b"from-b1"
        assert b2.get("key") == b"from-b2"

    def test_list_keys_does_not_bleed_across_prefixes(self, client):
        b1 = S3Backend(bucket="test-bucket", prefix="ns1/", client=client)
        b2 = S3Backend(bucket="test-bucket", prefix="ns2/", client=client)

        b1.put("shared-name", b"1")
        b2.put("shared-name", b"2")
        b2.put("extra", b"x")

        assert b1.list_keys() == ["shared-name"]
        assert sorted(b2.list_keys()) == ["extra", "shared-name"]

    def test_copy_does_not_bleed_across_prefixes(self, client):
        b1 = S3Backend(bucket="test-bucket", prefix="ns1/", client=client)
        b2 = S3Backend(bucket="test-bucket", prefix="ns2/", client=client)

        b1.put("item", b"b1-data")
        b2.put("item", b"b2-data")

        b1.copy("item", "item-copy")

        # b1's copy should exist under ns1/
        assert b1.get("item-copy") == b"b1-data"
        # b2 should be unaffected
        assert b2.list_keys() == ["item"]
