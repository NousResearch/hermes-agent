"""Connector file inputs and outputs: the model passes local paths, the agent stages them through the
tool gateway and storage, and files a tool returns are saved under the profile's downloads folder."""

import json
from pathlib import Path

import pytest

from hermes_constants import CONNECTOR_DOWNLOADS_SUBDIR, get_hermes_home
from tools.connectors.gateway import files
from tools.connectors.gateway.bridge import connector_describe, run_remote
from tools.connectors.gateway.client import ConnectorClient
from tools.connectors.gateway.merge import partition_calls

GATEWAY = "https://tool-gateway.test"
UPLOAD_URL = "https://storage.test/put?X-Amz-Signature=abc"
DOWNLOAD_URL = "https://storage.test/get?X-Amz-Signature=def"
STAGED_KEY = "project/gmail/GMAIL_SEND_EMAIL/request/key1"

# The vendor's real shape for a mail attachment: one file, or an array of files, behind a $ref.
GMAIL_SEND = {
    "type": "object",
    "properties": {
        "recipient_email": {"type": "string"},
        "attachment": {"description": "File to attach.", "anyOf": [
            {"$ref": "#/$defs/File"}, {"type": "array", "items": {"$ref": "#/$defs/File"}}]},
    },
    "$defs": {"File": {"type": "object", "file_uploadable": True, "description": "A file.",
                       "properties": {"name": {"type": "string"}, "mimetype": {"type": "string"},
                                      "s3key": {"type": "string"}}}},
}


class Response:
    def __init__(self, status_code, body=None, content=b""):
        self.status_code = status_code
        self._body = body
        self._content = content
        self.headers = {"x-request-id": "req-1"}
        self.text = json.dumps(body)

    def json(self):
        return self._body

    def iter_content(self, chunk_size):
        for start in range(0, len(self._content), chunk_size):
            yield self._content[start:start + chunk_size]

    def close(self):
        pass


class Gateway:
    """The tool gateway's connector routes: schemas, upload staging and single-call execute."""

    def __init__(self, schema=GMAIL_SEND, data=None):
        self.schema = schema
        self.data = data if data is not None else {"sent": True}
        self.requests = []

    def request(self, method, url, *, headers=None, json=None, timeout=None):
        self.requests.append({"url": url, "headers": dict(headers or {}), "json": json})
        route = url.removeprefix(f"{GATEWAY}/v1/connectors/")
        if route == "schemas":
            return Response(200, {"schemas": {"GMAIL_SEND_EMAIL": {
                "connector": "gmail", "tool": "GMAIL_SEND_EMAIL", "inputSchema": self.schema}}})
        if route == "files/upload":
            return Response(200, {"uploadUrl": UPLOAD_URL, "headers": {"Content-Type": json["mimetype"]},
                                  "descriptor": {"name": json["filename"], "mimetype": json["mimetype"],
                                                 "s3key": STAGED_KEY},
                                  "expiresAt": "2026-10-14T00:05:00Z"})
        return Response(200, {"result": {"connector": "gmail", "tool": json["call"]["tool"], "data": self.data}})

    def routes(self):
        return [request["url"].removeprefix(f"{GATEWAY}/v1/connectors/") for request in self.requests]


class Storage:
    def __init__(self, content=b"%PDF-1.4 attachment", status=200):
        self.content = content
        self.status = status
        self.requests = []

    def request(self, method, url, *, headers=None, data=None, timeout=None, stream=False):
        self.requests.append({"method": method, "url": url, "headers": dict(headers or {}), "data": data})
        return Response(self.status, content=self.content if method == "GET" else b"")


@pytest.fixture(autouse=True)
def _fresh_schema_cache(monkeypatch):
    monkeypatch.setattr(files, "_FILE_INPUTS", {})
    for name in ("HERMES_MEDIA_DELIVERY_STRICT", "TERMINAL_ENV", "TERMINAL_DOCKER_VOLUMES", "TERMINAL_CWD"):
        monkeypatch.delenv(name, raising=False)


def _client(gateway, storage):
    return ConnectorClient(transport=gateway, storage=storage, endpoint_resolver=lambda: GATEWAY,
                           header_provider=lambda url: {"Authorization": "Bearer nous-token"})


def _execute(gateway, storage, arguments, name="connectors__gmail__SEND_EMAIL"):
    planned = partition_calls([{"name": name, "arguments": arguments}]).remote
    (entry,) = run_remote(planned, "call-1", availability=lambda: True,
                          client_factory=lambda: _client(gateway, storage))
    return entry


def _pdf(path: Path, body=b"%PDF-1.4 local") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


def test_describe_shows_paths_and_records_both_attachment_shapes():
    class Client:
        def schemas(self, slugs):
            return {"schemas": {"GMAIL_SEND_EMAIL": {"connector": "gmail", "input_schema": GMAIL_SEND}}}

    leg = connector_describe(["connectors__gmail__SEND_EMAIL"], availability=lambda: True,
                             client_factory=lambda: Client())

    parameters = leg.payload["tools"]["connectors__gmail__SEND_EMAIL"]["parameters"]
    assert parameters["$defs"]["File"] == {"type": "string", "description": "A file. Pass a local file path."}
    assert "s3key" not in json.dumps(parameters)
    assert files._FILE_INPUTS["connectors__gmail__SEND_EMAIL"] == (("attachment",), ("attachment", "*"))


def test_a_local_path_is_staged_put_to_storage_and_sent_as_the_descriptor(tmp_path):
    gateway, storage = Gateway(), Storage()
    pdf = _pdf(tmp_path / "q3.pdf")

    entry = _execute(gateway, storage, {"recipient_email": "me@example.com", "attachment": str(pdf)})

    assert entry["response"] == {"sent": True}
    # One schema lookup on the cache miss, one stage, one execute; nothing else reaches the gateway.
    assert gateway.routes() == ["schemas", "files/upload", "execute"]
    assert gateway.requests[1]["json"] == {"toolkit": "gmail", "tool": "GMAIL_SEND_EMAIL", "filename": "q3.pdf",
                                           "mimetype": "application/pdf", "size": pdf.stat().st_size}
    assert gateway.requests[2]["json"]["call"]["arguments"]["attachment"] == {
        "name": "q3.pdf", "mimetype": "application/pdf", "s3key": STAGED_KEY}
    (put,) = storage.requests
    assert (put["method"], put["url"], put["data"]) == ("PUT", UPLOAD_URL, pdf.read_bytes())
    # Storage gets exactly the gateway's headers: never the gateway bearer.
    assert put["headers"] == {"Content-Type": "application/pdf"}


def test_every_path_in_an_array_input_is_staged_and_the_cache_spares_a_second_lookup(tmp_path):
    gateway, storage = Gateway(), Storage()
    paths = [str(_pdf(tmp_path / "a.pdf")), str(_pdf(tmp_path / "b.pdf"))]

    _execute(gateway, storage, {"attachment": paths})
    _execute(gateway, storage, {"attachment": paths[0]})

    assert gateway.routes() == ["schemas", "files/upload", "files/upload", "execute", "files/upload", "execute"]
    assert [put["data"] for put in storage.requests] == [b"%PDF-1.4 local"] * 3


@pytest.mark.parametrize(("make_path", "code"), [
    (lambda tmp: _pdf(get_hermes_home() / ".env"), "FILE_NOT_ALLOWED"),  # credential store: denied
    (lambda tmp: tmp / "missing.pdf", "FILE_NOT_FOUND"),
    (lambda tmp: tmp, "FILE_NOT_FOUND"),  # a directory is not a regular file
])
def test_a_refused_path_fails_the_call_before_any_upload(tmp_path, make_path, code):
    gateway, storage = Gateway(), Storage()

    entry = _execute(gateway, storage, {"attachment": str(make_path(tmp_path))})

    assert entry["error"]["code"] == code and entry["error"]["family"] == "product"
    assert gateway.routes() == ["schemas"] and storage.requests == []


def test_a_file_over_the_gateway_cap_is_refused_before_staging(tmp_path, monkeypatch):
    monkeypatch.setattr(files, "CONNECTOR_FILE_MAX_BYTES", 4)
    gateway, storage = Gateway(), Storage()

    entry = _execute(gateway, storage, {"attachment": str(_pdf(tmp_path / "big.pdf"))})

    assert entry["error"]["code"] == "FILE_TOO_LARGE"
    assert "files/upload" not in gateway.routes() and storage.requests == []


def test_a_failed_storage_put_never_executes_the_call(tmp_path):
    gateway, storage = Gateway(), Storage(status=403)

    entry = _execute(gateway, storage, {"attachment": str(_pdf(tmp_path / "q3.pdf"))})

    assert entry["error"]["code"] == "FILE_UPLOAD_FAILED" and entry["error"]["family"] == "backend"
    assert "execute" not in gateway.routes()
    assert UPLOAD_URL not in json.dumps(entry)


def test_a_returned_file_is_saved_and_the_model_sees_only_its_path():
    output = {"name": "invoice.pdf", "mimetype": "application/pdf", "url": DOWNLOAD_URL,
              "expiresAt": "2026-10-14T01:00:00Z"}
    gateway = Gateway(data={"attachments": [output]})
    storage = Storage(content=b"%PDF received")

    entry = _execute(gateway, storage, {"message_id": "m1"})

    (saved,) = entry["response"]["attachments"]
    assert set(saved) == {"name", "mimetype", "path"}
    path = Path(saved["path"])
    assert path.parent == get_hermes_home() / CONNECTOR_DOWNLOADS_SUBDIR
    assert path.name.endswith("-invoice.pdf") and path.read_bytes() == b"%PDF received"
    assert [request["headers"] for request in storage.requests] == [{}]
    assert "X-Amz" not in json.dumps(entry)


def test_a_second_download_of_the_same_name_never_overwrites_the_first():
    output = {"name": "a.pdf", "mimetype": "application/pdf", "url": DOWNLOAD_URL}
    first = _execute(Gateway(data=output), Storage(content=b"one"), {"id": "1"})["response"]["path"]
    second = _execute(Gateway(data=output), Storage(content=b"two"), {"id": "2"})["response"]["path"]

    assert first != second
    assert (Path(first).read_bytes(), Path(second).read_bytes()) == (b"one", b"two")


def test_a_failed_download_keeps_the_result_and_names_the_file_error():
    output = {"name": "a.pdf", "mimetype": "application/pdf", "url": DOWNLOAD_URL}
    entry = _execute(Gateway(data={"file": output, "subject": "hi"}), Storage(status=404), {"id": "1"})

    assert entry["response"]["subject"] == "hi"
    assert entry["response"]["file"]["error"].startswith("download failed")
    assert "url" not in entry["response"]["file"]
    assert not list((get_hermes_home() / CONNECTOR_DOWNLOADS_SUBDIR).iterdir())


def test_a_replayed_file_without_a_link_passes_through():
    replay = {"name": "a.pdf", "mimetype": "application/pdf"}
    storage = Storage()

    entry = _execute(Gateway(data=replay), storage, {"id": "1"})

    assert entry["response"] == replay and storage.requests == []


def test_strict_mode_refuses_a_fresh_file_but_resends_a_downloaded_one(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "1")
    monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "0")
    output = {"name": "invoice.pdf", "mimetype": "application/pdf", "url": DOWNLOAD_URL}
    downloaded = _execute(Gateway(data=output), Storage(content=b"%PDF received"), {"id": "1"})["response"]["path"]

    refused = _execute(Gateway(), Storage(), {"attachment": str(_pdf(tmp_path / "fresh.pdf"))})
    gateway, storage = Gateway(), Storage()
    resent = _execute(gateway, storage, {"attachment": downloaded})

    assert refused["error"]["code"] == "FILE_NOT_ALLOWED"
    assert resent["response"] == {"sent": True}
    assert [put["data"] for put in storage.requests] == [b"%PDF received"]


def test_a_search_hit_shows_paths_so_execute_needs_no_schema_lookup(tmp_path):
    from tools.connectors.gateway.bridge import ConnectorLeg
    from tools.connectors.search import connector_entries_by_group

    hits = {"results": [{"use_case": "email a file", "tools": ["GMAIL_SEND_EMAIL"]}],
            "schemas": {"GMAIL_SEND_EMAIL": {"connector": "gmail", "description": "Send", "input_schema": GMAIL_SEND}}}
    (entries,), failure = connector_entries_by_group(["email a file"], connector_search=lambda q: ConnectorLeg(hits))
    gateway, storage = Gateway(), Storage()

    _execute(gateway, storage, {"attachment": str(_pdf(tmp_path / "q3.pdf"))})

    assert failure is None and "s3key" not in json.dumps(entries[0].schema)
    assert gateway.routes() == ["files/upload", "execute"]


def test_a_relative_path_resolves_against_the_session_working_directory(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    _pdf(workspace / "q3.pdf", b"from the workspace")
    monkeypatch.setenv("TERMINAL_CWD", str(workspace))
    storage = Storage()

    _execute(Gateway(), storage, {"attachment": "q3.pdf"})

    assert [put["data"] for put in storage.requests] == [b"from the workspace"]


def test_dispatch_resolves_a_relative_path_in_the_calling_task_s_directory(tmp_path, monkeypatch):
    import model_tools
    from tools.connectors.gateway import bridge, config
    from tools.registry import invalidate_check_fn_cache
    from tools.terminal_tool import clear_task_env_overrides, register_task_env_overrides

    task_dir = tmp_path / "task"
    _pdf(task_dir / "q3.pdf", b"from the task directory")
    _pdf(tmp_path / "other" / "q3.pdf", b"from another directory")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path / "other"))
    gateway, storage = Gateway(), Storage()
    monkeypatch.setattr(config, "connectors_available", lambda: True)
    monkeypatch.setattr(bridge, "connectors_available", lambda: True)
    monkeypatch.setattr(bridge, "_default_client_factory", lambda call=None: _client(gateway, storage))
    invalidate_check_fn_cache()
    register_task_env_overrides("task-files", {"cwd": str(task_dir)})
    try:
        model_tools.handle_function_call(
            "connectors__gmail__SEND_EMAIL", {"attachment": "q3.pdf"}, task_id="task-files",
            enabled_toolsets=["connections"], session_id="s", skip_pre_tool_call_hook=True,
            skip_tool_request_middleware=True, skip_tool_execution_middleware=True)
    finally:
        clear_task_env_overrides("task-files")

    assert [put["data"] for put in storage.requests] == [b"from the task directory"]


def test_a_docker_path_is_read_through_its_volume_mount(tmp_path, monkeypatch):
    host = tmp_path / "shared"
    _pdf(host / "report.pdf", b"from the container mount")
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", json.dumps([f"{host}:/data"]))
    gateway, storage = Gateway(), Storage()

    entry = _execute(gateway, storage, {"attachment": "/data/report.pdf"})

    assert entry["response"] == {"sent": True}
    assert gateway.requests[1]["json"]["filename"] == "report.pdf"
    assert [put["data"] for put in storage.requests] == [b"from the container mount"]


def test_a_remote_terminal_backend_is_refused_like_media_delivery(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    gateway, storage = Gateway(), Storage()

    entry = _execute(gateway, storage, {"attachment": "/home/me/q3.pdf"})

    assert entry["error"]["code"] == "FILE_NOT_ALLOWED" and "ssh" in entry["error"]["message"]
    assert "files/upload" not in gateway.routes() and storage.requests == []


def test_a_storage_descriptor_from_the_model_is_refused_with_a_hint():
    entry = _execute(Gateway(), Storage(), {"attachment": {"name": "a", "mimetype": "b", "s3key": "x"}})

    assert entry["error"]["code"] == "FILE_NOT_ALLOWED" and "path" in entry["error"]["hint"]
