import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "jiuwenmemory_server_audit.py"
spec = importlib.util.spec_from_file_location("jiuwenmemory_server_audit", SCRIPT_PATH)
audit = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


VALID_MEMORY_SERVER = '''
from pydantic import BaseModel

_GRAPH_DB_PATH = "graph_memory.sqlite3"
_TURBO_DB_PATH = "memory_turbo.sqlite3"
_SWARM_DB_PATH = "swarm_memory.sqlite3"


class AddMessagesRequest(BaseModel):
    session_id: str = ""


class DreamingStartRequest(BaseModel):
    pass


class DreamingStopRequest(BaseModel):
    pass


class GraphEntityRequest(BaseModel):
    pass


class GraphRelationRequest(BaseModel):
    pass


class GraphExtractRequest(BaseModel):
    session_id: str = ""


class GraphQueryRequest(BaseModel):
    pass


class SwarmPromoteRequest(BaseModel):
    pass


class SwarmSearchRequest(BaseModel):
    pass


def _dreaming_config_from_env():
    pass


def _dreaming_status_payload():
    pass


def _turbo_enqueue():
    pass


async def _turbo_process_next_pending():
    pass


def _turbo_status_payload():
    pass


def _graph_connect():
    pass


def _graph_extract_heuristic():
    pass


def _swarm_insert_record():
    pass


def _swarm_status_payload():
    pass


@app.post("/turbo/add_messages_async")
async def turbo_add_messages_async_endpoint():
    pass


@app.get("/turbo/status")
async def turbo_status_endpoint():
    pass


@app.post("/turbo/process_once")
async def turbo_process_once_endpoint():
    pass


@app.post("/swarm/promote")
async def swarm_promote_endpoint():
    pass


@app.get("/swarm/status")
async def swarm_status_endpoint():
    pass


@app.post("/swarm/search")
async def swarm_search_endpoint():
    pass


@app.get("/graph/status")
async def graph_status_endpoint():
    pass


@app.post("/graph/extract")
async def graph_extract_endpoint():
    pass


@app.post("/graph/query")
async def graph_query_endpoint():
    pass


@app.get("/dreaming/status")
async def dreaming_status_endpoint():
    pass


@app.post("/dreaming/start")
async def dreaming_start_endpoint():
    pass


@app.post("/dreaming/stop")
async def dreaming_stop_endpoint():
    pass
'''


def _write_server(tmp_path, source=VALID_MEMORY_SERVER):
    server_file = tmp_path / "memory_server.py"
    server_file.write_text(source, encoding="utf-8")
    return server_file


def test_audit_memory_server_accepts_required_poc_surface(tmp_path):
    server_file = _write_server(tmp_path)

    report = audit.audit_memory_server(server_file)

    assert report.ok is True
    assert report.missing() == []
    assert report.sha256


def test_audit_memory_server_reports_missing_required_features(tmp_path):
    source = VALID_MEMORY_SERVER.replace('@app.post("/graph/query")', '@app.post("/graph/search")')
    source = source.replace("class SwarmSearchRequest(BaseModel):", "class SwarmLookupRequest(BaseModel):")
    server_file = _write_server(tmp_path, source)

    report = audit.audit_memory_server(server_file)

    assert report.ok is False
    missing = {feature.name for feature in report.missing()}
    assert "route:POST /graph/query" in missing
    assert "model:SwarmSearchRequest" in missing


def test_export_server_artifacts_writes_backup_patch_and_manifest(tmp_path):
    server_file = _write_server(tmp_path)
    output_dir = tmp_path / "out"

    result = audit.export_server_artifacts(
        server_file,
        output_dir,
        timestamp="20260709T000000Z",
    )

    backup_file = Path(result.backup_file)
    patch_file = Path(result.patch_file)
    manifest_file = Path(result.manifest_file)
    assert backup_file.read_text(encoding="utf-8") == VALID_MEMORY_SERVER
    patch = patch_file.read_text(encoding="utf-8")
    assert "--- /dev/null" in patch
    assert "+++ b/memory_server.py" in patch
    assert "+class AddMessagesRequest(BaseModel):" in patch
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["note"] == audit.POC_NOTE
    assert manifest["source_sha256"] == result.audit.sha256
    assert manifest["audit"]["ok"] is True
