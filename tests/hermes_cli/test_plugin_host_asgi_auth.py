"""A dashboard-forwarded plugin request must satisfy the plugin-side ``_require_token`` guard.

``plugins.isolation: host`` serves a profile's dashboard plugins on a bare ``FastAPI()`` inside
the host process (``HostRuntime.op_asgi``), reached only through the dashboard's authenticated
forward. The host mints its own unrelated ``_SESSION_TOKEN`` and the gate injects no token at
all, so without the ``dashboard_forwarded`` marker every ``_require_token`` endpoint 401s even
though the dashboard already authenticated the caller (#140499).
"""

from fastapi import APIRouter, FastAPI, Request
from fastapi.testclient import TestClient

from hermes_cli.plugin_host_child import HostRuntime
from hermes_cli.plugin_host_wire import decode

# The documented dashboard-plugin pattern: mutating endpoints guarded by the shared
# ``_require_token`` seam, late-bound so it resolves inside whichever process imports it.
PLUGIN_API = """
from fastapi import APIRouter, Request

from hermes_cli.web_deps import late

_require_token = late("_require_token")
router = APIRouter()


@router.put("/config")
async def put_config(request: Request):
    _require_token(request)
    return {"ok": True}


@router.get("/config")
async def get_config():
    return {"ok": True}
"""


def test_dashboard_forwarded_request_satisfies_require_token(tmp_path):
    runtime = HostRuntime(reader=None, writer=None)
    (tmp_path / "api.py").write_text(PLUGIN_API, encoding="utf-8")
    result = decode(
        runtime.op_asgi(
            {
                "plugin": "demo",
                "dashboard_dir": str(tmp_path),
                "api_file": "api.py",
                "method": "PUT",
                "path": "/config",
                "query": "",
                "headers": [],
                "body": b"",
            }
        )
    )
    assert result["status"] == 200


def test_require_token_still_rejects_unforwarded_requests():
    # Same bare-app shape the host builds, but WITHOUT the forward marker: directly mounted
    # routers keep failing closed when the caller brings no credentials.
    from hermes_cli.web_server import _require_token

    router = APIRouter()

    @router.put("/config")
    async def put_config(request: Request):
        _require_token(request)
        return {"ok": True}

    app = FastAPI()
    app.include_router(router)
    assert TestClient(app).put("/config").status_code == 401
