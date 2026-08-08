"""HTTP MCP transport must not be gated on a symbol the SDK has removed.

mcp >= 1.24.0 dropped the deprecated ``streamablehttp_client`` alias in favour
of ``streamable_http_client``. ``_MCP_HTTP_AVAILABLE`` is set only from the
deprecated import, so on a current SDK every HTTP MCP server is parked at
startup with:

    MCP server '<name>' requires HTTP transport but
    mcp.client.streamable_http is not available. Upgrade the mcp package.

which is the opposite of the actual cause -- upgrading is what triggers it, and
the module imports fine. ``_run_http`` raises on that flag before it can reach
its own ``_MCP_NEW_HTTP`` branch, which is fully implemented and works.

Each case runs in its own interpreter. Deciding these flags means importing
``tools.mcp_tool`` against a stubbed ``mcp`` package, and doing that in-process
leaves the real modules swapped out for anything that imports later in the same
pytest session -- which silently broke ~20 unrelated MCP tests when this file
was first written in-process.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PROBE = textwrap.dedent(
    """
    import sys, types

    def _stub(**attrs):
        m = types.ModuleType("mcp.client.streamable_http")
        for k, v in attrs.items():
            setattr(m, k, v)
        return m

    names = set(sys.argv[1].split(",")) - {""}
    mcp = types.ModuleType("mcp")
    mcp.ClientSession = object
    mcp.StdioServerParameters = object
    client = types.ModuleType("mcp.client")
    stdio = types.ModuleType("mcp.client.stdio")
    stdio.stdio_client = lambda *a, **k: None
    sse = types.ModuleType("mcp.client.sse")
    sse.sse_client = lambda *a, **k: None
    sys.modules.update({
        "mcp": mcp,
        "mcp.client": client,
        "mcp.client.stdio": stdio,
        "mcp.types": types.ModuleType("mcp.types"),
        "mcp.client.sse": sse,
        "mcp.client.streamable_http": _stub(
            **{n: (lambda *a, **k: None) for n in names}
        ),
    })

    import tools.mcp_tool as m
    print(f"{bool(m._MCP_HTTP_AVAILABLE)},{bool(m._MCP_NEW_HTTP)}")
    """
)


def _flags(*exported):
    """Import tools.mcp_tool in a fresh interpreter with these SDK symbols."""
    proc = subprocess.run(
        [sys.executable, "-c", PROBE, ",".join(exported)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stdout}\n{proc.stderr}"
    http_available, new_http = proc.stdout.strip().splitlines()[-1].split(",")
    return http_available == "True", new_http == "True"


def test_new_sdk_only_still_enables_http():
    """mcp >= 1.24.0 exports only the NEW name. This is the regression."""
    http_available, new_http = _flags("streamable_http_client")
    assert new_http is True
    assert http_available is True, (
        "HTTP transport reported unavailable on a current mcp SDK — every "
        "HTTP MCP server would be parked at startup"
    )


def test_old_sdk_only_still_enables_http():
    """Older SDKs export only the deprecated name; must keep working."""
    http_available, new_http = _flags("streamablehttp_client")
    assert new_http is False
    assert http_available is True


def test_both_names_enable_http():
    http_available, new_http = _flags("streamablehttp_client", "streamable_http_client")
    assert new_http is True
    assert http_available is True


def test_neither_name_disables_http():
    """No HTTP client at all — the flag must still go False, honestly."""
    http_available, new_http = _flags()
    assert new_http is False
    assert http_available is False
