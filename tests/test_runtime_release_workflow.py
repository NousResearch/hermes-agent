"""Regression tests for the signed desktop runtime release workflow.

The desktop runtime is a frozen PyInstaller executable, so it CANNOT
lazy-install dependencies at first use (no working pip inside the binary).
Every backend the desktop exposes must therefore be pre-baked via the
``cn-desktop`` aggregate extra and collected by PyInstaller. These tests pin
that contract so a backend can't silently drop out of the build again —?the
failure mode behind issue #16 (MCP) and the 飞书/钉钉/企微/微信 desktop reports.
"""

import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - CI runs 3.11+
    tomllib = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _workflow_text() -> str:
    workflow = _repo_root() / ".github" / "workflows" / "release-runtime.yml"
    return workflow.read_text(encoding="utf-8")


def _cn_desktop_extra() -> list[str]:
    if tomllib is None:  # pragma: no cover
        pytest.skip("tomllib unavailable")
    data = tomllib.loads((_repo_root() / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]["cn-desktop"]


def _frozen_verify_packages() -> set[str]:
    """The package names whose dist-info the workflow asserts in the frozen output.

    Parses the ``for pkg in ... ; do`` loop in the "Verify frozen runtime
    backends" step (continuation backslashes stripped) into a set of names.
    """
    text = _workflow_text()
    start = text.index("for pkg in ") + len("for pkg in ")
    body = text[start : text.index("; do", start)]
    return set(body.replace("\\", "").split())


def test_runtime_workflow_installs_cn_desktop_extra():
    """The frozen runtime installs the cn-desktop aggregate extra.

    cn-desktop is the single source of truth for "what the desktop ships".
    """
    assert 'pip install -e ".[cn-desktop]"' in _workflow_text()


def test_cn_desktop_extra_bundles_every_desktop_backend():
    """cn-desktop must pre-bake all backends the frozen runtime exposes."""
    extra = _cn_desktop_extra()
    blob = "\n".join(extra)
    # Aggregated sub-extras
    for sub in ("web", "anthropic", "mcp", "feishu", "dingtalk", "wecom"):
        assert f"[{sub}]" in blob, f"cn-desktop is missing the {sub} extra"
    # 微信 (weixin) has no dedicated extra —?its adapter deps are listed directly
    assert any(e.startswith("aiohttp") for e in extra), "cn-desktop missing aiohttp (微信/feishu/wecom)"
    assert any(e.startswith("qrcode") for e in extra), "cn-desktop missing qrcode (scan-login)"
    assert any(e.startswith("cryptography") for e in extra), "cn-desktop missing cryptography (微信/wecom AES)"


def test_runtime_workflow_freezes_anthropic_sdk_for_minimax_cn():
    """MiniMax-CN rides the Anthropic Messages transport; the SDK must ship."""
    workflow = _workflow_text()
    assert any("[anthropic]" in e for e in _cn_desktop_extra())
    assert "--collect-submodules anthropic" in workflow
    assert "--copy-metadata anthropic" in workflow
    assert "anthropic" in _frozen_verify_packages()


def test_runtime_workflow_freezes_native_mcp_client():
    """Issue #16: the native MCP client SDK must ship in the frozen runtime."""
    workflow = _workflow_text()
    assert any("[mcp]" in e for e in _cn_desktop_extra())
    assert "--collect-submodules mcp" in workflow
    assert "--copy-metadata mcp" in workflow
    assert "mcp" in _frozen_verify_packages()


def test_runtime_workflow_freezes_im_platform_backends():
    """飞书/钉钉/企微/微信 adapters must ship —?they can't lazy-install frozen."""
    workflow = _workflow_text()
    # Feishu / DingTalk SDKs collected by PyInstaller
    for mod in ("lark_oapi", "dingtalk_stream", "alibabacloud_dingtalk"):
        assert f"--collect-submodules {mod}" in workflow, f"missing --collect-submodules {mod}"
    # Shared adapter deps (Feishu webhook / WeCom / 微信)
    assert "--collect-submodules aiohttp" in workflow
    assert "--collect-submodules qrcode" in workflow
    # dist-info asserted in the frozen output
    verified = _frozen_verify_packages()
    for pkg in ("lark_oapi", "dingtalk_stream", "alibabacloud_dingtalk", "aiohttp", "qrcode", "defusedxml"):
        assert pkg in verified, f"verify step doesn't assert {pkg} dist-info"


def test_runtime_workflow_verifies_backends_in_build_env_and_frozen_output():
    """The workflow fails fast if a backend's SDK is missing.

    Two gates: a build-env import smoke test (catches a missing extra dep) and a
    dist-info assert over the frozen output (catches a PyInstaller collect miss).
    """
    workflow = _workflow_text()
    assert "Verify platform backends importable (build env)" in workflow
    assert "FEISHU_AVAILABLE" in workflow
    assert "Verify frozen runtime backends" in workflow
    assert ".dist-info" in workflow


def test_runtime_workflow_freezes_hindsight_client_for_long_term_memory():
    """The CN desktop frozen runtime pre-bakes the Hindsight memory client.

    mirrors test_runtime_workflow_freezes_native_mcp_client: cn-desktop extra
    pulls in [hindsight], the PyInstaller build collects hindsight_client and
    copies its dist metadata, and the frozen-output verify step asserts the
    hindsight_client-*.dist-info directory is present. A missing piece here
    means hermes_recall/reflect/retain fail with ``ModuleNotFoundError`` inside
    the frozen binary (issue: hindsight-client not in PyInstaller bundle).
    """
    workflow = _workflow_text()
    extra = _cn_desktop_extra()

    # 1. cn-desktop must transitively pull in the [hindsight] sub-extra.
    assert any("[hindsight]" in e for e in extra), (
        "cn-desktop extra is missing hermes-agent[hindsight] "
        "(frozen runtime cannot lazy-install hindsight-client)"
    )

    # 2. Build-env import smoke test must include hindsight_client.
    #    Re-parse the same for-tuple the existing helper does, so a future
    #    re-shuffle of the import list still keeps this assertion honest.
    text = workflow
    start = text.index("for m in (") + len("for m in (")
    body = text[start : text.index("):", start)]
    import_list = body.replace("\n", " ").replace('"', " ").split()
    assert "hindsight_client" in import_list, (
        "release-runtime.yml build-env import smoke test does not "
        "check hindsight_client; PyInstaller may bundle a missing SDK."
    )

    # 3. PyInstaller collect / metadata lines.
    assert "--collect-submodules hindsight_client" in workflow, (
        "PyInstaller invocation missing --collect-submodules hindsight_client"
    )
    assert "--copy-metadata hindsight-client" in workflow, (
        "PyInstaller invocation missing --copy-metadata hindsight-client "
        "(pip distribution name with hyphen, not the underscored module name)"
    )

    # 4. Frozen-output verify list must assert the dist-info directory.
    verified = _frozen_verify_packages()
    assert "hindsight_client" in verified, (
        "frozen-output verify list does not assert hindsight_client dist-info"
    )

def test_runtime_workflow_freezes_hindsight_client_api_and_aiohttp_retry():
    """hindsight-client==0.6.1 ships hindsight_client_api as a bundled top-level
    package and depends on aiohttp-retry as an independent distribution. The
    previous PR (#39 round 1) only collected hindsight_client itself, which
    proved insufficient in a real frozen runtime smoke test: the recall tool
    returned HTTP 500 with
        No module named 'hindsight_client_api'
    after the first round, and
        No module named 'aiohttp_retry'
    after the second round. Both transitive pieces must be explicitly
    collected and the independent aiohttp-retry dist-info must be copied.
    """
    workflow = _workflow_text()
    extra = _cn_desktop_extra()

    # 1. cn-desktop extra still pulls in [hindsight] (round 1 contract).
    assert any("[hindsight]" in e for e in extra)

    # 2. Build-env import smoke test must cover all three modules.
    text = workflow
    start = text.index("for m in (") + len("for m in (")
    body = text[start : text.index("):", start)]
    import_list = body.replace("\n", " ").replace('"', " ").split()
    for mod in ("hindsight_client", "hindsight_client_api", "aiohttp_retry"):
        assert mod in import_list, (
            f"release-runtime.yml build-env import smoke test does not "
            f"check {mod}; PyInstaller may bundle a missing dependency."
        )

    # 3. PyInstaller must collect all three submodules.
    #    hindsight_client_api is bundled inside the same wheel as
    #    hindsight_client (verified by importlib.metadata.packages_distributions()
    #    mapping both modules to the same "hindsight-client" distribution) but
    #    PyInstaller static analysis does not automatically pick up sibling
    #    top-level packages, so it must be named explicitly.
    for mod in ("hindsight_client", "hindsight_client_api", "aiohttp_retry"):
        assert f"--collect-submodules {mod}" in workflow, (
            f"PyInstaller invocation missing --collect-submodules {mod}"
        )

    # 4. PyInstaller must copy the independent dist-info.
    #    - hindsight-client: required (covers hindsight_client + hindsight_client_api)
    #    - aiohttp-retry: required (independent distribution)
    #    - hindsight-client-api: NOT a real distribution, must NOT be added
    assert "--copy-metadata hindsight-client" in workflow, (
        "PyInstaller invocation missing --copy-metadata hindsight-client"
    )
    assert "--copy-metadata aiohttp-retry" in workflow, (
        "PyInstaller invocation missing --copy-metadata aiohttp-retry "
        "(aiohttp-retry is an independent distribution, not bundled inside "
        "hindsight-client)"
    )
    assert "--copy-metadata hindsight-client-api" not in workflow, (
        "release-runtime.yml must NOT pass --copy-metadata hindsight-client-api: "
        "hindsight_client_api has no independent distribution metadata; it "
        "shares hindsight_client-*.dist-info with the main hindsight_client "
        "package (verified via importlib.metadata.packages_distributions()). "
        "Adding a non-existent --copy-metadata is a build-time typo."
    )

    # 5. Frozen-output verify list must assert the dist-info directory of
    #    every independent distribution. aiohttp_retry is independent, so
    #    its aiohttp_retry-*.dist-info must be asserted; hindsight_client_api
    #    shares the hindsight_client-*.dist-info, so the single hindsight_client
    #    entry already covers it.
    # Frozen-output verify list asserts dist-info directories of INDEPENDENT
    # distributions only. hindsight_client_api has no independent distribution;
    # it shares hindsight_client-*.dist-info with the main hindsight_client
    # package (verified by importlib.metadata.packages_distributions()), so
    # putting "hindsight_client_api" in the verify list would either be a
    # silent no-op (find would not match anything) or, worse, mask a real
    # build break. Only hindsight_client and aiohttp_retry are asserted here.
    verified = _frozen_verify_packages()
    for pkg in ("hindsight_client", "aiohttp_retry"):
        assert pkg in verified, (
            f"frozen-output verify list does not assert {pkg} dist-info"
        )
    assert "hindsight_client_api" not in verified, (
        "frozen-output verify list must NOT contain hindsight_client_api: "
        "it has no independent distribution metadata. Listing it here is "
        "either a silent no-op (the find pattern would never match) or a "
        "future regression hiding the real dist-info directory. The "
        "hindsight_client entry already covers it because both modules share "
        "hindsight_client-*.dist-info."
    )


def test_runtime_workflow_signs_and_preserves_macos_frameworks():
    workflow = _workflow_text()

    assert "Normalize macOS framework layout" in workflow
    assert "scripts/normalize_macos_pyinstaller_runtime.py" in workflow
    assert "Prepare macOS signing credentials" in workflow
    assert "scripts/sign_macos_runtime_payload.sh" in workflow
    assert "zip -r -y" in workflow
