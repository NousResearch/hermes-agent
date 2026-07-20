"""Tests for bundled skill devops/kubernetes-readonly."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest
from pydantic import ValidationError

SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "devops"
    / "kubernetes-readonly"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS))

import k8s_models  # noqa: E402
import k8s_readonly  # noqa: E402


class TestPydanticGuards:
    def test_rejects_shell_in_resource(self):
        with pytest.raises(ValidationError):
            k8s_models.OpGet.model_validate(
                {"op": "get", "resource": "pods;rm -rf /", "namespace": "default"}
            )

    def test_rejects_namespace_with_all_namespaces(self):
        with pytest.raises(ValidationError):
            k8s_models.OpGet.model_validate(
                {
                    "op": "get",
                    "resource": "pods",
                    "namespace": "kube-system",
                    "all_namespaces": True,
                }
            )

    def test_valid_get(self):
        m = k8s_models.OpGet.model_validate(
            {
                "op": "get",
                "resource": "deployments.apps",
                "namespace": "prod",
                "output": "yaml",
            }
        )
        assert m.resource == "deployments.apps"


class TestSkillFrontmatter:
    def test_description_hardline_limit(self):
        text = (SCRIPTS.parent / "SKILL.md").read_text(encoding="utf-8")
        import re

        m = re.search(r"^description:\s*(.*)$", text, re.MULTILINE)
        assert m is not None
        desc = m.group(1).strip().strip('"')
        assert len(desc) <= 60
        assert desc.endswith(".")


@mock.patch("k8s_readonly.subprocess.Popen")
@mock.patch("k8s_readonly._kubectl_bin", return_value="/bin/kubectl")
def test_run_request_success(_mock_kubectl_bin, mock_popen):
    proc = mock.Mock()
    proc.stdout = mock.Mock()
    proc.stderr = mock.Mock()
    proc.stdout.read.side_effect = [b'{"clientVersion":{}}', b""]
    proc.stderr.read.side_effect = [b"", b""]
    proc.returncode = 0
    proc.wait.return_value = 0
    mock_popen.return_value = proc

    out = k8s_readonly.run_request(k8s_models.OpVersion(op="version"))
    assert out["ok"] is True
    assert out["argv"][0] == "/bin/kubectl"
    assert out["argv"][1:] == ["version", "-o", "json"]
    assert out["truncated"] is False
    mock_popen.assert_called_once()


@mock.patch("k8s_readonly._kubectl_bin", return_value=None)
def test_run_request_no_kubectl(_mock_bin):
    out = k8s_readonly.run_request(k8s_models.OpClusterInfo(op="cluster_info"))
    assert out["ok"] is False
    assert out["error"] == "kubectl_not_found"


@mock.patch("k8s_readonly.subprocess.Popen")
@mock.patch("k8s_readonly._kubectl_bin", return_value="/bin/kubectl")
def test_top_nodes_argv(_mock_kubectl_bin, mock_popen):
    proc = mock.Mock()
    proc.stdout = mock.Mock()
    proc.stderr = mock.Mock()
    proc.stdout.read.side_effect = [b"NAME CPU\n", b""]
    proc.stderr.read.side_effect = [b"", b""]
    proc.returncode = 0
    proc.wait.return_value = 0
    mock_popen.return_value = proc

    k8s_readonly.run_request(k8s_models.OpTopNodes(op="top_nodes"))
    argv = mock_popen.call_args[0][0]
    assert argv == ["/bin/kubectl", "top", "nodes"]


@mock.patch("k8s_readonly.subprocess.Popen")
@mock.patch("k8s_readonly._kubectl_bin", return_value="/bin/kubectl")
def test_oversized_output_is_truncated_during_stream(_mock_kubectl_bin, mock_popen):
    limit = 32
    big = b"x" * (limit + 64)
    proc = mock.Mock()
    proc.stdout = mock.Mock()
    proc.stderr = mock.Mock()
    proc.stdout.read.side_effect = [big, b""]
    proc.stderr.read.side_effect = [b"", b""]
    proc.returncode = 0
    proc.wait.return_value = 0
    mock_popen.return_value = proc

    out = k8s_readonly.run_request(
        k8s_models.OpVersion(op="version"),
        max_capture=limit,
    )
    assert out["truncated"] is True
    assert len(out["stdout"].encode("utf-8")) <= limit


@mock.patch("k8s_readonly.subprocess.Popen")
@mock.patch("k8s_readonly._kubectl_bin", return_value="/bin/kubectl")
def test_timeout_returns_structured_error(_mock_kubectl_bin, mock_popen):
    proc = mock.Mock()
    proc.stdout = mock.Mock()
    proc.stderr = mock.Mock()
    proc.stdout.read.return_value = b""
    proc.stderr.read.return_value = b""
    proc.wait.side_effect = subprocess.TimeoutExpired(cmd=["kubectl"], timeout=0.01)
    proc.kill.return_value = None
    mock_popen.return_value = proc

    out = k8s_readonly.run_request(
        k8s_models.OpVersion(op="version"),
        timeout=0.01,
    )
    assert out["ok"] is False
    assert out["error"] == "kubectl_timeout"
    proc.kill.assert_called()
