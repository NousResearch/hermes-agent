"""Tests for the Hermes Global Internet GPU DePIN & Swarm Mesh."""

import io
import json
import os
import stat
import sys
import time
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from gateway.global_mesh import (
    GPUDescriptor,
    GlobalMeshCoordinator,
    MeshExecutionReceipt,
    MeshNodeInfo,
    MeshTaskRequest,
    compute_signature,
    http_transport,
    probe_local_gpu,
    verify_signature,
)


@pytest.fixture
def temp_mesh_dir(tmp_path):
    mesh_dir = tmp_path / "hermes_mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    return mesh_dir


def test_node_identity_and_keypair(temp_mesh_dir):
    coord = GlobalMeshCoordinator(state_dir=temp_mesh_dir)
    assert coord.node_id.startswith("node-")
    assert len(coord.secret_key) == 64
    assert len(coord.public_key) in (32, 64)

    # Persistence check
    coord2 = GlobalMeshCoordinator(state_dir=temp_mesh_dir)
    assert coord2.node_id == coord.node_id
    assert coord2.secret_key == coord.secret_key
    assert coord2.public_key == coord.public_key


def test_secure_file_permissions(temp_mesh_dir):
    coord = GlobalMeshCoordinator(state_dir=temp_mesh_dir)
    key_file = temp_mesh_dir / "mesh_node_secret.key"
    assert key_file.exists()

    if os.name != "nt":
        file_mode = stat.S_IMODE(key_file.stat().st_mode)
        # Verify 0o600 permissions on POSIX systems
        assert file_mode == 0o600


def test_gpu_descriptor_serialization():
    gpu = GPUDescriptor(
        device_name="NVIDIA RTX 4090",
        vram_mb=24576,
        free_vram_mb=20480,
        compute_backend="cuda",
        supported_models=["hermes-3-8b", "hermes-3-70b"],
        compute_rating=6.0,
        reputation_score=0.98,
    )
    d = gpu.to_dict()
    assert d["device_name"] == "NVIDIA RTX 4090"
    assert d["vram_mb"] == 24576

    restored = GPUDescriptor.from_dict(d)
    assert restored.device_name == gpu.device_name
    assert restored.vram_mb == gpu.vram_mb
    assert restored.supported_models == ["hermes-3-8b", "hermes-3-70b"]


def test_local_node_initialization_with_explicit_hardware(temp_mesh_dir):
    coord = GlobalMeshCoordinator(state_dir=temp_mesh_dir)
    node = coord.init_local_node(
        endpoint="https://node1.hermes.ai:8443",
        offer_gpu=True,
        vram_mb=32768,
        backend="mps",
        device_name="Apple M3 Max",
    )
    assert node.node_id == coord.node_id
    assert node.endpoint == "https://node1.hermes.ai:8443"
    assert node.gpu is not None
    assert node.gpu.vram_mb == 32768
    assert node.gpu.compute_backend == "mps"
    assert node.gpu.device_name == "Apple M3 Max"


def test_local_node_auto_probe_client_only_when_no_gpu(temp_mesh_dir):
    coord = GlobalMeshCoordinator(state_dir=temp_mesh_dir)
    # When probe returns None, auto-probing must fall back to client-only mode
    with patch("gateway.global_mesh.probe_local_gpu", return_value=None):
        node = coord.init_local_node(offer_gpu=None)
        assert node.gpu is None


def test_probe_local_gpu_cuda_mocked():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_name.return_value = "NVIDIA A100-SXM4-80GB"
    mock_props = MagicMock()
    mock_props.total_memory = 80 * 1024 * 1024 * 1024
    mock_torch.cuda.get_device_properties.return_value = mock_props
    mock_torch.cuda.mem_get_info.return_value = (70 * 1024 * 1024 * 1024, 80 * 1024 * 1024 * 1024)
    mock_torch.version.hip = None

    with patch.dict(sys.modules, {"torch": mock_torch}):
        gpu = probe_local_gpu()
        assert gpu is not None
        assert gpu.device_name == "NVIDIA A100-SXM4-80GB"
        assert gpu.vram_mb == 81920
        assert gpu.free_vram_mb == 71680
        assert gpu.compute_backend == "cuda"
        assert "hermes-3-70b" in gpu.supported_models


def test_probe_local_gpu_nvidia_smi_mocked():
    mock_res = MagicMock()
    mock_res.returncode = 0
    mock_res.stdout = "NVIDIA GeForce RTX 3080, 10240, 8500\n"

    with patch.dict(sys.modules, {"torch": None}):
        with patch("subprocess.run", return_value=mock_res):
            gpu = probe_local_gpu()
            assert gpu is not None
            assert gpu.device_name == "NVIDIA GeForce RTX 3080"
            assert gpu.vram_mb == 10240
            assert gpu.free_vram_mb == 8500
            assert gpu.compute_backend == "cuda"


def test_asymmetric_ed25519_verification(temp_mesh_dir):
    coord = GlobalMeshCoordinator(state_dir=temp_mesh_dir)
    payload = b"Delegated turn prompt instructions"

    sig = compute_signature(payload, coord.secret_key)
    # Valid signature verified with public key
    assert verify_signature(payload, sig, coord.public_key)

    # Tampered payload rejected
    assert not verify_signature(payload + b"tampered", sig, coord.public_key)

    # Invalid public key rejected
    assert not verify_signature(payload, sig, "00" * 32)


def test_peer_registration_and_heartbeat(temp_mesh_dir):
    coord = GlobalMeshCoordinator(state_dir=temp_mesh_dir)

    # Self-registration rejected
    assert not coord.register_peer(MeshNodeInfo(node_id=coord.node_id, endpoint="mesh://self"))

    peer = MeshNodeInfo(
        node_id="node-remote-4090",
        endpoint="mesh://node2.hermes.network",
        gpu=GPUDescriptor(
            device_name="RTX 4090",
            vram_mb=24576,
            free_vram_mb=24576,
            compute_backend="cuda",
            supported_models=["hermes-3-8b"],
        ),
        last_seen=time.time(),
    )
    assert coord.register_peer(peer)
    assert "node-remote-4090" in coord.peers

    # Update heartbeat
    assert coord.update_peer_heartbeat("node-remote-4090", free_vram_mb=18000)
    assert coord.peers["node-remote-4090"].gpu.free_vram_mb == 18000


def test_peer_pruning(temp_mesh_dir):
    coord = GlobalMeshCoordinator(state_dir=temp_mesh_dir)

    active_peer = MeshNodeInfo(
        node_id="peer-active",
        endpoint="mesh://active",
        last_seen=time.time(),
    )
    stale_peer = MeshNodeInfo(
        node_id="peer-stale",
        endpoint="mesh://stale",
        last_seen=time.time() - 400.0,
    )
    coord.register_peer(active_peer)
    coord.register_peer(stale_peer)

    pruned = coord.prune_inactive_peers(timeout_seconds=300.0)
    assert pruned == 1
    assert "peer-active" in coord.peers
    assert "peer-stale" not in coord.peers


def test_candidate_peer_filtering_and_ranking(temp_mesh_dir):
    coord = GlobalMeshCoordinator(state_dir=temp_mesh_dir)

    p1 = MeshNodeInfo(
        node_id="node-low-vram",
        endpoint="mesh://p1",
        gpu=GPUDescriptor(
            device_name="GTX 1060",
            vram_mb=6144,
            free_vram_mb=4096,
            compute_backend="cuda",
            supported_models=["hermes-3-8b"],
            compute_rating=1.0,
            reputation_score=0.9,
        ),
    )
    p2 = MeshNodeInfo(
        node_id="node-high-vram",
        endpoint="mesh://p2",
        gpu=GPUDescriptor(
            device_name="RTX 4090",
            vram_mb=24576,
            free_vram_mb=20480,
            compute_backend="cuda",
            supported_models=["hermes-3-8b", "hermes-3-70b"],
            compute_rating=5.0,
            reputation_score=1.0,
        ),
    )
    coord.register_peer(p1)
    coord.register_peer(p2)

    # Filter >= 8GB
    candidates = coord.find_candidate_peers(min_vram_mb=8192)
    assert len(candidates) == 1
    assert candidates[0].node_id == "node-high-vram"

    # Best peer selection
    best = coord.select_best_peer(min_vram_mb=2048)
    assert best.node_id == "node-high-vram"


def test_task_creation_and_cryptographic_signatures(temp_mesh_dir):
    coord = GlobalMeshCoordinator(state_dir=temp_mesh_dir)
    task = coord.create_task("Analyze market trends", target_model="hermes-3-70b", min_vram_mb=16384)

    assert task.task_id.startswith("mtask-")
    assert task.requester_id == coord.node_id
    assert len(task.signature) in (64, 128)

    # Verify signature using public key
    assert verify_signature(task.canonical_bytes(), task.signature, coord.public_key)


def test_local_execution_and_receipt_proof(temp_mesh_dir):
    coord = GlobalMeshCoordinator(state_dir=temp_mesh_dir)
    coord.init_local_node(offer_gpu=True, vram_mb=16384)

    task = coord.create_task("What is quantum computing?", min_vram_mb=8192)
    receipt = coord.execute_task_locally(task)

    assert receipt.status == "completed"
    assert receipt.executor_id == coord.node_id
    assert "quantum" in receipt.output
    # Verify receipt using executor public key
    assert coord.verify_receipt(receipt, coord.public_key)


def test_http_transport_success():
    peer = MeshNodeInfo(node_id="peer-remote", endpoint="https://peer.remote.ai:8443")
    task = MeshTaskRequest(
        task_id="mtask-test-1",
        requester_id="node-local",
        target_model="hermes-3-8b",
        min_vram_mb=8192,
        payload="Say hello",
    )

    expected_receipt = {
        "task_id": "mtask-test-1",
        "executor_id": "peer-remote",
        "status": "completed",
        "output": "Hello from peer",
        "tokens": 12,
        "latency_ms": 32.5,
        "proof_sig": "abcdef",
    }

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(expected_receipt).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        receipt = http_transport(peer, task)
        assert receipt.status == "completed"
        assert receipt.executor_id == "peer-remote"
        assert receipt.output == "Hello from peer"

        # Verify request parameters
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.get_full_url() == "https://peer.remote.ai:8443/mesh/task"
        assert req.get_method() == "POST"
        assert req.get_header("Content-type") == "application/json"
        assert req.get_header("X-hermes-requester-id") == "node-local"


def test_http_transport_network_error():
    peer = MeshNodeInfo(node_id="peer-offline", endpoint="https://offline.node.ai:8443")
    task = MeshTaskRequest(
        task_id="mtask-test-2",
        requester_id="node-local",
        target_model="hermes-3-8b",
        min_vram_mb=8192,
        payload="Say hello",
    )

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
        receipt = http_transport(peer, task)
        assert receipt.status == "failed"
        assert "Connection refused" in receipt.error_message


def test_rendezvous_sync(temp_mesh_dir):
    coord = GlobalMeshCoordinator(
        state_dir=temp_mesh_dir,
        rendezvous_url="https://rendezvous.hermes.ai",
    )
    coord.init_local_node(endpoint="https://my.node:8443", offer_gpu=False)

    peers_payload = {
        "peers": [
            {
                "node_id": "peer-synced-1",
                "endpoint": "https://synced1.mesh:8443",
                "gpu": {
                    "device_name": "A100",
                    "vram_mb": 81920,
                    "free_vram_mb": 65536,
                    "compute_backend": "cuda",
                    "supported_models": ["hermes-3-70b"],
                },
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(peers_payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        success, peers = coord.sync_rendezvous()
        assert success is True
        assert len(peers) == 1
        assert "peer-synced-1" in coord.peers


def test_delegation_with_failover_and_reputation(temp_mesh_dir):
    coord = GlobalMeshCoordinator(state_dir=temp_mesh_dir)

    flaky_peer = MeshNodeInfo(
        node_id="node-flaky",
        endpoint="mesh://flaky",
        gpu=GPUDescriptor(
            device_name="RTX 3090",
            vram_mb=24576,
            free_vram_mb=20000,
            supported_models=["hermes-3-8b"],
            compute_rating=4.0,
            reputation_score=1.0,
        ),
    )
    solid_peer = MeshNodeInfo(
        node_id="node-solid",
        endpoint="mesh://solid",
        gpu=GPUDescriptor(
            device_name="RTX 4090",
            vram_mb=24576,
            free_vram_mb=19000,
            supported_models=["hermes-3-8b"],
            compute_rating=3.9,
            reputation_score=1.0,
        ),
    )
    coord.register_peer(flaky_peer)
    coord.register_peer(solid_peer)

    def mock_transport(peer: MeshNodeInfo, task: MeshTaskRequest) -> MeshExecutionReceipt:
        if peer.node_id == "node-flaky":
            return MeshExecutionReceipt(
                task_id=task.task_id,
                executor_id=peer.node_id,
                status="failed",
                output="",
                tokens=0,
                latency_ms=10.0,
                error_message="GPU OOM during kernel launch",
            )
        return MeshExecutionReceipt(
            task_id=task.task_id,
            executor_id=peer.node_id,
            status="completed",
            output="Solid peer completed prompt execution",
            tokens=42,
            latency_ms=35.0,
        )

    receipt = coord.delegate_task("Complex reasoning task", min_vram_mb=8192, transport_fn=mock_transport)

    assert receipt.status == "completed"
    assert receipt.executor_id == "node-solid"
    # Flaky peer reputation penalized
    assert coord.peers["node-flaky"].gpu.reputation_score < 1.0
    # Solid peer reputation rewarded
    assert coord.peers["node-solid"].gpu.reputation_score == 1.0


def test_cluster_summary_aggregation(temp_mesh_dir):
    coord = GlobalMeshCoordinator(state_dir=temp_mesh_dir)
    coord.init_local_node(offer_gpu=True, vram_mb=8192)

    peer = MeshNodeInfo(
        node_id="peer-1",
        endpoint="mesh://p1",
        gpu=GPUDescriptor(device_name="A100", vram_mb=81920, free_vram_mb=65536),
    )
    coord.register_peer(peer)

    summary = coord.get_mesh_summary()
    assert summary["total_peers"] == 1
    assert summary["active_peers"] == 1
    assert summary["cluster_vram_mb"] == 81920
    assert summary["cluster_free_vram_mb"] == 65536
