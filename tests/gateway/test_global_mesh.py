"""Tests for the Hermes Global Internet GPU DePIN & Swarm Mesh."""

import os
import time
from pathlib import Path
import pytest

from gateway.global_mesh import (
    GPUDescriptor,
    GlobalMeshCoordinator,
    MeshExecutionReceipt,
    MeshNodeInfo,
    MeshTaskRequest,
    compute_signature,
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

    # Persistence check
    coord2 = GlobalMeshCoordinator(state_dir=temp_mesh_dir)
    assert coord2.node_id == coord.node_id
    assert coord2.secret_key == coord.secret_key


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


def test_local_node_initialization(temp_mesh_dir):
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
    assert len(task.signature) == 64

    # Verify signature
    expected = compute_signature(task.canonical_bytes(), coord.secret_key)
    assert task.signature == expected


def test_local_execution_and_receipt_proof(temp_mesh_dir):
    coord = GlobalMeshCoordinator(state_dir=temp_mesh_dir)
    coord.init_local_node(offer_gpu=True, vram_mb=16384)

    task = coord.create_task("What is quantum computing?", min_vram_mb=8192)
    receipt = coord.execute_task_locally(task)

    assert receipt.status == "completed"
    assert receipt.executor_id == coord.node_id
    assert "quantum" in receipt.output
    assert coord.verify_receipt(receipt, coord.secret_key)


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

    # Mock transport where flaky_peer fails, triggering failover to solid_peer
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
